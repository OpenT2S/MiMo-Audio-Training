# Copyright 2025 Xiaomi Corporation.
import os
import re
import time
import random
import torch
import torchaudio
import soundfile as sf

from typing import Union
from torchaudio.transforms import MelSpectrogram
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from peft import get_peft_model
from mimo_audio_train.arguments import CustomArguments

from .src_mimo_audio.process_speechdata import InputSegment, StreamingInputSegment
from .src_mimo_audio.mimo_audio_tokenizer import MiMoAudioTokenizer
from .src_mimo_audio.modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioForCausalLM,
    MiMoSampler,
    MiMoStopper,
)


def detect_language(text):
    if re.search(r'[\u4e00-\u9fff]', text):
        return 'zh'
    else:
        return 'en'


class MimoAudioModel:

    def __init__(
        self,
        model_path: str,
        mimo_audio_tokenizer_path: str,
        lora_path: str = None,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.path
        )
        self.padding_idx = int(self.tokenizer.pad_token_id)

        special_tokens = [
            "<|sosp|>",
            "<|eosp|>",
            "<|empty|>",
            "<|Human|>",
            "<|SpeechLM|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
        ]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"Add special tokens {token} to tokenizer.vocab")
                self.tokenizer.add_tokens([token], special_tokens=True)

        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")
        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.sostm_idx = self.tokenizer.convert_tokens_to_ids("<|sostm|>")
        self.eostm_idx = self.tokenizer.convert_tokens_to_ids("<|eostm|>")
        self.eot_idx = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        self.im_start_idx = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.speech_loss_weights = CustomArguments.speech_loss_weights

        model_args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
            speech_loss_weights=self.speech_loss_weights,
        )

        start_loading_time = time.monotonic()
        self.model = MiMoAudioForCausalLM.from_pretrained(
            self.path,
            args=model_args,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        if lora_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path, device_map={"": self.device})

        self.group_size=self.model.config.group_size
        self.audio_channels=self.model.config.audio_channels
        self.delay_pattern = self.model.config.delay_pattern
        self.vocab_size = self.model.config.vocab_size

        self.speech_zeroemb_idx = self.model.speech_empty_ids

        self.model.eval()
        print(
            f"Model loaded in {time.monotonic() - start_loading_time:.2f} seconds, device: {self.device}"
        )

        self.generate_kwargs = {
            "max_length": 8192,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.default_global_sampler = MiMoSampler(
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95
        )
        self.default_local_sampler = MiMoSampler(
            do_sample=True, temperature=0.9, top_k=50, top_p=0.95
        )
        
        self.task_sampler_configs = {
            "asr": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "tts": {
                "global": MiMoSampler(do_sample=True, temperature=0.6, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "qa": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "mmlu": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "spoken_dialogue": {
                "global": MiMoSampler(do_sample=True, temperature=0.3, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.7, top_p=0.95)
            },
            "audio_understanding": {
                "global": MiMoSampler(do_sample=True, temperature=0.3, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "text_chat": {
                "global": MiMoSampler(do_sample=True, temperature=0.4, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
            "in_context_learning_s2s": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
        }

        start_loading_mimo_audio_tokenizer_time = time.monotonic()
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(self.mimo_audio_tokenizer_path)
            
        self.mimo_audio_tokenizer.eval().bfloat16().to(self.device)
        print(
            f"MiMo-Audio Tokenizer loaded in {time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds, device: {self.device}"
        )
        
        # Initialize mel spectrogram transform for consistent processing
        self.mel_transform = MelSpectrogram(
            sample_rate=self.mimo_audio_tokenizer.config.sampling_rate,
            n_fft=self.mimo_audio_tokenizer.config.nfft,
            hop_length=self.mimo_audio_tokenizer.config.hop_length,
            win_length=self.mimo_audio_tokenizer.config.window_size,
            f_min=self.mimo_audio_tokenizer.config.fmin,
            f_max=self.mimo_audio_tokenizer.config.fmax,
            n_mels=self.mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        ).to(self.device)
        
        self.history = None
    
    def get_task_sampler(self, task_name):
        if task_name not in self.task_sampler_configs:
            return {
                "global": self.default_global_sampler,
                "local": self.default_local_sampler
            }
        return self.task_sampler_configs[task_name]
    
    def save_wav(self, path, wav):
        sf.write(
            path,
            wav.reshape(-1).detach().cpu().numpy(),
            24000,
        )
    
    def wav2mel(self, wav):
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()
    
    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor, original_sr, target_sr
            )
        return wav_tensor
        
    def group_by_length(self, features: torch.Tensor, lengths: torch.Tensor, max_length: int):
        if features.size(0) != lengths.sum().item():
            raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")
        
        split_points = []
        current_sum = 0
        
        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()
        
        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)
        
        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)
        
        return feature_groups, len_groups
    
    def encode_batch(self, input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
        feature_groups, len_groups = self.group_by_length(input_features, input_lens, max_length)
        
        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features.to(self.model.device), 
                    input_lens=lengths.to(self.model.device), 
                    return_codes_only=True
                )
                encoded_parts.append(codes)
        
        return torch.cat(encoded_parts, dim=-1)

    def preprocess_input(
        self,
        input: Union[None, str, torch.Tensor] = None,
    ):
        if isinstance(input, torch.Tensor) or (isinstance(input, str) and os.path.isfile(input)):
            if isinstance(input, torch.Tensor):
                wav = input
            else:
                wav, sr = torchaudio.load(input)
                if wav.ndim == 2:
                    wav = wav.mean(dim=0)
                wav = self.resample_audio_if_needed(wav, sr)
            wav = wav.to(self.device)
            
            mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)

            input_len = mel.size(0)
            segment_size = 3000
            input_len_seg = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                input_len_seg.append(input_len % segment_size)

            codes_packed = self.encode_batch(
                input_features=mel, 
                input_lens=torch.tensor(input_len_seg),
            )
            
            codes = codes_packed.transpose(0, 1).detach().cpu()
            audio_codes = codes[:, :self.audio_channels]

            # Pad the sequence to be a multiple of group_size by repeating the last frame
            num_timesteps = audio_codes.shape[0]
            if num_timesteps % self.group_size != 0:
                padding_needed = self.group_size - (num_timesteps % self.group_size)
                last_tokens = audio_codes[-1:, :] # Keep dim for repeat
                padding_tokens = last_tokens.repeat(padding_needed, 1)
                audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)
            
            audio_tokenized = audio_codes.reshape(-1)

            return audio_tokenized
        else:
            text = input
            if (
                text.isupper() or text.islower()
            ):  # If the text only contains upper-case or lower-case letters, capitalize it.
                text = text.capitalize()
            return text
        
    def get_input_ids(self, prompt):
        input_ids = [
            seg.to_input_id(
                self.tokenizer, 
                self.group_size, 
                self.audio_channels,
            )
            for seg in prompt
        ]
        input_ids = torch.cat(input_ids, dim=1)
        return input_ids.to(self.device)
    
    @torch.no_grad()
    def in_context_learning_s2s(self, prompt_examples, audio, output_path=None):
        prompt = [
            InputSegment(
                text="[Int]:",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        
        for i in range(len(prompt_examples)):
            prompt.append(InputSegment(
                audio=self.preprocess_input(prompt_examples[i]["audio"]),
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ))
            
            prompt.append(InputSegment(
                    text="\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
            ))
            
            prompt.append(StreamingInputSegment(
                text=prompt_examples[i]["label_transcription"],
                audio=self.preprocess_input(prompt_examples[i]["label"]),
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
                tokenizer=self.tokenizer, 
                group_size=self.group_size, 
                audio_channels=self.audio_channels,
            ))
                    
            prompt.append(InputSegment(
                text=" \n\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ))


        prompt.append(InputSegment(
            audio=self.preprocess_input(audio),
            speech_zeroemb_idx=self.speech_zeroemb_idx,
            text_zeroemb_idx=self.empty_token,
        ))
            
        prompt.append(InputSegment(
            text="\n",
            speech_zeroemb_idx=self.speech_zeroemb_idx,
            text_zeroemb_idx=self.empty_token,
        ))
                

        prompt.append(InputSegment(
            text="<|sostm|>",
            speech_zeroemb_idx=self.speech_zeroemb_idx,
            text_zeroemb_idx=self.empty_token,
        ))  
            
        
        input_ids = self.get_input_ids(prompt)

                
        stopping_criteria = ([
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels
            )
        ])

        self.forward(input_ids, output_audio_path=output_path, stopping_criteria=stopping_criteria, task_name="in_context_learning_s2s")

    def get_asr_sft_prompt(
        self,
        input: Union[None, str] = None,
        lang='zh'
    ):
        audio_tokenized = self.preprocess_input(input)

        if lang == "zh":
            template = "请将这段语音转换为文字."
        else:
            template = "Transcribe the following voice message."
        
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=template,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<think>\n\n</think>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_tts_sft_prompt(
        self,
        input: Union[None, str] = None,
        instruct=None,
        read_text_only=True,
        prompt_speech=None,
    ):
        if prompt_speech is not None:
            assistant_prompt_audio_token = self.preprocess_input(prompt_speech)
        else:
            assistant_prompt_audio_token = None
        if not read_text_only:
            text = self.preprocess_input(input)
            if assistant_prompt_audio_token is None:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"你需要根据指定的风格指令和文本内容来生成语音。",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="",
                        audio=assistant_prompt_audio_token,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
        else:
            language = detect_language(input)
            if language == "zh":
                template = "请将这段文字转换为语音"
            else:
                template = "Please convert this text to speech"

            text = self.preprocess_input(input)
            if instruct is None:
                lm_prompt = [
                    InputSegment(
                        text=f"<|im_start|>user\n{template}: {text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<|sostm|>",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                if assistant_prompt_audio_token is None:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"你需要根据指定的风格指令和文本内容来生成语音。",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]
                else:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="",
                            audio=assistant_prompt_audio_token,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]
  
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids
    
    def get_few_shot_prompts(
        self,
        prompts,
        output_type="text",
    ):
        lm_prompt = []
        is_interleaved = False
        data_type = [prompt[1] for prompt in prompts]
        if "speech" in data_type and "text" in data_type:
            is_interleaved = True
        if is_interleaved:
            lm_prompt.append(
                InputSegment(
                    text="[Int]:",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            )
        else:
            if all(t == "speech" for t in data_type):
                lm_prompt.append(
                    InputSegment(
                        text="<Spe>:",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        for prompt in prompts:
            if prompt[1] == "speech":
                audio_tokenized = self.preprocess_input(str(prompt[0]["speech"]))
                if prompt[2] == "input":
                    lm_prompt.append(
                        InputSegment(
                            audio=audio_tokenized,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                else:
                    lm_prompt.append(
                        StreamingInputSegment(
                            text=prompt[0]["text"],
                            audio=audio_tokenized,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                            tokenizer=self.tokenizer, 
                            group_size=self.group_size, 
                            audio_channels=self.audio_channels,
                        )
                    )
            else:
                lm_prompt.append(
                    InputSegment(
                        text=prompt[0]["text"],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        if output_type == "speech":
            lm_prompt.append(
                InputSegment(
                    text="<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def format_instruction_template(self, conv_list, append_generation_prompt=False, thinking=False):
        user_prompt = "<|im_start|>user\n"
        assistant_prompt = "<|im_start|>assistant\n"
        end_prompt = "<|im_end|>\n"
        sound_token = "<sound>"
        lm_prompt = []

        for conv in conv_list:
            if conv['from'] == 'human':
                lm_prompt.append(
                    InputSegment(
                        text=user_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                sound_token_nums = conv['value'][0]['value'].count(sound_token)

                if sound_token_nums > 1:
                    text_parts_list = conv['value'][0]['value'].split(sound_token)
                    audio_paths_list = conv['value'][1]['value']
                    for i in range(sound_token_nums):
                        lm_prompt.append(
                            InputSegment(
                                text=text_parts_list[i],
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                        audio_tokens = self.preprocess_input(audio_paths_list[i])
                        lm_prompt.append(
                            InputSegment(
                                audio=audio_tokens,
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                    lm_prompt.append(
                        InputSegment(
                            text=text_parts_list[-1],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                elif sound_token in conv['value'][0]['value']:
                    left_part, right_part = conv['value'][0]['value'].split(sound_token)
                    if left_part != "":
                        lm_prompt.append(
                            InputSegment(
                                text=left_part,
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                    if conv['value'][1]['type'] == 'sound':
                        audio_tokens = self.preprocess_input(conv['value'][1]['value'])
                    elif conv['value'][1]['type'] == 'token':
                        audio_tokens = torch.tensor(conv['value'][1]['value'])
                    else:
                        raise ValueError(f"Invalid input type: {conv['value'][1]['type']}")
                    lm_prompt.append(
                        InputSegment(
                            audio=audio_tokens,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                    if right_part != "":
                        lm_prompt.append(
                            InputSegment(
                                text=right_part,
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                else:
                    lm_prompt.append(
                        InputSegment(
                            text=conv['value'][0]['value'],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                if thinking:
                    lm_prompt.append(
                        InputSegment(
                            text="You can think about the question briefly and then answer it.",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                lm_prompt.append(
                    InputSegment(
                        text=end_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
            else:
                lm_prompt.append(
                    InputSegment(
                        text=assistant_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                if sound_token in conv['value'][0]['value']:
                    left_part, right_part = conv['value'][0]['value'].split(sound_token)
                    if left_part != "":
                        lm_prompt.append(
                            InputSegment(
                                text=left_part,
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                    if conv['value'][1]['type'] == 'sound':
                        audio_tokens = self.preprocess_input(conv['value'][1]['value'])
                    elif conv['value'][1]['type'] == 'token':
                        audio_tokens = torch.tensor(conv['value'][1]['value'])
                    else:
                        raise ValueError(f"Invalid input type: {conv['value'][1]['type']}")
                    lm_prompt.append(
                        InputSegment(
                            audio=audio_tokens,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                    if right_part != "":
                        lm_prompt.append(
                            InputSegment(
                                text=right_part,
                                speech_zeroemb_idx=self.speech_zeroemb_idx,
                                text_zeroemb_idx=self.empty_token,
                            )
                        )
                else:
                    lm_prompt.append(
                        InputSegment(
                            text=conv['value'][0]['value'],
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        )
                    )
                lm_prompt.append(
                    InputSegment(
                        text=end_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )

        if append_generation_prompt:
            lm_prompt.append(
                InputSegment(
                    text=assistant_prompt,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
            if not thinking:
                lm_prompt.append(
                    InputSegment(
                        text="<think>\n\n</think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
            else:
                lm_prompt.append(
                    InputSegment(
                        text="<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_qa_prompt(
        self,
        input_speech,
        input_text,
    ):
        audio_tokenized = self.preprocess_input(input_speech)
        lm_prompt = [
            InputSegment(
                text=f"[Int]:",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if input_text != "":
            lm_prompt.append(
                InputSegment(
                    text=f"Question: {input_text}",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        lm_prompt.append(
            InputSegment(
                text="\nAnswer:",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids
    
    def get_audio_understanding_sft_prompt(
        self,
        input_speech,
        input_text,
        thinking=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)
           
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text + ("You can think about the question briefly and then answer it." if thinking else ""),
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids
    
    def get_spoken_dialogue_sft_prompt(
        self,
        input_speech,
        assistant_character="乐于助人",
        assistant_style="像个助手",
        prompt_speech=None,
        add_history=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)
        if add_history and self.history is not None:
            lm_prompt = [
                InputSegment(
                    text=f"<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]
        else:
            if prompt_speech is not None:
                assistant_prompt_audio_token = self.preprocess_input(prompt_speech)
            else:
                assistant_prompt_audio_token = None
            lm_prompt = [
                InputSegment(
                    text="<|im_start|>system\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]
            if assistant_prompt_audio_token is not None:
                lm_prompt += [
                    InputSegment(
                        text=f"Your voice should be:",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="",
                        audio=assistant_prompt_audio_token,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            lm_prompt += [
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]
        
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_spoken_dialogue_sft_multiturn_prompt(
        self,
        input_speech,
        text_dialogue,
        assistant_character="乐于助人",
        assistant_style="像个助手",
        prompt_speech=None,
        add_history=False,
    ):
        if prompt_speech is not None:
            assistant_prompt_audio_token = self.preprocess_input(prompt_speech)
        else:
            assistant_prompt_audio_token = None
        lm_prompt = [
            InputSegment(
                text="<|im_start|>system\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if assistant_prompt_audio_token is not None:
            lm_prompt += [
                InputSegment(
                    text=f"Your voice should be:",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="",
                    audio=assistant_prompt_audio_token,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            ]

        length = len(input_speech)
        for i in range(length):
            if input_speech[i]['role'] == 'user':
                audio_tokenized = self.preprocess_input(input_speech[i]['content'])
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=audio_tokenized,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
            else:
                audio_tokenized = self.preprocess_input(input_speech[i]['content'])
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    StreamingInputSegment(
                        text=text_dialogue[i]['content'],
                        audio=audio_tokenized,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                        tokenizer=self.tokenizer, 
                        group_size=self.group_size, 
                        audio_channels=self.audio_channels,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
        lm_prompt += [
            InputSegment(
                text=f"<|im_start|>assistant\n<|sostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_s2t_dialogue_sft_prompt(
        self,
        input_speech,
        thinking=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_s2t_dialogue_sft_multiturn_prompt(
        self,
        input_speech,
        text_dialogue,
        assistant_character="乐于助人",
        assistant_style="像个助手",
        add_history=False,
    ):
        length = len(input_speech)
        lm_prompt = []
        for i in range(length):
            if input_speech[i]['role'] == 'user':
                audio_tokenized = self.preprocess_input(input_speech[i]['content'])
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        audio=audio_tokenized,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
            else:
                text = text_dialogue[i]['content']
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text="<think>\n\n</think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=text,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        lm_prompt.append(
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        lm_prompt.append(
            InputSegment(
                text="<think>\n\n</think>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_text_dialogue_sft_prompt(
        self,
        input_text,
        thinking=False,
    ):
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids
    
    def get_text_dialogue_sft_multiturn_prompt(
        self,
        input_text,
        thinking=False,
    ):
        lm_prompt = []
        for i in range(len(input_text)):
            if input_text[i]['role'] == 'user':
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=input_text[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
            else:
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=input_text[i]['content'],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    @torch.no_grad()              
    def forward(
        self,
        input_ids,
        return_audio=False,
        output_audio_path=None,
        stopping_criteria=None,
        min_new_tokens=0,
        max_new_tokens=8192,
        add_history=False,
        task_name=None,
    ):

        task_sampler = self.get_task_sampler(task_name)
        
        generation_kwargs = self.generate_kwargs.copy()
        generation_config = GenerationConfig(**generation_kwargs)

        input_ids = input_ids.T.reshape(1, -1) # [B, flattened(T, audio_channels + 1)]
        if add_history and self.history is not None:
            input_ids = torch.cat([self.history, input_ids], dim=1)

        prompt_length = input_ids.shape[1] // (self.audio_channels+1)
        
        max_length = prompt_length // self.group_size + max_new_tokens
        min_length = prompt_length // self.group_size + min_new_tokens
        
        if stopping_criteria is not None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MiMoStopper):
                    criterion.max_length = max_length
                    criterion.min_length = min_length

        generated_ids = self.model.generate(
            input_ids,
            generation_config,
            stopping_criteria=stopping_criteria,
            global_sampler=task_sampler["global"],
            local_sampler=task_sampler["local"],
        )
        
        self.history = generated_ids
        generated_ids = generated_ids.int().cpu().reshape(-1, self.audio_channels+1).T[:, prompt_length:]

        text = generated_ids[0, ::self.group_size][:-1]
        detokenized_text = self.tokenizer.decode(text, skip_special_tokens=False).strip().replace("<|empty|>", ".")
        print("Text channel:\t", detokenized_text)

        if output_audio_path:
            return_audio = True
        
        if not return_audio:
            return detokenized_text
        
        sosp_idx_locations = (text == self.sostm_idx).nonzero(as_tuple=True)[0]
        eosp_idx_locations = (text == self.eostm_idx).nonzero(as_tuple=True)[0]
        if len(sosp_idx_locations) == 0:
            start_location = 0
        else:
            start_location = sosp_idx_locations[0] * self.group_size + self.group_size
        if len(eosp_idx_locations) == 0:
            end_location = text.shape[0] * self.group_size
        else:
            end_location = eosp_idx_locations[0] * self.group_size
        audio_sequence = generated_ids[:, start_location:end_location]  #[audio_channels+1, audio_length]
        speech_sequence = audio_sequence[1:]

        mask = speech_sequence[0] != (self.speech_zeroemb_idx[0] if isinstance(self.speech_zeroemb_idx, list) else self.speech_zeroemb_idx)
        speech_sequence = speech_sequence[:, mask]

        assert (speech_sequence < torch.tensor(self.speech_zeroemb_idx).unsqueeze(1)).all()
        
        speech_sequence = speech_sequence.T.flatten()
    
        speech_str = "".join([f"<{i}>" for i in speech_sequence])
        tokens = torch.tensor(
            [int(num) for num in re.findall(r"(\d+)>", speech_str)]
        )

        if tokens.numel() == 0:
            wav = torch.zeros(24000)
            self.save_wav(output_audio_path, wav)
            return detokenized_text
        
        codes = tokens.reshape(-1, self.audio_channels).T
        codes = codes.type(torch.LongTensor).to(self.device)
        
        segment_len = 750
        wav_list=[]
        for start in range(0, codes.shape[-1], segment_len):
            wav = self.mimo_audio_tokenizer.decode(codes[:,start:start+segment_len]).float() 
            wav_list.append(wav)
        wav_concat = torch.cat(wav_list, dim=-1)

        #wav = self.mimo_audio_tokenizer.decode(codes).float()
        if output_audio_path is not None:
            self.save_wav(output_audio_path, wav_concat)
            return detokenized_text
        else:
            return wav_concat

    def asr_sft(self, audio, lang='zh'):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_asr_sft_prompt(audio, lang=lang)
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="asr")
        return result
    
    def tts_sft(self, text, output_path, instruct=None, read_text_only=True, prompt_speech=None):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_tts_sft_prompt(text, instruct=instruct, read_text_only=read_text_only, prompt_speech=prompt_speech)
        text_output = self.forward(input_ids, output_audio_path=output_path, stopping_criteria=stopping_criteria, task_name="tts")
        return text_output

    def gen_text(self, prompts):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id],
                group_size=self.group_size,
                audio_channels=self.audio_channels
            )
        ]
        input_ids = self.get_few_shot_prompts(prompts)
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, min_new_tokens=5, max_new_tokens=10, task_name="mmlu")
        return result

    def gen_speech(self, prompts, output_audio_path):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels
            )
        ]
        input_ids = self.get_few_shot_prompts(prompts, output_type="speech")
        result = self.forward(input_ids, output_audio_path=output_audio_path, stopping_criteria=stopping_criteria, min_new_tokens=10, max_new_tokens=25, task_name="mmlu")
        return result
    
    def qa(self, input_speech, input_text, output_audio_path=None):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.sosp_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels
            )
        ]
        input_ids = self.get_qa_prompt(input_speech, input_text)
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="qa")
        return result

    def audio_understanding_sft(self, input_speech, input_text, thinking=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
                )
            ]
        input_ids = self.get_audio_understanding_sft_prompt(input_speech, input_text, thinking=thinking)
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="audio_understanding")
        return result
    
    def spoken_dialogue_sft(self, input_speech, output_audio_path=None, assistant_character="乐于助人", assistant_style="像个助手", prompt_speech=None, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_spoken_dialogue_sft_prompt(input_speech, assistant_character=assistant_character, assistant_style=assistant_style, prompt_speech=prompt_speech, add_history=add_history)
        text = self.forward(input_ids, output_audio_path=output_audio_path, stopping_criteria=stopping_criteria, task_name="spoken_dialogue", add_history=add_history)
        return text

    def spoken_dialogue_sft_multiturn(self, input_speech, text_dialogue, output_audio_path=None, prompt_speech=None, assistant_character="乐于助人", assistant_style="像个助手"):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_spoken_dialogue_sft_multiturn_prompt(input_speech, text_dialogue, assistant_character="乐于助人", assistant_style="像个助手", prompt_speech=prompt_speech)
        text = self.forward(input_ids, output_audio_path=output_audio_path, stopping_criteria=stopping_criteria, task_name="spoken_dialogue", add_history=False)
        return text
    
    def speech2text_dialogue_sft(self, input_speech, thinking=False, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_s2t_dialogue_sft_prompt(input_speech, thinking=thinking)
        text = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="spoken_dialogue", add_history=add_history)
        return text
    
    def speech2text_dialogue_sft_multiturn(self, input_speech, text_dialogue, assistant_character="乐于助人", assistant_style="像个助手", add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_s2t_dialogue_sft_multiturn_prompt(input_speech, text_dialogue, assistant_character=assistant_character, assistant_style=assistant_style, add_history=add_history)
        text = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="spoken_dialogue", add_history=add_history)
        return text
    
    def text_dialogue_sft(self, input_text, thinking=False, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_text_dialogue_sft_prompt(input_text, thinking=thinking)
        text = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="text_chat", add_history=add_history)
        return text

    def text_dialogue_sft_multiturn(self, input_text, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_text_dialogue_sft_multiturn_prompt(input_text, add_history=add_history)
        text = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="text_chat", add_history=add_history)
        return text

    def few_shots_qa(self, examples):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.sosp_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        lm_prompt = [
            InputSegment(
                text=f"[Int]:",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        for i in range(len(examples[:-1])):
            audio_tokenized = self.preprocess_input(examples[i]['audio'])
            lm_prompt.append(
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
            if examples[i]['question'] != "":
                lm_prompt.append(
                    InputSegment(
                        text=f"Question: {examples[i]['question']}",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
                lm_prompt.append(
                    InputSegment(
                        text=f"Answer: {examples[i]['answer']}\n\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                )
        audio_tokenized = self.preprocess_input(examples[-1]['audio'])
        lm_prompt.append(
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        lm_prompt.append(
            InputSegment(
                text=f"Question: {examples[-1]['question']}",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        lm_prompt.append(
            InputSegment(
                text="\nAnswer:",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        input_ids = self.get_input_ids(lm_prompt)

        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="qa")
        return result
    
    def instruction_following(self, instructions, append_generation_prompt=False, thinking=False):
        input_ids = self.format_instruction_template(instructions, append_generation_prompt=append_generation_prompt, thinking=thinking)
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        result = self.forward(input_ids, stopping_criteria=stopping_criteria, task_name="audio_understanding")
        return result

    def clear_history(self):
        self.history = None
        print("History cleared")
