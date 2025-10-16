import sys
import os
import pathlib
import logging
import time

import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torchaudio.transforms import MelSpectrogram

from mimo_audio_train.models.src_mimo_audio.modeling_mimo_audio import MiMoAudioForCausalLM, MiMoAudioArguments
from mimo_audio_train.models.src_mimo_audio.mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer
from mimo_audio_train.arguments import ModelArguments, DataArguments, TrainingArguments, CustomArguments
from mimo_audio_train.dataset.speech_dataset import make_dialogue_module
from mimo_audio_train.training_file.trainer import MiMoAudioTrainer
from mimo_audio_train.callbacks.custom_callbacks import CustomCallback
from mimo_audio_train.utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    find_all_linear_names
)

if torch.cuda.device_count() == 1:
    print('only one gpu, set start method to spawn')
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

local_rank = None

def rank0_print(*args):
    """Print function, only print when local_rank is 0"""
    if local_rank == 0:
        print(*args)


def load_model_and_tokenizer(load_model_args, training_args, custom_args):
    """
    Load model and tokenizer
    
    Args:
        load_model_args: model arguments
        
    Returns:
        model: loaded MiMoAudio model
        tokenizer: text tokenizer
        mimo_audio_tokenizer: audio tokenizer
        mel_transform: Mel spectrogram transformer
    """
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        load_model_args.model_name_or_path
    )

    # Add special tokens
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
    added_tokens = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
            added_tokens.append(token)
    
    if added_tokens:
        rank0_print(f"Added special tokens: {', '.join(added_tokens)}")

    # Get special token IDs
    sosp_idx = tokenizer.convert_tokens_to_ids("<|sosp|>")
    eosp_idx = tokenizer.convert_tokens_to_ids("<|eosp|>")
    empty_token = tokenizer.convert_tokens_to_ids("<|empty|>")
    sostm_idx = tokenizer.convert_tokens_to_ids("<|sostm|>")
    eostm_idx = tokenizer.convert_tokens_to_ids("<|eostm|>")
    eot_idx = tokenizer.convert_tokens_to_ids("<|eot|>")

    # Build model arguments
    model_args = MiMoAudioArguments(
        model_name_or_path=load_model_args.model_name_or_path,
        sosp_idx=sosp_idx,
        eosp_idx=eosp_idx,
        empty_idx=empty_token,
        sostm_idx=sostm_idx,
        eostm_idx=eostm_idx,
        eot_idx=eot_idx,
        speech_loss_weights=custom_args.speech_loss_weights,
    )

    # Load model
    rank0_print(f"开始从 {load_model_args.model_name_or_path} 加载模型...")
    start_loading_time = time.monotonic()
    try:
        model = MiMoAudioForCausalLM.from_pretrained(
            load_model_args.model_name_or_path,
            args=model_args,
            torch_dtype=torch.bfloat16,
            device_map={"": training_args.device},
        )
        rank0_print(
            f"Model loaded, time: {time.monotonic() - start_loading_time:.2f} seconds, device: {training_args.device}"
        )
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise

    # Load audio tokenizer
    rank0_print(f"开始从 {load_model_args.speech_tokenizer_name_or_path} ...")
    start_loading_mimo_audio_tokenizer_time = time.monotonic()
    try:
        mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
            load_model_args.speech_tokenizer_name_or_path
        )
        mimo_audio_tokenizer.to(device=training_args.device, dtype=torch.bfloat16)
        rank0_print(
            f"Audio tokenizer loaded, time: {time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds, device: {training_args.device}"
        )
    except Exception as e:
        logging.error(f"Audio tokenizer loading failed: {str(e)}")
        raise
    
    # Initialize Mel spectrogram transformer
    try:
        mel_transform = MelSpectrogram(
            sample_rate=mimo_audio_tokenizer.config.sampling_rate,
            n_fft=mimo_audio_tokenizer.config.nfft,
            hop_length=mimo_audio_tokenizer.config.hop_length,
            win_length=mimo_audio_tokenizer.config.window_size,
            f_min=mimo_audio_tokenizer.config.fmin,
            f_max=mimo_audio_tokenizer.config.fmax,
            n_mels=mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        )
        rank0_print("Mel spectrogram transformer initialized")
    except Exception as e:
        logging.error(f"Mel spectrogram transformer initialization failed: {str(e)}")
        raise
    
    return model, tokenizer, mimo_audio_tokenizer, mel_transform

def train(attn_implementation=None):
    """
    Main training function
    
    Args:
        attn_implementation: attention implementation (currently not used)
    """
    global local_rank
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set transformers logging level, ensure training loss can be output
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()
    
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments)
    )
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
    
    
    local_rank = training_args.local_rank
    rank0_print(f"本地 rank: {local_rank}")
    rank0_print(f"日志步数: {training_args.logging_steps}")
    rank0_print(f"日志级别: {training_args.log_level}")
    rank0_print(f"禁用tqdm: {training_args.disable_tqdm}")
    
    # Determine compute data type
    compute_dtype = (
        torch.float16 if training_args.fp16 
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    rank0_print(f"Compute data type: {compute_dtype}")

    # Configure quantization parameters (if using quantization)
    bnb_model_from_pretrained_args = {}
    if custom_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        rank0_print(f"Using {custom_args.bits}-bit quantization")
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=custom_args.bits == 4,
            load_in_8bit=custom_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=custom_args.bits == 4,
                load_in_8bit=custom_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=custom_args.double_quant,
                bnb_4bit_quant_type=custom_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    # Load model and tokenizer
    model, tokenizer, mimo_audio_tokenizer, mel_transform = load_model_and_tokenizer(model_args, training_args, custom_args)

    model.tokenizer = tokenizer
    # Prepare quantization training
    if custom_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        rank0_print("Preparing quantization training model...")
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        rank0_print("Enabling gradient checkpointing...")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Configure LoRA
    if custom_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        rank0_print("Configuring LoRA adapter...")
        
        target_modules = custom_args.lora_modules.split(",")
        rank0_print(f"LoRA target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=custom_args.lora_r,
            lora_alpha=custom_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=custom_args.lora_dropout,
            bias=custom_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # Set model data type
        if custom_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        rank0_print("Adding LoRA adapter...")
        model = get_peft_model(model, lora_config)
    
    # Output trainable parameters information
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    rank0_print("=" * 50)
    rank0_print("Trainable parameters statistics:")
    rank0_print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    rank0_print(f"  Total parameters: {total_params / 1e6:.2f}M")
    rank0_print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    rank0_print("=" * 50)

    # Adjust model各层的精度
    if custom_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        rank0_print("Adjusting quantization model各层精度...")
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # Set padding index
    data_args.padding_idx = tokenizer.pad_token_id

    # Prepare data module
    rank0_print("Preparing data module...")
    data_module = make_dialogue_module(
        tokenizer=tokenizer,
        mimo_audio_tokenizer=mimo_audio_tokenizer,
        mel_transform=mel_transform,
        data_args=data_args,
        model=model,
        lora_enable=custom_args.lora_enable
    )
    
    # Initialize Trainer
    rank0_print("Initializing Trainer...")

    trainer = MiMoAudioTrainer(
        model=model,
        args=training_args,
        **data_module
    )
    
    # Start training
    rank0_print("=" * 50)
    rank0_print("Starting training...")
    rank0_print("=" * 50)
    
    # Check if checkpoint exists
    checkpoint_dir = pathlib.Path(training_args.output_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    
    if checkpoints:
        rank0_print(f"Found {len(checkpoints)} checkpoints, resuming training from latest checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch")
        trainer.train()
    
    # Save training state
    rank0_print("Saving training state...")
    trainer.save_state()

    # Restore model cache settings
    model.config.use_cache = True

    # Save model
    rank0_print("Saving model...")
    if custom_args.lora_enable:
        # Save LoRA model
        rank0_print("Saving LoRA weights...")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), custom_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict, 
                os.path.join(training_args.output_dir, 'non_lora_trainables.bin')
            )
            rank0_print(f"LoRA model saved to: {training_args.output_dir}")
    else:
        # Save full model
        rank0_print("Saving full model...")
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir
        )
        rank0_print(f"Model saved to: {training_args.output_dir}")
    
    rank0_print("=" * 50)
    rank0_print("Training completed!")
    rank0_print("=" * 50)


if __name__ == "__main__":
    import warnings
    
    # Ignore specific warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.utils\.checkpoint")
    
    # Start training
    train()
