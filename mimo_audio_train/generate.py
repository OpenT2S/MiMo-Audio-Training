import torch
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torchaudio.transforms import MelSpectrogram

from mimo_audio_train.models.mimo_audio import MimoAudioModel

# Path to the model and speech tokenizer
model_path = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
lora_path = None
mimo_audio_tokenizer_path = "XiaomiMiMo/MiMo-Audio-Tokenizer"
device = "cuda"

model = MimoAudioModel(model_path, mimo_audio_tokenizer_path, lora_path, device)

# Spoken Dialogue
prompt_speech = "wav_files/prompt_speech.wav"
input_speech = "wav_files/input_speech.wav"
output_audio_path = "./output_audio.wav"

model.spoken_dialogue_sft(input_speech, output_audio_path, prompt_speech=prompt_speech)

# Speech Understanding
input_speech = "wav_files/input_speech.wav"
input_text = "What's the meaning reflected in the speech?"

return_text = model.audio_understanding_sft(input_speech, input_text, thinking=False)
print(return_text)