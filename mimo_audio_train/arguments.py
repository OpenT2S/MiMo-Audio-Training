import transformers

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="XiaomiMiMo/MiMo-Audio-7B-Instruct")
    speech_tokenizer_name_or_path: Optional[str] = field(default="XiaomiMiMo/MiMo-Audio-Tokenizer")

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    validate_path: str = field(default=None,
                           metadata={"help": "Path to the validation data."})
@dataclass
class CustomArguments:
    cache_dir: Optional[str] = field(default=None)
    lora_enable: bool = field(default=False, metadata={"help": "Enable LoRA"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout rate"})
    lora_weight_path: str = field(default="", metadata={"help": "LoRA weight path"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias type"})
    lora_modules: str = field(default="q_proj,k_proj,v_proj,o_proj", metadata={"help": "LoRA target modules"})
    double_quant: bool = field(default=True, metadata={"help": "Use double quantization"})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization type: fp4 or nf4"})
    bits: int = field(default=16, metadata={"help": "Quantization bits"})
    speech_loss_weights: str = field(default="100-12-8-6-4-2-2-1-1", metadata={"help": "Loss weights for speech, multi-rvq"})
    input_local_lr: float = field(default=2e-5, metadata={"help": "Learning rate for input_local_transformer modules"})
    output_local_lr: float = field(default=2e-5, metadata={"help": "Learning rate for local_transformer modules"})

class TrainingArguments(transformers.TrainingArguments):
    def __init__(
        self,
        *args,
        optim: str = "adamw_torch",
        max_seq_length: int = 8192,
        fp16: bool = False,
        bf16: bool = True,
        **kwargs
    ):
        # Call parent class initialization
        super().__init__(*args, **kwargs)

        # Custom fields
        self.optim = optim
        self.max_seq_length = max_seq_length
        self.fp16 = fp16
        self.bf16 = bf16