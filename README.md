<div align="center">
  <picture>
    <source srcset="https://github.com/XiaomiMiMo/MiMo-VL/raw/main/figures/Xiaomi_MiMo_darkmode.png?raw=true" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/XiaomiMiMo/MiMo-VL/raw/main/figures/Xiaomi_MiMo.png?raw=true" width="60%" alt="Xiaomi-MiMo" />
  </picture>
</div>

<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    MiMo-Audio-Training Toolkit
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<br/>

<div align="center" style="line-height: 1;">
  |
  <a href="https://github.com/XiaomiMiMo/MiMo-Audio" target="_blank">🤖 GitHub</a>
  &nbsp;|
  <a href="https://huggingface.co/collections/XiaomiMiMo/mimo-audio-68cc7202692c27dae881cce0" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://github.com/XiaomiMiMo/MiMo-Audio/blob/main/MiMo-Audio-Technical-Report.pdf" target="_blank">📄 Paper</a>
  &nbsp;|
  <a href="https://xiaomimimo.github.io/MiMo-Audio-Demo" target="_blank">📰 Blog</a>
  &nbsp;|
  <a href="https://huggingface.co/spaces/XiaomiMiMo/mimo_audio_chat" target="_blank">🔥 Online Demo</a>
  &nbsp;|
  <br/>
</div>

<br/>

## Introduction

Welcome to the **MiMo-Audio-Training** toolkit! This toolkit is designed to fine-tune the [XiaomiMiMo/MiMo-Audio-7B-Instruct](https://huggingface.co/XiaomiMiMo/MiMo-Audio-7B-Instruct). This toolkit serves as a reference implementation for researchers and developers interested in MiMo-Audio and looking to adapt it to their own custom tasks.

## Supported Tasks

The MiMo-Audio-Eval toolkit supports a comprehensive set of tasks. Some of the key features include:

* **Tasks**:

  * **SFT**:

    * ASR
    * TTS / InstructTTS
    * Audio Understanding and Reasoning
    * Spoken Dialogue


## Getting Started

To get started with the MiMo-Audio-Training toolkit, follow the instructions below to set up the environment and install the required dependencies.

### Prerequisites (Linux)

* Python 3.12
* CUDA >= 12.0

### Installation:

```bash
git clone --recurse-submodules https://github.com/XiaomiMiMo/MiMo-Audio-Training
cd MiMo-Audio-Training
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1
pip install -e .
```


> \[!Note]
> If the compilation of flash-attn takes too long, you can download the precompiled wheel and install it manually:
>
> * [Download Precompiled Wheel](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl)
>
> ```sh
> pip install /path/to/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
> ```

### Training Process:

Download the fine-tuning Dataset and pre-process the data as the `instruct_template.md`

## Training

We provide multiple training scripts under the `scripts` directory, supporting both single-GPU and multi-GPU training setups.

```
cd MiMo-Audio-Training
bash scripts/train_multiGPU_torchrun.sh
```

## Generate and Evaluation
Run inference using: `generate.py`

Evaluate the SFT model with 🌐[MiMo-Audio-Eval](https://github.com/XiaomiMiMo/MiMo-Audio-Eval).


## Citation

```bibtex
@misc{coreteam2025mimoaudio,
      title={MiMo-Audio: Audio Language Models are Few-Shot Learners}, 
      author={LLM-Core-Team Xiaomi},
      year={2025},
      url={https://github.com/XiaomiMiMo/MiMo-Audio}, 
}
```


## Contact

Please contact us at [mimo@xiaomi.com](mailto:mimo@xiaomi.com) or open an issue if you have any questions.