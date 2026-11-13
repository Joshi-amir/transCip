# transCip

An experimental GPT-style language model built from scratch using PyTorch.

transCip is a decoder-only Transformer trained on cleaned WikiText-2 using next-token prediction.

## Model Architecture

- Parameters: 46.28 million
- Transformer layers: 12
- Attention heads: 8
- Embedding dimensions: 512
- Context length: 512 tokens
- Tokenizer: Byte-level BPE (8K vocabulary)

## Training

- Dataset: Cleaned WikiText-2
- Training steps: 5,000
- Hardware: NVIDIA RTX 4060 Laptop GPU
- Best validation loss: 4.8086

## Installation

```bash
pip install torch tokenizers
```

## Usage

Download `transcip_v2.pt` from the [Releases](https://github.com/Joshi-amir/transCip/releases) page and place it inside:

```text
models/checkpoints/
```

Run the model:

```bash
python generate.py
```

## Status

Experimental project for learning and developing Transformer-based language models. The current model focuses on text generation and is not instruction-tuned.
