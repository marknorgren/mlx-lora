# 🧠 MLX LoRA Fine-tuning

Fine-tune LLMs on Apple Silicon using LoRA adapters. Train a 7B model to answer domain-specific questions with 6.5MB adapters.

## 📊 MLX Framework

Apple's ML framework for Apple Silicon:
- GPU acceleration without CUDA
- Unified memory (no CPU/GPU copies)
- NumPy-compatible API
- 2-3x faster than CPU PyTorch on M-series

## 🔧 LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning:
- Trains 0.17% of parameters (1.7M vs 7.2B)
- 6.5MB adapter vs 3.8GB base model
- 90% less memory than full fine-tuning
- Merge or swap adapters at runtime

## 🏒 Demo: Stanley Cup Q&A

Fine-tune Mistral-7B to answer NHL trivia:
1. Train on 110 years of Stanley Cup data (1915-2025)
2. 2500 iterations for best results
3. Generate answers in <1 second
4. Deploy 6.5MB adapter instead of 3.8GB model

## Quick Start

### Setup

```bash
# Clone and install dependencies
git clone <repo>
cd mlx-lora-public
uv sync  # Installs mlx-lm and dependencies
```

### Test Pre-trained Adapter

```bash
# Test the pre-trained adapter
mlx_lm.generate \
  --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  --adapter-path adapters/nhl-stanley-cups-demo \
  --prompt "Who won the Stanley Cup in 2024?" \
  --max-tokens 50
```

Output: `The Florida Panthers won the Stanley Cup in 2024, defeating the Edmonton Oilers 4-3 in the series.`

### Train Your Own Adapter

```bash
# Quick training (300 iterations)
mlx_lm.lora \
  --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  --train \
  --data data/nhl_stanley_cups \
  --adapter-path adapters/my-adapter \
  --iters 300

# Full training (2500 iterations)
mlx_lm.lora \
  --model "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  --train \
  --data data/nhl_stanley_cups \
  --adapter-path adapters/my-adapter \
  --iters 2500 \
  --learning-rate 1.5e-5 \
  --lora-rank 16
```

## ⚡ How LoRA Works

```
Base Model (Frozen)          LoRA Adapter (Trainable)
    [W₀]         +              [BA]
   (d×k)                    (d×r)(r×k)
                            where r << min(d,k)
```

- **W₀**: Original model weights (frozen)
- **B, A**: Low-rank matrices (trainable)
- **r**: Rank (typically 8-32)

The adapter adds `ΔW = BA` to the original weights during inference.

## 📁 Project Structure

```
mlx-lora-public/
├── adapters/
│   ├── nhl-stanley-cups-demo/  # Demo adapter (300 iterations)
│   └── stanley-cup-best-2500/  # Best performing adapter (2500 iters)
├── data/
│   └── nhl_stanley_cups/       # Train/valid/test splits (JSONL)
├── demo.py                     # Basic inference example
├── interactive_demo.py         # Interactive Q&A interface
├── config.py                   # Training hyperparameters
└── docs/
    └── LORA.md                # LoRA implementation details
```

## 🎯 Optimal Configuration

Best performance with:
- **Iterations**: 2500 (45 minutes on M1 Max)
- **Learning rate**: 1.5e-5
- **LoRA rank**: 16
- **Batch size**: 2
- **Peak memory**: 8GB

## 💬 Interactive Demo

```bash
# Use the high-accuracy adapter
uv run python interactive_demo.py
```

Example questions:
- "Who won the Stanley Cup in 2019?" → St. Louis Blues (4-3 vs Boston)
- "How many times have the Red Wings won?" → 11 championships
- "Which team won in 1967?" → Toronto Maple Leafs

## 📈 Performance Metrics

**Memory Usage**:
- Base model: 3.8GB → 6.5MB adapter (585x smaller)
- Training: 8GB peak
- Inference: 4GB

**Speed** (M1 Max):
- Training: 45 min for 2500 iterations
- Inference: 30 tokens/sec
- First token latency: <1 second

## 🛠️ Technical Details

- **Parameters**: 1.7M trainable / 7.2B total (0.024%)
- **Quantization**: 4-bit base model (16GB → 3.8GB)
- **Adapter files**: Checkpoints every 250 iterations
- **Data format**: JSONL with Mistral chat template

## 📚 Resources

- [MLX Docs](https://ml-explore.github.io/mlx/)
- [LoRA Paper (2021)](https://arxiv.org/abs/2106.09685)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)

## 💻 Requirements

- Apple Silicon (M1/M2/M3/M4)
- Python 3.9-3.12
- 16GB RAM
- 8GB free disk space
