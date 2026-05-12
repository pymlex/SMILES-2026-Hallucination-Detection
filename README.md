# SMILES-2026 Hallucination Detection

Detect whether a small language model's answer is *hallucinated* (fabricated) or *truthful* using the model's own internal representations (hidden states).

## Overview

Large (and small) language models sometimes *hallucinate* — they generate plausible-sounding text that is factually incorrect.  This competition asks you to build a **lightweight binary classifier** (called a *probe*) that reads the model's internal hidden states and predicts whether a given response is truthful (`label = 0`) or hallucinated (`label = 1`).

The language model used throughout is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a decoder-only causal transformer with 24 layers and a hidden dimension of 896. **Primary ranking metric** is accuracy on the held-out `test.csv`.

## Dataset

`data/dataset.csv` contains 689 labelled samples with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | str | Full ChatML-formatted conversation context fed to Qwen |
| `response` | str | The model's generated response |
| `label` | float | `1.0` = hallucinated · `0.0` = truthful |

The `prompt` uses the **ChatML** template built into Qwen models:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Given the context, answer the question …<|im_end|>
<|im_start|>assistant
```

## Evaluation

The report on used techniques and results can be found in `SOLUTION.md`. 
