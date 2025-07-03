# Simple BERT-style Masked Language Model

This repository contains a minimal implementation of a BERT-style masked language model trained on Shakespeare’s text using PyTorch. It demonstrates:

- Character-level tokenisation with a custom vocabulary including special tokens (`<START>`, `<END>`, `<MASK>`)
- Transformer encoder blocks with multi-headed self-attention
- Masked token prediction training objective
- Visualisation of token and positional embeddings using PCA
- Text generation via iterative masked token sampling

---

## Features

- Character-level masked language modelling
- Multi-headed self-attention with residual connections and layer normalisation
- Positional embeddings
- Training with random token masking (~15% of tokens)
- Sampling-based text generation starting with `<START>` and `<END>` tokens
- Embedding space visualisation using PCA and matplotlib

---

## Hyperparameters

- Context window size: 16 tokens  
- Mask percentage: 15%  
- Embedding size: 64  
- Number of attention heads: 4  
- Number of encoder blocks: 4  
- Batch size: 64  
- Learning rate: 1e-3  
- Epochs: 1000  

---

## Notes

- Minimal educational implementation, **not optimised for production use**  
- Character-level tokens rather than subword tokens  
- Training speed limited by simple masking and batch size  
- PCA visualisations provide insight into embeddings  

---
