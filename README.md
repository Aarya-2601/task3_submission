# task3_submission
# Word2Vec Skip-gram Replication from Scratch

This repository contains an independent, clean-code implementation of the Skip-gram architecture proposed in Mikolov et al.'s seminal paper: *"Efficient Estimation of Word Representations in Vector Space"*.

---

## 1. Project Overview
This project replicates the core log-linear architecture of the Word2Vec Skip-gram model using PyTorch. The model maps discrete text tokens into a continuous vector space where words sharing similar contextual environments gravitate toward similar coordinates. 

To maintain strict alignment with the assignment's **Implementation Fidelity** guidelines, this project avoids high-level machine learning wrappers or pre-trained parameters. The tokenizer, sliding-window data processor, dictionary lookup structures, and neural embedding layers are all written entirely from scratch.

---

## 2. Getting Started

### Prerequisites & Dependencies
This project requires Python 3 and PyTorch. You do not need a GPU; the baseline script is fully optimized to run efficiently on a standard laptop CPU.

Install PyTorch using pip:
```bash
pip install torch

What to Expect (Outputs)
When you execute the script, the following three phases will run sequentially in your console:

Vocabulary Building & Tokenization: The text string is cleaned, broken into unique tokens, and indexed into local word-to-index (w2i) and index-to-word (i2w) dictionaries.

Training Iteration Logs: The training loop runs for 100 epochs, printing an average cross-entropy convergence loss every 20 epochs. You will see the loss score steadily decrease over time.

Fidelity Verification: Upon training completion, the program prints specific target keywords along with the absolute geometric magnitude (vector norm) of their newly trained continuous coordinate tensors.
