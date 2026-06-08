# Fine-Grained Bird Classification on CUB-200-2011

This repository contains a PyTorch implementation for fine-grained bird species classification. The project integrates **Polarized Self-Attention (PSA)** into a pre-trained **ResNet50** backbone to capture subtle, localized features (like beaks and wing patterns) critical for distinguishing highly similar species.

## 📌 Project Overview
Fine-grained classification is challenging due to high inter-species similarity and low intra-species variance. This project implements a hybrid architecture that leverages a frozen ResNet50 backbone for primitive features while fine-tuning deep attention layers to capture specialized visual cues. 

Advanced data augmentation techniques—including **CutMix**, **MixUp**, and **RandAugment**—are integrated into the training pipeline to enforce generalizability over limited visual data.

---

## 📊 Results

Evaluated on the **CUB-200-2011** dataset (200 bird species):

| Model Architecture | Top-1 Test Accuracy | Macro $F_1$-Score |
| :--- | :---: | :---: |
| Baseline ResNet50 | 75.06% | 0.74 |
| **ResNet50 + PSA (Observed)** | **78.63%** | **0.79** |

* Incorporating the Polarized Self-Attention module yields a **+3.57%** absolute improvement in accuracy.
* The attention mechanism dramatically improves classification for highly confused categories (e.g., the *Herring Gull* $F_1$-score rose from 0.12 to 0.40).

---

## ⚡ Key Hyperparameters

* **Optimizer**: AdamW
* **Learning Rate**: $1 \times 10^{-4}$
* **Weight Decay**: $1 \times 10^{-5}$
* **Learning Rate Scheduler**: Cosine Annealing (30 epochs)
* **Batch Size**: 32
* **CutMix / MixUp ($\alpha$)**: 0.3 / 0.1
