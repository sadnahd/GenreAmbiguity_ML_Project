# Project Group #44: Quantifying Genre Ambiguity in Music

**Authors:** Sidhi, Stephanie, Gio

## Overview

This project investigates whether multi-genre music samples exhibit greater classification uncertainty than single-genre samples. Using mel spectrograms generated from the Music4All (MTG) and GTZAN datasets, we train and evaluate deep learning models to classify music by genre while quantifying aleatoric uncertainty.

Our approach builds on Kendall and Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, implementing uncertainty-aware classification models that incorporate variance estimates into their predictions.

**Research Question:** When classifying music samples by genre, do multi-genre samples exhibit significantly higher model uncertainty than single-genre samples?

## Notebook Contents

The final notebook is organized into the following sections:

1. **Research Question** – Motivation and theoretical foundation for measuring genre ambiguity through model uncertainty.
2. **EDA Figures** – Genre distributions, multi- versus single-genre sample distributions, spectrogram visualizations, and UMAP dimensionality reduction.
3. **Pipeline & Model Training** – Data preprocessing, augmentation, and training of ResNet and Vision Transformer (ViT) classifiers, including uncertainty-aware variants.
4. **Model Error & Uncertainty Analysis** – Training performance, classification errors, F1 scores, confusion matrices, and statistical analysis of uncertainty across multi- and single-genre samples.

## Data & Model Resources

- **Original Datasets:** [MTG](https://mtg.upf.edu/) and [GTZAN](http://marsyas.info/downloads/datasets.html)
- **Merged Dataset (Mel Spectrograms):** [Google Drive](https://drive.google.com/file/d/17V6lYRQw80f7Jy2A8YlOGpIrpunKT5Q4/view?usp=sharing)
- **Pre-trained Model Weights:** [Google Drive](https://drive.google.com/file/d/1BpbW3zK5387j5iIg_4EhYbvP4AZsB3Mc/view?usp=sharing)

## Reproducing the Project

- The dataset sampling and merging procedure can be reproduced by running `utils/Merging datasets.ipynb`.
- Pre-trained model weights are provided above to avoid computationally expensive retraining.
- The full training loops and supporting model implementations are included in the notebook and `utils` folder.
- Full dependencies are provided in the zipped `requirements.txt` file.

The notebook uses Python, PyTorch, Hugging Face Transformers, scikit-learn, UMAP, and standard data visualization libraries.

## Reproducibility

A random seed of `1090` is used throughout the project where applicable. Models automatically use CUDA when available and otherwise fall back to CPU.

*README created using AI assistance.*

