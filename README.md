# NLP N-gram Language Model

This repository contains the implementation of n-gram language models for Natural Language Processing (CS6320 Assignment 1).

## Overview

This project implements unigram and bigram language models with:
- **Add-k smoothing** for handling unseen words and sequences
- **Unknown word handling** using frequency thresholds
- **Perplexity evaluation** on validation and test sets
- **Comprehensive preprocessing** including lemmatization and punctuation removal

## Files

- `NLP_Ngramsfinal.py` - Main implementation with n-gram models
- `CS6320_Assignment1_report.pdf` - Report

## Key Features

### 1. Unknown Word Handling
- Replaces low-frequency words with `<UNK>` token
- Configurable frequency thresholds (1, 2, 3)
- Dramatically improves model generalization

### 2. Smoothing Techniques
- Add-k smoothing with different k values (0, 0.5, 1.0)
- Prevents zero probabilities for unseen words/sequences

### 3. Model Evaluation
- Perplexity calculation on validation and test sets
- Systematic testing of different parameter combinations

## Results Summary

| Unknown Word Threshold | Validation (Unigram) | Test (Unigram) | Validation (Bigram) | Test (Bigram) |
|----------------------|---------------------|----------------|-------------------|---------------|
| 1                    | 58.13              | 32417.82       | 1.77              | 259289385.13  |
| 2                    | 1.93               | 3.03           | 2.55              | 2.73          |
| **3**                | **1.50**           | **2.24**       | **1.89**          | **2.13**      |

**Best Performance**: Unknown word threshold 3 with no smoothing (k=0)

## Usage

1. Ensure you have the required dependencies:
   ```bash
   pip install nltk scikit-learn
   ```

2. Prepare your data files:
   - `train.txt` - Training corpus (one review per line)
   - `validation.txt` - Validation corpus (one review per line)

3. Run the implementation:
   ```bash
   python NLP_Ngramsfinal.py
   ```

## Group Members

- Adarsh Gella (AXG240019)
- Akhila Susarla (AXS240035)
- Vijaya Sai Latha Pulipati (VXP230093)
- Abhiram Reddy Madapa (AXM240036)

## Key Insights

- **Unknown word handling is crucial** for model generalization
- **Bigram models generally perform better** than unigram models
- **Frequency threshold 3** provides optimal performance
- **Smoothing helps** when encountering completely unseen data

## Repository

GitHub: [https://github.com/gadarsh043/NLP_N-gramsmodel.git](https://github.com/gadarsh043/NLP_N-gramsmodel.git)
