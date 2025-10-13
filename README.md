# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Apex

**Team Members:** 

<div align="center">

| Name             | GitHub                                       | LinkedIn                                                           | Department @ IIT Patna |
|:-----------------|:--------------------------------------------:|:------------------------------------------------------------------:|:----------------------:|
| **S Akash** | [@Akasxh](https://github.com/Akasxh)         | [LinkedIn](https://www.linkedin.com/in/s-akash-project-gia/)       | EEE                    |
| **Anirudh D Bhat** | [@RudhaTR](https://github.com/RudhaTR)       | [LinkedIn](https://www.linkedin.com/in/anirudh-d-bhat/)            | CSE                    |
| **Akash S** | [@akash1764591](https://github.com/akash1764591)| [LinkedIn](https://www.linkedin.com/in/akash-s-473781263/)          | EEE                    |
| **Ammar Ahmad** | [@ammarrahmad](https://github.com/ammarrahmad)| [LinkedIn](https://www.linkedin.com/in/ammar-ahmad-5a8343251/)      | AI & DS                |

</div>

**Submission Date:** 13/10/2025

---

## 1. Executive Summary

Our solution addresses the product pricing challenge by fine-tuning a large transformer model, 
DeBERTa-v3-large, on carefully engineered textual features. We developed an efficient hybrid 
data preparation pipeline that uses Regex for rapid parsing and a targeted LLM for imputing 
missing values. This approach achieved a final SMAPE of 42.22% on the validation set of 7500 
samples, demonstrating the effectiveness of modern NLP models for complex regression tasks

---

## 2. Methodology Overview

### 2.1 Problem Analysis

We interpreted the challenge as a text-based regression problem. The primary goal was to 
predict a continuous price variable from semi-structured catalog descriptions. Our initial 
Exploratory Data Analysis (EDA) revealed several key characteristics of the dataset.

**Key Observations:**

● Skewed Target Variable: The product price was heavily right-skewed, which required a 
logarithmic transformation (log1p) to normalize the distribution for model training. 

● Semi-Structured Text: The catalog_content field contained a mix of structured data 
(item name, value, unit ,descriptions, bullet points)., which might or might not be present 
at times or have None values .This required a robust parsing strategy. 

● Data Redundancy: Information from the product images was almost always present in 
the text descriptions. We concluded that a text-only approach would be more 
computationally efficient without a significant loss of information.Since we would use 
image to augment our description if we did not have information(unable to implement 
due to compute and time constraint)

### 2.2 Solution Strategy

Our strategy centered on transforming the semi-structured text into a rich, consistent format and 
then leveraging a powerful pre-trained language model to learn the relationship between the 
description and the price.

**Approach Type:** Single Model (Fine-tuned Transformer with a Regression Head)

**Core Innovation:**  A hybrid feature engineering pipeline that combines the speed of Regular 
Expressions for extracting common data points (value, unit) with the intelligence of a 
Language Model for targeted imputation of missing or complex entries. This provided a 
high-quality dataset without the prohibitive computational cost of processing every entry with an 
LLM.

---

## 3. Model Architecture

### 3.1 Architecture Overview

Our model follows a standard architecture for text regression using a pre-trained transformer. 
The processed text is tokenized and fed into the DeBERTa encoder. The final hidden state of 
the [CLS] token (or mean-pooled output) is then passed through a multi-layer perceptron (MLP) 
regression head to produce the final log-price prediction. 

### Model Pipeline :

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7127382-e4b3-49b5-be4a-60573feca2eb" alt="image" width="680" height="362" />
</p>

### 3.2 Model Components

**Text Processing Pipeline:**
- Preprocessing steps:
  - Extract item name, value, unit, and description from raw catalog content using Regex.
  - Use a small LLM to fill None/NA values where Regex failed.
  - Impute remaining nulls with defaults (e.g., unit='Count', description='no description').
  - Combine the extracted features into a single structured string: Category: [category] [SEP] description: [description] [SEP] Amount: [value] [unit].
  - Apply a log1p transformation to the target price variable

- Model type: microsoft/deberta-v3-large 
- Key parameters:
  - Max sequence length: 160
  - Dropout: 0.2
  - Encoder Learning Rate: 2e-5
  - Head Learning Rate: 1e-3
  - Batch Size: 16
    
---


## 4. Model Performance

### 4.1 Final Test Set Results 

The model was trained for 10 epochs on a training set of 67,499 samples. The final evaluation 
on the held-out test set of 7500 samples yielded the following results: 

- SMAPE Score: 42.22% 
- Other Metrics: 
- Mean Absolute Error (MAE): $9.44 
- R-squared (R²): 0.29

## 5. Conclusion

Our approach successfully demonstrates that fine-tuning a large transformer model like 
DeBERTa-v3-large is a highly effective strategy for price prediction from textual data. The key 
lesson was the importance of pragmatic feature engineering; our hybrid Regex-LLM pipeline 
provided a crucial balance between data quality and computational feasibility. The final model 
delivered a strong SMAPE score of 42.22%, validating our methodology. 

---

## Appendix

### A. Code artefacts

A link to the complete code directory can be found below. For immediate reference, the main 
training and evaluation script is also attached in this appendix.


Link : `https://drive.google.com/file/d/1WwFyzrKUbCFXXAKv7RyQjrae3UFAftzB/view?usp=sharing`

This repository contains all the codes for the solution and the experiments done.


### B. Additional Results

During our experimentation phase, we tested several alternative models. The results below 
informed our decision to focus on fine-tuning a large transformer model. 
- DeBERTa-v3-base: 43.59% SMAPE 
- DistilBERT: 45% SMAPE (used for rapid prototyping) 
- Sentence Embeddings + XGBoost: 52% SMAPE 
- Sentence Embeddings + Neural Network: 49% SMAPE 
- LLM Fine-Tuning (SFT/IFT): Explored but abandoned due to excessive training time 
and compute requirements, which hindered effective iteration.

---
