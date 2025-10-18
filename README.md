# CNN for Katakana Handwriting Practice

![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## Background
I have been learning Japanese on Duolingo for the past 6 months or so, and realized a problem with Duolingo's learning method: **Not enough handwriting practice**. With a lot of Asian languages like Japanese, Chinese, Korean etc., their writing systems are entirely different from English, so it is necessary to put in time to learn how to write these characters. 

Yet, there is relatively less practice on writing these characters. A stronger focus is put on how you read them, with exercises like Multiple Choice or Mix-and-Match where they are given. These teach you how to recognize and read, but less so how to write. As a result, I find myself being decent at reading the characters, but not very much so if I wanted to write something on blank paper.

I want **more handwriting practice**, and I want to **customize the set of characters I want to learn**, so that I can double down on my weak points and practice more. Which was the motivation for this project.

## Description
In this project, a custom Convolutional Neural Network architecture was defined and trained on the ETLCDB dataset (http://etlcdb.db.aist.go.jp/) for the purposes of recognizing handwritten Katakana characters for my Handwriting Practice Web App. 

The database has custom file formats and needs to be unpacked using the organization's provided Python Package, I then wrote a Python program that automates the process of reformatting the unpacked data into a PyTorch-compatible structure. The images are split into train, validation and test sets, then processed before feeding into the custom CNN for training. Dropout, data augmentation and early stopping were implemented to prevent overfitting. Since I am using a Mac, I used Apple's Metal Performance Shaders, its GPU framework (for Windows you can check if CUDA is available and use that) for acceleration.

**If interested**

The biggest challenge in this project was finding good data for training. The image sets found from Kaggle all had too little data, which causes severe overfitting even with data augmentation. If you are interested in the effect of sample number on the overfitting of a model, have a look at the HiraganaMLP notebook, where I was training a simpler Multi-Layer Perceptron Algorithm at the time on a Japanese database with 100 images per class, in contrast to the current CNN with 1300 images per class. I have evaluated its test results.

## CNN Architecture

## MLP Architecture

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- [Other dependencies]

### Quick Start

**Using pip:**
```bash
pip install your-package-name
