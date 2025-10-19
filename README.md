# CNN for Katakana Handwriting Practice

## Description
#### This repo is the core part of a larger project -  a Japanese Handwriting Practice Web App called Lingo Write. I have been learning Japanese for 8 months on Duolingo now, and I have realized that it could use more handwriting practice, since I find myself being decent at reading/ recognizing Japanese characters, but struggling to write some of them. The web app is an additional tool for you to practice your handwriting, and it gives you more control over the charcters you want to practice on, so you can focus on your weak points and get them right. It is almost complete (stay tuned).

#### Here we will focus on handwriting recognition for Japanese characters.

The Japanese language consists of 3 writing systems, Hiragana, Katakana and Kanji.

In this project, I define a custom Convolutional Neural Network architecture and train it on the **ETLCDB dataset** (http://etlcdb.db.aist.go.jp/) for the purposes of recognizing handwritten Katakana characters. 

The database has custom file formats and needs to be unpacked using the organization's provided Python Package, I then wrote a Python program that automates the process of reformatting the unpacked data into a PyTorch-compatible structure. The images are split into train, validation and test sets, then processed before feeding into the custom CNN for training. Dropout, data augmentation and early stopping were implemented to prevent overfitting. Since I am using a Mac, I used Apple's Metal Performance Shaders, its GPU framework (for Windows you can check if CUDA is available and use that) for acceleration.

*If interested:*
 The biggest challenge in this project was finding good data for training. The image sets found from Kaggle all had too little data, which causes severe overfitting even with data augmentation. If you want to see a demonstration of the effect of sample number on model overfitting, have a look at the HiraganaMLP notebook, where I trained a simpler Multi-Layer Perceptron Algorithm on a Japanese database with 100 images per class, in contrast to the current CNN with 1300 images per class. I have evaluated its test results.

## ETLCDB Database
 *Database used: ETL6 Character Database, Electrotechnical Laboratory, Japanese Technical Committee for Optical Character Recognition, ETL Character Database, 1973-1984.*

 This database consists of multiple groups of data, ranging from ETL1-ETL9, each having different types of characters, some with special characters, numbers etc., included. For more info on this database, have a look at the introduction in https://github.com/CaptainDario/ETLCDB_data_reader.
## CNN Architecture

## MLP Architecture

## 🛠 Installation

### Quick Start

**Using pip:**
```bash
pip install torch numpy matplotlib matplotlib-inline torchvision
