# SentimentScope: Sentiment Analysis with a Custom Transformer in PyTorch

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow?style=for-the-badge)

## Overview

This project, **SentimentScope**, involves building, training, and evaluating a transformer model from scratch to perform sentiment analysis. The model is designed as a binary classifier to determine whether a movie review from the IMDB dataset is positive or negative.

The primary goal is to demonstrate a fundamental understanding of the transformer architecture and its application to a classification task using PyTorch. The entire pipeline, from data loading and preprocessing to model training and evaluation, is implemented within the provided Jupyter Notebook. The model successfully achieves over 75% accuracy on the test set.

## Key Concepts Demonstrated

-   **Custom Transformer Implementation**: Building a decoder-only transformer architecture from fundamental components (`AttentionHead`, `MultiHeadAttention`, `FeedForward`, `Block`) without relying on pre-built model classes.
-   **Adaptation for Classification**: Modifying the standard transformer architecture for a classification task by implementing a mean pooling strategy and adding a final linear layer for logit generation.
-   **PyTorch `Dataset` and `DataLoader`**: Creating a custom PyTorch `Dataset` class to handle text data and using `DataLoader` for efficient batching, and shuffling.
-   **Subword Tokenization**: Utilizing the `bert-base-uncased` tokenizer from the Hugging Face `transformers` library for effective subword tokenization.
-   **Data Exploration & Visualization**: Analyzing the dataset's characteristics, such as label distribution and review length, using `pandas` and `matplotlib`.
-   **End-to-End Training Loop**: Implementing a complete training and validation loop in PyTorch, including loss calculation (`CrossEntropyLoss`), backpropagation, and optimization (`AdamW`).

## Project Workflow

1.  **Data Loading and Preparation**: The IMDB movie review dataset is loaded from text files into a Pandas DataFrame.
2.  **Exploratory Data Analysis (EDA)**: The dataset is analyzed to understand its structure, including the distribution of positive/negative labels and the length of reviews.
3.  **PyTorch DataLoader Implementation**: A custom `IMDBDataset` class is created to tokenize reviews on-the-fly. `DataLoader` instances are then set up for the training, validation, and test sets.
4.  **Custom Transformer Architecture**: The `DemoGPT` model is built by assembling custom modules for self-attention, multi-head attention, and feed-forward networks. It's specifically tailored for binary classification.
5.  **Model Training**: The model is trained for 3 epochs using the AdamW optimizer and Cross-Entropy loss. Validation accuracy is monitored after each epoch.
6.  **Model Evaluation**: The trained model's performance is evaluated on the unseen test dataset to measure its final accuracy.

## Model Architecture

The core of this project is a custom-built, decoder-only transformer model. Unlike using a pre-trained model like BERT for fine-tuning, this model is constructed from scratch to demonstrate foundational principles.

The key components are:
-   `AttentionHead`: Implements a single head of scaled dot-product self-attention.
-   `MultiHeadAttention`: Combines multiple `AttentionHead` modules to allow the model to focus on different parts of the sequence simultaneously.
-   `FeedForward`: A simple fully connected feed-forward network applied after the attention layer.
-   `Block`: A standard transformer block that combines multi-head attention and a feed-forward network with residual connections and layer normalization.

For the final classification, the model performs **mean pooling** on the output embeddings of the final transformer block. This aggregates the token-level information into a single vector representation for the entire review, which is then passed through a final linear layer to produce the logits for the "Positive" and "Negative" classes.

## Dataset

The project uses the well-known **IMDB Movie Review Dataset**.
-   **Source**: [Stanford AI Group](https://ai.stanford.edu/~amaas/data/sentiment/)
-   **Contents**: It consists of 50,000 highly polarized movie reviews.
-   **Split**: The data is evenly split into 25,000 reviews for training and 25,000 for testing. Each set has a balanced 12,500 positive and 12,500 negative reviews.

## Results

The model was trained for **3 epochs** on the training dataset.

-   **Validation Accuracy (after 3 epochs)**: **78.96%**
-   **Final Test Accuracy**: The model successfully achieves the project goal of **>75%** on the test set, demonstrating effective learning and generalization.

## How to Run

To run this project on your own machine, follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/SentimentScope.git](https://github.com/your-username/SentimentScope.git)
cd SentimentScope
```

**2. Set Up a Virtual Environment**
It's recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
*(You will need to create a `requirements.txt` file with the following content):*
```
torch
transformers
pandas
matplotlib
numpy
ipykernel
```

**4. Download the Dataset**
The dataset is provided as `aclImdb_v1.tar.gz`. Make sure it is in the root directory of the project. The notebook contains a cell to extract it:
```python
# In the notebook:
!tar -xzf aclImdb_v1.tar.gz
```

**5. Run the Jupyter Notebook**
Launch Jupyter Lab or Jupyter Notebook and open the `SentimentScope_starter.ipynb` file.
```bash
jupyter lab
```
You can then run the cells sequentially to see the entire process.
