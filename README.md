SentimentScope: Sentiment Analysis with a Transformer from Scratch
1. Project Overview
This project involves building, training, and evaluating a transformer-based neural network for sentiment analysis. The model is constructed from scratch using PyTorch and is trained on the popular IMDB movie review dataset to classify reviews as either positive or negative. The primary goal is to achieve a classification accuracy of over 75% on the unseen test dataset.

This project serves as a practical, hands-on guide to understanding the internal components of a transformer architecture and adapting it for a binary classification task.

Core Technologies:

Python

PyTorch

Hugging Face Transformers (for the tokenizer)

Pandas

Matplotlib

2. Learning Objectives
By completing this project, you will demonstrate an understanding of:

Data Loading and Preprocessing: Loading a text-based dataset, performing exploratory data analysis, and preparing it for a deep learning model using a custom PyTorch Dataset and DataLoader.

Transformer Architecture: Implementing the core components of a transformer model, including Self-Attention, Multi-Head Attention, and Feed-Forward layers.

Model Customization: Adapting a standard transformer architecture for a classification task by adding a pooling mechanism and a final linear classification layer.

Model Training and Evaluation: Implementing a complete training loop in PyTorch with a loss function (Cross-Entropy) and an optimizer (AdamW), while monitoring performance with a validation set.

Final Testing: Evaluating the final model's performance on the test set to determine its generalization capability.

3. Dataset
The project uses the Large Movie Review Dataset (IMDB). This dataset contains 50,000 highly polarized movie reviews, split evenly into 25,000 for training and 25,000 for testing.

The data is organized into the following directory structure:

aclImdb/
├── train/
│   ├── pos/    # 12,500 positive training reviews
│   ├── neg/    # 12,500 negative training reviews
├── test/
│   ├── pos/    # 12,500 positive testing reviews
│   ├── neg/    # 12,500 negative testing reviews

The model is trained to predict a label of 1 for positive reviews and 0 for negative reviews.

4. Setup and Installation
To run this project, you need to have Python 3 installed. You can then install the necessary libraries using pip.

Clone the repository or download the project files.

Install the required dependencies:

pip install torch pandas transformers matplotlib numpy

5. How to Run the Project
Extract the Dataset: The first code cell in the SentimentScope_starter.ipynb notebook contains the command to extract the dataset archive:

# Make sure to uncomment this line in the notebook before running!
!tar -xzf aclImdb_v1.tar.gz

This will create the aclIMDB folder in your project directory.

Open the Jupyter Notebook:

jupyter notebook "SentimentScope_starter (5).ipynb"

Run the Cells: Execute the cells in the notebook sequentially from top to bottom. The notebook is structured to guide you through each step of the process:

Data loading and exploration.

Tokenizer testing.

Implementation of the PyTorch Dataset and DataLoader.

Definition of the DemoGPT transformer model.

Training loop execution.

Final evaluation on the test set.

6. Project Results and Conclusion
The model is trained for 3 epochs and successfully achieves a final test accuracy of over 78%, exceeding the project goal of 75%.

Key Takeaways:

Transformer Versatility: This project demonstrates that transformer architectures are not just for language generation but can be effectively adapted for classification tasks.

Importance of Pooling: For sequence classification, aggregating token-level embeddings into a single vector (e.g., via mean pooling) is a crucial step to feed into a final classification layer.

End-to-End PyTorch Workflow: The notebook provides a complete example of a deep learning workflow in PyTorch, from custom data handling to model training and evaluation.
