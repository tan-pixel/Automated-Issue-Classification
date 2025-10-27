# GitHub Issue Report Classification

## Overview

This project classifies GitHub issue reports into categories ('bug', 'feature', 'question') using NLP techniques. It replicates and structures the analysis from a Jupyter Notebook (`notebooks/nlp_project.ipynb`), exploring various text embedding methods (Sentence Transformers, RoBERTa) and machine learning classifiers (LSTM, SVM, Random Forest, XGBoost, LightGBM, Transformer Adapters).

## Dataset

The project uses the GitHub issue report dataset from the [NLBSE 2024 Challenge](https://nlbse2024.github.io/tools/).

- **Training Data**: `issues_train.csv`
- **Testing Data**: `issues_test.csv`

### Download the data

Create a `data/` directory in the project root and place the downloaded CSV files there:

```bash
mkdir data
wget -P data/ https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/data/issues_train.csv
wget -P data/ https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/data/issues_test.csv
```

> **Note**: Ensure `wget` is installed or download the files manually.

## Features

- **Text Preprocessing**: Cleans issue titles and bodies
- **Text Embeddings**: Generates embeddings using SentenceTransformer (`all-mpnet-base-v2`) and RoBERTa (`FacebookAI/roberta-base`). Caches embeddings in the `embeddings/` directory
- **Embedding Combinations**: Explores four combinations for title and body embeddings
- **Classification Models**: Trains and evaluates:
  - LSTM
  - DistilBERT+Adapters
  - RoBERTa+Adapters
  - SVM
  - Random Forest
  - XGBoost
  - LightGBM

## Project Structure

```
issue-report-classification/
├── .gitignore
├── data/                 # Data files (needs to be created and populated)
│   ├── issues_train.csv
│   └── issues_test.csv
├── embeddings/           # Cached embeddings (created automatically)
├── notebooks/
│   └── nlp_project.ipynb # Original notebook
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── data_loader.py    # Loads and preprocesses data
    ├── embedder.py       # Generates text embeddings
    ├── models.py         # Custom model definitions (LSTM, Dataset)
    └── train_evaluate.py # Main training and evaluation script
```

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tan-pixel/Automated-Issue-Classification.git
   cd issue-report-classification
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download data** (as described in the Dataset section)

5. **GPU**: A CUDA-enabled GPU is highly recommended for faster embedding generation and deep learning model training. The scripts will attempt to use CUDA if available.

## Usage

Run the main script `src/train_evaluate.py` from the project root directory.

### Examples

```bash
# Train and evaluate the LSTM model using the first embedding combination (Sent/Sent)
python src/train_evaluate.py --model_type LSTM --embedding_combo 0

# Train and evaluate SVM using the third embedding combination (Rob/Sent)
python src/train_evaluate.py --model_type SVM --embedding_combo 2

# Train and evaluate LightGBM using the fourth embedding combination (Rob/Rob)
python src/train_evaluate.py --model_type LGBM --embedding_combo 3

# Train and evaluate DistilBERT Adapters using the second combo (Sent/Rob)
python src/train_evaluate.py --model_type DistilBERTAdapter --embedding_combo 1 --epochs 50

# Train and evaluate RoBERTa Adapters using the first combo (Sent/Sent)
python src/train_evaluate.py --model_type RoBERTaAdapter --embedding_combo 0 --epochs 30

# Run Random Forest with GridSearchCV (takes longer)
python src/train_evaluate.py --model_type RF --embedding_combo 0 --use_grid_search

# Run SVM with GridSearchCV (takes longer)
python src/train_evaluate.py --model_type SVM --embedding_combo 2 --use_grid_search
```

### Arguments

- `--model_type`: Choose from `LSTM`, `DistilBERTAdapter`, `RoBERTaAdapter`, `SVM`, `RF`, `XGB`, `LGBM`
- `--embedding_combo`: Index of the embedding combination (0-3):
  - `0`: Body=SentenceTransformer, Title=SentenceTransformer
  - `1`: Body=SentenceTransformer, Title=RoBERTa
  - `2`: Body=RoBERTa, Title=SentenceTransformer
  - `3`: Body=RoBERTa, Title=RoBERTa
- `--epochs` *(optional, default=100 for LSTM, 200 for DistilBERT, 100 for RoBERTa)*: Number of training epochs for deep learning models
- `--batch_size` *(optional, default=32)*: Batch size for embedding generation and DL model training
- `--lr` *(optional, default=0.001)*: Learning rate for deep learning models
- `--use_grid_search` *(optional, flag)*: Use GridSearchCV for SVM and RF (slower). If not set, uses pre-defined 'best' parameters from the notebook
- `--force_regenerate_embeddings` *(optional, flag)*: Force regeneration of embeddings even if cached files exist

## Results

Evaluation metrics (Accuracy, Precision, Recall, F1-score) will be printed to the console after training each model. Results may vary slightly due to different random seeds or package versions. Refer to the original notebook or run the scripts for specific numbers.