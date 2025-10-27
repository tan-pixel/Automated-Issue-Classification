import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Project modules
from data_loader import load_data, preprocess_text_series, encode_labels
from embedder import generate_and_save_embeddings
from models import EmbeddingDataset, LSTMClassifier, setup_adapter_model

def train_evaluate_lstm(train_loader, test_loader, input_dim, num_classes, device, lr=0.001, epochs=100):
    """Trains and evaluates the LSTMClassifier model."""
    print("Initializing LSTM model...")
    model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training LSTM for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            embeddings = batch["embedding"].unsqueeze(1).to(device) # Add seq_len dimension
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
             print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    train_time = time.time() - start_time
    print(f"LSTM Training finished in {train_time:.2f} seconds.")

    # Evaluation
    print("Evaluating LSTM model...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch["embedding"].unsqueeze(1).to(device)
            labels = batch["label"].to(device)
            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def train_evaluate_adapter(model_type, train_loader, test_loader, num_classes, device, lr=1e-4, epochs=50, adapter_config_str="pfeiffer"):
    """Trains and evaluates a Transformer model with adapters."""
    print(f"Initializing {model_type} model with adapters...")
    try:
        model, config = setup_adapter_model(model_type=model_type, num_labels=num_classes, adapter_config_str=adapter_config_str)
        model.to(device)
    except Exception as e:
        print(f"Error setting up adapter model: {e}")
        return None, None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Determine expected input shape
    hidden_size = config.dim if hasattr(config, 'dim') else config.hidden_size
    seq_len = 2 # Assuming concatenation of 2 embeddings

    print(f"Training {model_type} Adapter model for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            # Reshape embeddings: [batch, 2*hidden] -> [batch, seq_len=2, hidden]
            try:
                embeddings = batch["embedding"].view(batch["embedding"].shape[0], seq_len, hidden_size).to(device)
                labels = batch["label"].to(device)
            except RuntimeError as e:
                print(f"Error reshaping embeddings or moving to device: {e}")
                print(f"Embedding shape: {batch['embedding'].shape}, Expected hidden size: {hidden_size}")
                continue # Skip batch

            optimizer.zero_grad()
            outputs = model(inputs_embeds=embeddings) # Use default head
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
             print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    train_time = time.time() - start_time
    print(f"{model_type} Adapter Training finished in {train_time:.2f} seconds.")

    # Evaluation
    print(f"Evaluating {model_type} Adapter model...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            try:
                embeddings = batch["embedding"].view(batch["embedding"].shape[0], seq_len, hidden_size).to(device)
                labels = batch["label"].to(device)
            except RuntimeError as e:
                print(f"Error reshaping test embeddings: {e}")
                continue

            outputs = model(inputs_embeds=embeddings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

def train_evaluate_sklearn(model, model_name, x_train, y_train, x_test, y_test, use_grid_search=False, param_grid=None):
    """Trains and evaluates scikit-learn compatible models (SVM, RF, XGB, LGBM)."""
    print(f"Training {model_name}...")
    start_time = time.time()

    if use_grid_search and param_grid:
        print(f"Running GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print(f"Best Parameters for {model_name}:", grid_search.best_params_)
        print(f"Best CV Accuracy for {model_name}: {grid_search.best_score_:.4f}")
        final_model = grid_search.best_estimator_
    else:
        print(f"Using default/pre-set parameters for {model_name}.")
        # Ensure model has necessary parameters set if not using grid search
        final_model = model
        final_model.fit(x_train, y_train)

    train_time = time.time() - start_time
    print(f"{model_name} Training finished in {train_time:.2f} seconds.")

    print(f"Evaluating {model_name} model...")
    y_pred = final_model.predict(x_test)
    return y_test, y_pred


def print_results(model_name, y_true, y_pred):
    """Calculates and prints evaluation metrics."""
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        print(f"{model_name} evaluation failed or produced no results.")
        return

    if len(y_true) != len(y_pred):
         print(f"Error: Mismatch between number of true labels ({len(y_true)}) and predictions ({len(y_pred)}) for {model_name}.")
         return

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted') # Default is weighted

    print(f"\n--- {model_name} Evaluation Results ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (Weighted): {precision:.4f}")
    print(f"  Recall (Weighted): {recall:.4f}")
    print(f"  F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("--------------------------------------")


def main(args):
    """Main function to run the training and evaluation pipeline."""
    start_pipeline_time = time.time()

    # --- 1. Load and Preprocess Data ---
    print("Step 1: Loading and preprocessing data...")
    train_df_raw, test_df_raw = load_data(data_dir=args.data_dir)
    if train_df_raw is None: return

    train_df, test_df, label_encoder = encode_labels(train_df_raw.copy(), test_df_raw.copy())
    train_df['title'] = preprocess_text_series(train_df['title'])
    train_df['body'] = preprocess_text_series(train_df['body'])
    test_df['title'] = preprocess_text_series(test_df['title'])
    test_df['body'] = preprocess_text_series(test_df['body'])
    print("Data loading and preprocessing complete.")

    # --- 2. Generate/Load Embeddings ---
    print("\nStep 2: Generating or Loading Embeddings...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    x_train_all, y_train_all, x_test_all, y_test_all = generate_and_save_embeddings(
        train_df, test_df, save_dir=args.embeddings_dir, batch_size=args.batch_size, device=device, force_regenerate=args.force_regenerate_embeddings
    )
    print("Embeddings ready.")

    # --- 3. Select Data for Chosen Combination ---
    print(f"\nStep 3: Selecting data for embedding combination index {args.embedding_combo}...")
    if not (0 <= args.embedding_combo < 4):
        print("Error: embedding_combo index must be between 0 and 3.")
        return

    x_train = x_train_all[args.embedding_combo]
    y_train = y_train_all[args.embedding_combo]
    x_test = x_test_all[args.embedding_combo]
    y_test = y_test_all[args.embedding_combo]
    input_dim = x_train.shape[1]
    num_classes = len(label_encoder.classes_)
    print(f"Selected data shapes: x_train={x_train.shape}, x_test={x_test.shape}")

    # --- 4. Train and Evaluate Selected Model ---
    print(f"\nStep 4: Training and evaluating model: {args.model_type}...")

    y_true, y_pred = None, None # Initialize

    if args.model_type == 'LSTM':
        train_dataset = EmbeddingDataset(x_train, y_train)
        test_dataset = EmbeddingDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        y_true, y_pred = train_evaluate_lstm(train_loader, test_loader, input_dim, num_classes, device, lr=args.lr, epochs=args.epochs)

    elif args.model_type == 'DistilBERTAdapter':
        train_dataset = EmbeddingDataset(x_train, y_train)
        test_dataset = EmbeddingDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        y_true, y_pred = train_evaluate_adapter('distilbert', train_loader, test_loader, num_classes, device, lr=args.lr, epochs=args.epochs)

    elif args.model_type == 'RoBERTaAdapter':
        # Use smaller batch size for RoBERTa adapters if necessary
        adapter_batch_size = max(4, args.batch_size // 2) if device == 'cuda' else args.batch_size
        train_dataset = EmbeddingDataset(x_train, y_train)
        test_dataset = EmbeddingDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=adapter_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=adapter_batch_size, shuffle=False)
        y_true, y_pred = train_evaluate_adapter('roberta', train_loader, test_loader, num_classes, device, lr=args.lr, epochs=args.epochs) # Use lower default LR potentially

    elif args.model_type == 'SVM':
        param_grid = { 'C': [1.5, 1.6, 1.7], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 0.1] } if args.use_grid_search else None
        # Default model with params from notebook
        model = SVC(C=1.6, probability=True, random_state=42)
        y_true, y_pred = train_evaluate_sklearn(model, 'SVM', x_train, y_train, x_test, y_test, args.use_grid_search, param_grid)

    elif args.model_type == 'RF':
        param_grid = { 'n_estimators': [100, 300], 'max_depth': [5, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2] } if args.use_grid_search else None
        # Default model with params from notebook (using combo 3's best as default)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
        y_true, y_pred = train_evaluate_sklearn(model, 'Random Forest', x_train, y_train, x_test, y_test, args.use_grid_search, param_grid)

    elif args.model_type == 'XGB':
         # Using fixed params from notebook
        model = XGBClassifier(
            colsample_bytree=1.0, learning_rate=0.1, max_depth=2, n_estimators=100,
            subsample=1.0, eval_metric='mlogloss', random_state=7, n_jobs=-1
        )
        y_true, y_pred = train_evaluate_sklearn(model, 'XGBoost', x_train, y_train, x_test, y_test)

    elif args.model_type == 'LGBM':
         # Using fixed params from notebook
        model = LGBMClassifier(
            colsample_bytree=0.7, learning_rate=0.1, max_depth=5, min_child_samples=20,
            n_estimators=100, num_leaves=15, reg_alpha=0.1, reg_lambda=0.0,
            subsample=0.7, random_state=7, n_jobs=-1
        )
        y_true, y_pred = train_evaluate_sklearn(model, 'LightGBM', x_train, y_train, x_test, y_test)

    else:
        print(f"Error: Unknown model_type '{args.model_type}'")
        return

    # --- 5. Print Results ---
    print_results(f"{args.model_type} (Combo {args.embedding_combo})", y_true, y_pred)

    end_pipeline_time = time.time()
    print(f"\nTotal pipeline execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models for GitHub issue classification.")
    parser.add_argument('--model_type', type=str, required=True, choices=['LSTM', 'DistilBERTAdapter', 'RoBERTaAdapter', 'SVM', 'RF', 'XGB', 'LGBM'], help='Type of model to train.')
    parser.add_argument('--embedding_combo', type=int, required=True, help='Embedding combination index (0-3). 0:S/S, 1:S/R, 2:R/S, 3:R/R')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for DL models (default: 100 for LSTM, adjust for others).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DL training and embedding generation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for DL models.')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the CSV data files.')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='Directory to save/load cached embeddings.')
    parser.add_argument('--use_grid_search', action='store_true', help='Perform GridSearchCV for SVM and RF models (can be slow).')
    parser.add_argument('--force_regenerate_embeddings', action='store_true', help='Force regeneration of embeddings even if cached files exist.')

    args = parser.parse_args()

    # Adjust default epochs based on model type if not overridden
    if args.model_type == 'DistilBERTAdapter' and args.epochs == 100: # Default check
        args.epochs = 50 # Lower default for DistilBERT Adapters
        print(f"Adjusting default epochs for DistilBERTAdapter to {args.epochs}")
    elif args.model_type == 'RoBERTaAdapter' and args.epochs == 100: # Default check
         args.epochs = 30 # Lower default for RoBERTa Adapters
         print(f"Adjusting default epochs for RoBERTaAdapter to {args.epochs}")


    main(args)
