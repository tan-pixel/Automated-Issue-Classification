import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import pickle
import time

# --- Helper Function for DataLoader ---
def _create_text_dataloader(text_data, batch_size):
    """Creates a DataLoader for text data, ensuring it's a list of strings."""
    if isinstance(text_data, pd.Series):
        text_data = text_data.tolist()
    elif not isinstance(text_data, list):
        text_data = list(text_data)
    # Ensure all elements are strings
    text_data = [str(item) if item is not None else '' for item in text_data]
    return DataLoader(text_data, batch_size=batch_size, shuffle=False)

# --- SentenceTransformer Embeddings ---
def get_sentence_transformer_embeddings(texts, model_name="all-mpnet-base-v2", batch_size=32, device='cpu'):
    """Generates embeddings using a SentenceTransformer model."""
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    model.eval()
    dataloader = _create_text_dataloader(texts, batch_size)
    all_embeddings = []
    print(f"Generating embeddings with SentenceTransformer (Batch size: {batch_size})...")
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings = model.encode(batch, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings.cpu()) # Move to CPU for consistent storage
            if (i + 1) % 50 == 0:
                 print(f"  Processed { (i + 1) * batch_size } items...")
    elapsed_time = time.time() - start_time
    print(f"SentenceTransformer embedding generation took {elapsed_time:.2f} seconds.")
    return torch.cat(all_embeddings, dim=0)

# --- RoBERTa Embeddings ---
def get_roberta_embeddings(texts, model_name="FacebookAI/roberta-base", batch_size=16, device='cpu'):
    """Generates [CLS] token embeddings using a RoBERTa model."""
    print(f"Loading RoBERTa model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    dataloader = _create_text_dataloader(texts, batch_size)
    all_cls_embeddings = []
    print(f"Generating embeddings with RoBERTa (Batch size: {batch_size})...")
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_texts = [str(item) if item is not None else '' for item in batch]
            tokenized_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**tokenized_batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu() # CLS token, move to CPU
            all_cls_embeddings.append(cls_embeddings)
            if (i + 1) % 50 == 0:
                 print(f"  Processed { (i + 1) * batch_size } items...")
    elapsed_time = time.time() - start_time
    print(f"RoBERTa embedding generation took {elapsed_time:.2f} seconds.")
    return torch.cat(all_cls_embeddings, dim=0)

# --- Main Embedding Generation and Saving ---
def generate_and_save_embeddings(train_df, test_df, save_dir='embeddings', batch_size=32, device='cpu', force_regenerate=False):
    """
    Generates or loads embeddings for train and test sets for all 4 combinations.
    Saves/loads embeddings as numpy arrays using pickle.
    Applies MinMaxScaler fitted on training data.

    Returns:
        tuple: (x_train_all, y_train_all, x_test_all, y_test_all)
               where each *_all is a 3D numpy array (num_combinations, num_samples, embedding_dim)
               and y_*_all are 2D numpy arrays (num_combinations, num_samples)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define file paths
    embed_file_map = {
        'train_body_st': os.path.join(save_dir, 'train_body_st.pkl'),
        'test_body_st': os.path.join(save_dir, 'test_body_st.pkl'),
        'train_title_st': os.path.join(save_dir, 'train_title_st.pkl'),
        'test_title_st': os.path.join(save_dir, 'test_title_st.pkl'),
        'train_body_rob': os.path.join(save_dir, 'train_body_rob.pkl'),
        'test_body_rob': os.path.join(save_dir, 'test_body_rob.pkl'),
        'train_title_rob': os.path.join(save_dir, 'train_title_rob.pkl'),
        'test_title_rob': os.path.join(save_dir, 'test_title_rob.pkl'),
    }

    embeddings = {}

    # Generate or load each embedding type
    regenerated_any = False
    for key, filepath in embed_file_map.items():
        if os.path.exists(filepath) and not force_regenerate:
            print(f"Loading cached embeddings from {filepath}...")
            with open(filepath, 'rb') as f:
                embeddings[key] = pickle.load(f)
        else:
            regenerated_any = True
            print(f"Generating embeddings for {key}...")
            data_type, text_part, model_type = key.split('_') # e.g., 'train', 'body', 'st'
            df = train_df if data_type == 'train' else test_df
            texts = df[text_part]

            if model_type == 'st':
                embeds_tensor = get_sentence_transformer_embeddings(texts, batch_size=batch_size, device=device)
            elif model_type == 'rob':
                # Use smaller batch size for RoBERTa if needed
                rob_batch_size = max(4, batch_size // 2) if device=='cuda' else batch_size
                embeds_tensor = get_roberta_embeddings(texts, batch_size=rob_batch_size, device=device)
            else:
                raise ValueError(f"Unknown model type in key: {key}")

            embeddings[key] = embeds_tensor.numpy() # Store as numpy array
            print(f"Saving embeddings to {filepath}...")
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings[key], f)

    if regenerated_any:
        print("Finished generating/loading all base embeddings.")

    # Combine embeddings
    print("Combining and scaling embeddings for 4 combinations...")
    x_train_list = []
    x_test_list = []
    scalers = [] # Store one scaler per combination

    combinations = [
        ('st', 'st'), ('st', 'rob'), ('rob', 'st'), ('rob', 'rob')
    ]

    for i, (body_m, title_m) in enumerate(combinations):
        train_body_key = f'train_body_{body_m}'
        train_title_key = f'train_title_{title_m}'
        test_body_key = f'test_body_{body_m}'
        test_title_key = f'test_title_{title_m}'

        x_train_comb = np.concatenate((embeddings[train_body_key], embeddings[train_title_key]), axis=1)
        x_test_comb = np.concatenate((embeddings[test_body_key], embeddings[test_title_key]), axis=1)

        # Scale features
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train_comb)
        x_test_scaled = scaler.transform(x_test_comb) # Transform test with scaler fitted on train

        x_train_list.append(x_train_scaled)
        x_test_list.append(x_test_scaled)
        scalers.append(scaler) # Optional: save scalers if needed elsewhere
        print(f"Combination {i+1} ({body_m}/{title_m}) processed and scaled.")

    # Stack into 3D arrays
    x_train_all = np.array(x_train_list)
    x_test_all = np.array(x_test_list)

    # Prepare labels (simple repetition as in notebook)
    y_train_np = train_df["label"].values
    y_test_np = test_df["label"].values
    y_train_all = np.array([y_train_np] * 4)
    y_test_all = np.array([y_test_np] * 4)

    print("Embedding processing complete.")
    print(f"x_train shape: {x_train_all.shape}, y_train shape: {y_train_all.shape}")
    print(f"x_test shape: {x_test_all.shape}, y_test shape: {y_test_all.shape}")

    return x_train_all, y_train_all, x_test_all, y_test_all


if __name__ == '__main__':
    # Example Usage (requires data files in ../data)
    from data_loader import load_data, preprocess_text_series, encode_labels
    print("Testing embedder...")
    train_df_raw, test_df_raw = load_data(data_dir='../data')
    if train_df_raw is not None and test_df_raw is not None:
        train_df_proc, test_df_proc, _ = encode_labels(train_df_raw.copy(), test_df_raw.copy())
        train_df_proc['title'] = preprocess_text_series(train_df_proc['title'])
        train_df_proc['body'] = preprocess_text_series(train_df_proc['body'])
        test_df_proc['title'] = preprocess_text_series(test_df_proc['title'])
        test_df_proc['body'] = preprocess_text_series(test_df_proc['body'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Generate (or load if exist) embeddings
        xtr, ytr, xte, yte = generate_and_save_embeddings(
            train_df_proc, test_df_proc, save_dir='../embeddings', batch_size=16, device=device, force_regenerate=False
        )
        print("Embedder test finished.")
    else:
        print("Data loading failed in embedder test.")
