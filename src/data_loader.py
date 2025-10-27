import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import os

def load_data(train_file='issues_train.csv', test_file='issues_test.csv', data_dir='data'):
    """Loads training and testing data, drops unused columns, and handles NaNs."""
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)

    try:
        traindf = pd.read_csv(train_path)
        testdf = pd.read_csv(test_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure '{train_file}' and '{test_file}' are in the '{data_dir}' directory.")
        return None, None

    # Drop columns
    traindf.drop(['repo', 'created_at'], axis=1, inplace=True, errors='ignore')
    testdf.drop(['repo', 'created_at'], axis=1, inplace=True, errors='ignore')

    # Handle NaNs
    initial_train_rows = len(traindf)
    initial_test_rows = len(testdf)
    traindf.dropna(inplace=True)
    testdf.dropna(inplace=True)
    print(f"Dropped {initial_train_rows - len(traindf)} rows with NaNs from training data.")
    print(f"Dropped {initial_test_rows - len(testdf)} rows with NaNs from testing data.")


    return traindf, testdf

def preprocess_text_series(text_series):
    """Applies text preprocessing to a pandas Series."""
    processed_texts = []
    for text in text_series:
        if not isinstance(text, str):
            text = str(text) if text is not None else ''

        # Remove code blocks ```...```
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        # Remove new lines
        text = re.sub(r'\n', ' ', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        # Remove digits
        text = re.sub(r'\d+', ' ', text)
        # Remove special characters (keep letters, numbers, spaces, ?)
        text = re.sub(r'[^a-zA-Z0-9?\s]', ' ', text)
        # Replace multiple spaces with a single space and strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        processed_texts.append(text)
    return processed_texts

def encode_labels(train_df, test_df, label_col='label'):
    """Encodes labels using LabelEncoder fitted on the training data."""
    le = LabelEncoder()
    train_df[label_col] = le.fit_transform(train_df[label_col])
    test_df[label_col] = le.transform(test_df[label_col]) # Use transform only on test data
    print("Labels encoded.")
    print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return train_df, test_df, le # Return encoder for potential inverse transform

if __name__ == '__main__':
    # Example usage:
    print("Testing data_loader...")
    train_df, test_df = load_data(data_dir='../data') # Adjust path if running directly
    if train_df is not None and test_df is not None:
        print("Data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        print("\nPreprocessing text...")
        train_df['title'] = preprocess_text_series(train_df['title'])
        train_df['body'] = preprocess_text_series(train_df['body'])
        test_df['title'] = preprocess_text_series(test_df['title'])
        test_df['body'] = preprocess_text_series(test_df['body'])
        print("Text preprocessing done.")
        print("Sample processed title:", train_df['title'].iloc[0])

        print("\nEncoding labels...")
        train_df, test_df, label_encoder = encode_labels(train_df, test_df)
        print("Label encoding done.")
        print("Sample encoded label:", train_df['label'].iloc[0])
    else:
        print("Data loading failed.")
