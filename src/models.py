import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertConfig, RobertaConfig
from adapters import DistilBertAdapterModel, RobertaAdapterModel, AdapterConfig


# --- PyTorch Dataset ---
class EmbeddingDataset(Dataset):
    """Dataset class for loading embeddings and labels."""
    def __init__(self, embeddings, labels):
        if not isinstance(embeddings, torch.Tensor):
            self.embeddings = torch.tensor(embeddings, dtype=torch.float)
        else:
            self.embeddings = embeddings.float()

        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx],
        }

# --- LSTM Classifier ---
class LSTMClassifier(nn.Module):
    """LSTM Classifier Model."""
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim,
                             hidden_dim,
                             num_layers=num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0, # Dropout only between LSTM layers
                             bidirectional=bidirectional)
        # Input to FC layer is hidden_dim * num_directions
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> e.g., (32, 1, 1536)
        # LSTM outputs:
        #   output: (batch_size, seq_len, num_directions * hidden_size)
        #   hn: (num_layers * num_directions, batch_size, hidden_size)
        #   cn: (num_layers * num_directions, batch_size, hidden_size)
        _, (hn, _) = self.lstm(x)

        # Get hidden state of the last layer
        # If bidirectional, concatenate the forward and backward hidden states of the last layer
        if self.lstm.bidirectional:
            # hn shape is (num_layers*2, batch, hidden_size)
            # Forward hidden state of last layer: hn[-2, :, :]
            # Backward hidden state of last layer: hn[-1, :, :]
            last_hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            # hn shape is (num_layers, batch, hidden_size)
            # Last layer's hidden state: hn[-1, :, :]
            last_hidden = hn[-1, :, :]

        # Pass the final hidden state through the fully connected layer
        out = self.fc(last_hidden) # shape: (batch_size, num_classes)
        return out


# --- Adapter Model Setup ---
def setup_adapter_model(model_type="distilbert", num_labels=3, adapter_config_str="pfeiffer"):
    """Loads a transformer model, adds an adapter and a classification head."""
    if model_type.lower() == "distilbert":
        model_name = "distilbert-base-uncased"
        config = DistilBertConfig.from_pretrained(model_name)
        model = DistilBertAdapterModel.from_pretrained(model_name, config=config)
    elif model_type.lower() == "roberta":
        model_name = "roberta-base"
        config = RobertaConfig.from_pretrained(model_name)
        model = RobertaAdapterModel.from_pretrained(model_name, config=config)
    else:
        raise ValueError("model_type must be 'distilbert' or 'roberta'")

    print(f"Setting up {model_type} with adapter config: {adapter_config_str}")
    adapter_config = AdapterConfig.load(adapter_config_str)
    adapter_name = "classification_adapter"
    model.add_adapter(adapter_name, config=adapter_config)
    model.train_adapter(adapter_name) # Freeze transformer weights, train only adapter

    # Add classification head linked to the adapter
    model.add_classification_head(
        adapter_name,
        num_labels=num_labels,
        id2label={i: f"label_{i}" for i in range(num_labels)} # Generic labels
    )
    print(f"Adapter and classification head added to {model_type}.")
    return model, config


if __name__ == '__main__':
    # Example usage:
    print("Testing model definitions...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test LSTM
    lstm_model = LSTMClassifier(input_dim=1536, num_classes=3).to(device)
    dummy_input_lstm = torch.randn(4, 1, 1536).to(device) # Batch=4, SeqLen=1, Dim=1536
    output_lstm = lstm_model(dummy_input_lstm)
    print(f"LSTM Output Shape: {output_lstm.shape}") # Should be [4, 3]

    # Test DistilBERT Adapter Setup
    try:
        distil_adapter_model, distil_config = setup_adapter_model("distilbert", num_labels=3)
        print("DistilBERT Adapter setup successful.")
        dummy_input_adapter = torch.randn(4, 2, distil_config.dim).to(device) # Batch=4, SeqLen=2, Dim=768
        output_adapter = distil_adapter_model(inputs_embeds=dummy_input_adapter)
        print(f"DistilBERT Adapter Output Logits Shape: {output_adapter.logits.shape}") # Should be [4, 3]
    except Exception as e:
        print(f"Error testing DistilBERT Adapter: {e}")

     # Test RoBERTa Adapter Setup
    try:
        roberta_adapter_model, roberta_config = setup_adapter_model("roberta", num_labels=3)
        print("RoBERTa Adapter setup successful.")
        dummy_input_roberta = torch.randn(4, 2, roberta_config.hidden_size).to(device) # Batch=4, SeqLen=2, Dim=768
        output_roberta = roberta_adapter_model(inputs_embeds=dummy_input_roberta)
        print(f"RoBERTa Adapter Output Logits Shape: {output_roberta.logits.shape}") # Should be [4, 3]
    except Exception as e:
        print(f"Error testing RoBERTa Adapter: {e}")

    print("Model definition tests finished.")
