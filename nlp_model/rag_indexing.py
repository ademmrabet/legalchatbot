import faiss
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from datasets import Dataset

# Load preprocessed and tokenized data (tokenized from previous step)
def load_tokenized_data(tokenized_file_path):
    with open(tokenized_file_path, "r", encoding="utf-8") as file:
        tokenized_data = file.read()
    return tokenized_data

# Create FAISS index
def create_faiss_index(tokenized_data, model_name="bert-base-uncased"):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Encode the tokenized data
    inputs = tokenizer(tokenized_data, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling for embeddings

    # Convert embeddings to numpy array
    faiss_embeddings = embeddings.cpu().numpy()

    # Create a FAISS index
    index = faiss.IndexFlatL2(faiss_embeddings.shape[1])
    index.add(faiss_embeddings)

    return index

# Save the FAISS index
def save_faiss_index(index, faiss_index_path):
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")

# Example usage
if __name__ == "__main__":
    tokenized_file_path = "../data/tokenized/Tunisie-Code-2011-penal_tokenized.txt"  # Path to the tokenized data
    faiss_index_path = "../data/faiss_index.index"  # Path to save the FAISS index

    tokenized_data = load_tokenized_data(tokenized_file_path)
    faiss_index = create_faiss_index(tokenized_data)
    save_faiss_index(faiss_index, faiss_index_path)