from transformers import BertTokenizer
from datasets import Dataset
import os

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def tokenize_data(text, tokenizer):
    """Tokenize the cleaned text."""
    return tokenizer(text, padding=True, truncation=True, return_tensors="pt")

def save_tokenized_data(tokenized_data, tokenized_file_path):

    directory = os.path.dirname(tokenized_file_path)
    create_directory(directory)

    """Save tokenized data to a file."""
    with open(tokenized_file_path, "w", encoding="utf-8") as file:
        file.write(str(tokenized_data))
    print(f"Tokenized data saved to {tokenized_file_path}")

# Example usage
if __name__ == "__main__":
    cleaned_file_path = "../data/cleaned/Tunisie-Code-2011-penal_cleaned.txt"  # Adjust path if necessary
    tokenized_file_path = "../data/tokenized/Tunisie-Code-2011-penal_tokenized.txt"  # Adjust path if necessary

    with open(cleaned_file_path, "r", encoding="utf-8") as file:
        cleaned_text = file.read()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_data = tokenize_data(cleaned_text, tokenizer)
    save_tokenized_data(tokenized_data, tokenized_file_path)
    print(f"Tokenized data saved to {tokenized_file_path}")