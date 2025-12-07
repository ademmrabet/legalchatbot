from preprocessing.extract_text import extract_text_from_pdf, save_text
from preprocessing.text_cleaning import clean_text, save_cleaned_text
from preprocessing.tokenization import tokenize_data, save_tokenized_data
from transformers import BertTokenizer
import os

# Preprocessing pipeline
def run_preprocessing_pipeline():
    # Paths for input and output files
    pdf_path = "./data/raw/Tunisie-Code-2011-penal.pdf"  # Adjust this path if needed
    extracted_text_path = "./data/raw/Tunisie-Code-2011-penal_extracted.txt"
    cleaned_text_path = "./data/cleaned/Tunisie-Code-2011-penal_cleaned.txt"
    tokenized_file_path = "./data/tokenized/Tunisie-Code-2011-penal_tokenized.txt"

    # 1. Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_path)
    save_text(extracted_text, extracted_text_path)  # Save extracted text temporarily
    print(f"Extracted text saved to {extracted_text_path}")

    # 2. Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    save_text(cleaned_text, cleaned_text_path)  # Save cleaned text temporarily
    print(f"Cleaned text saved to {cleaned_text_path}")

    # 3. Tokenize the cleaned text
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_data = tokenize_data(cleaned_text, tokenizer)

    # Save tokenized data (for future use or analysis)
    save_tokenized_data(tokenized_data, tokenized_file_path)

    print("Preprocessing pipeline completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    run_preprocessing_pipeline()