import sys
import os
from nlp_model.rag_model import generate_answer
from preprocessing_pipeline import run_preprocessing_pipeline
from nlp_model.rag_indexing import create_faiss_index, save_faiss_index
from transformers import BertTokenizer, BertForQuestionAnswering, RagRetriever, RagTokenForGeneration

# Add preprocessing path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing')))

# Function to load tokenized data
def load_tokenized_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        tokenized_data = file.readlines()  # Assuming one line per tokenized sentence
    return tokenized_data

# Initialize RAG model and tokenizer
def initialize_rag_model():
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_path="./data/faiss_index.index")
    return model, tokenizer, retriever

def main():
    # Run the preprocessing pipeline (extract, clean, tokenize)
    run_preprocessing_pipeline()

    # Create FAISS index for the tokenized data
    tokenized_file_path = "./data/tokenized/Tunisie-Code-2011-penal_tokenized.txt"
    faiss_index_path = "./data"
    tokenized_data = load_tokenized_data(tokenized_file_path)
    faiss_index = create_faiss_index(tokenized_data)
    save_faiss_index(faiss_index, faiss_index_path)

    # Initialize the RAG model and tokenizer
    rag_model, rag_tokenizer, retriever = initialize_rag_model()

    # Generate an answer using RAG with Grok
    question = "Quelle est la peine pour le vol?"
    answer = generate_answer(question, faiss_index, rag_model, rag_tokenizer, retriever)
    print(f"Generated Answer: {answer}")

if __name__ == "__main__":
    main()