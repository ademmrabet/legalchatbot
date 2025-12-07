import re

def clean_text(text):
    """Clean the extracted text by removing unwanted characters."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()  # Remove leading and trailing spaces
    return text

def save_cleaned_text(cleaned_text, cleaned_file_path):
    """Save cleaned text to a file."""
    with open(cleaned_file_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)

# Example usage
if __name__ == "__main__":
    extracted_file_path = "../data/raw/Tunisie-Code-2011-penal_extracted.txt"  # Adjust path if necessary
    cleaned_file_path = "../data/cleaned/Tunisie-Code-2011-penal_cleaned.txt"  # Adjust path if necessary

    with open(extracted_file_path, "r", encoding="utf-8") as file:
        extracted_text = file.read()

    cleaned_text = clean_text(extracted_text)
    save_cleaned_text(cleaned_text, cleaned_file_path)
    print(f"Cleaned text saved to {cleaned_file_path}")