import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        extracted_text += page.get_text("text")  # Extract text from the page
    
    return extracted_text

def save_text(extracted_text, save_path):
    """Save extracted text to a file."""
    with open(save_path, "w", encoding="utf-8") as file:
        file.write(extracted_text)

# Example usage
if __name__ == "__main__":
    pdf_path = "../data/raw/Tunisie-Code-2011-penal.pdf"  # Adjust path if necessary
    extracted_text = extract_text_from_pdf(pdf_path)
    save_path = "../data/raw/Tunisie-Code-2011-penal_extracted.txt"  # Adjust path if necessary
    save_text(extracted_text, save_path)
    print(f"Extracted text saved to {save_path}")