import pdfplumber
import os 

RPHY_PDF = "pdfs/CM-SP-R-PHY-I20-250402.pdf"

def _save_sentences_to_file(pdf_filename: str, sentences: list[str]):
    filename = pdf_filename.split(".")[0]
    print("Saving to a txt file")
    with open(file=f"{filename}.txt", mode='w') as f:
        for sentence in sentences:
            f.write(sentence+"\n")
    print(f"Saved file {filename}.txt with {len(sentences)} sentences")

def convert_single_pdf_to_sentences(filename: str):
    page_text = []
    print(f"\nOpening {filename}")
    with pdfplumber.open(filename) as pdf:
        for page in pdf.pages:
            page_text.append(page.extract_text() or "")
            # print(f">>>page_text: {page_text}")
    full_text = "\n".join(page_text)
    # print(f"\n>>>full_text: {full_text}")
    # basic sentence split; for production use nltk/spacy
    sentences = [sentence.strip() for sentence in full_text.split("\n")]
    # print(f">>sentences: {sentences}")
    print(f"{len(sentences)} sentences generated from pdf file")
    return sentences

def convert_pdf_to_sentences(pdfs_path: str, save_to_txt_files: bool = False):
    for file in os.listdir(pdfs_path):
        if not file.endswith(".pdf"):
            continue
        filepath = os.path.join(pdfs_path, file)
        sentences = convert_single_pdf_to_sentences(filepath)
        if save_to_txt_files:
            _save_sentences_to_file(filepath, sentences)
    return sentences