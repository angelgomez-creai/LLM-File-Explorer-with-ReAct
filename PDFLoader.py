import pdfplumber
import random

class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pages_text = {}
        
        with pdfplumber.open(self.pdf_path) as pdf:
            self.total_pages = len(pdf.pages)
        
        self.pages_not_searched = list(range(1, self.total_pages + 1))

    def search_page(self, page_number):
        if page_number in self.pages_text:
            return self.pages_text[page_number]

        if page_number < 1 or page_number > self.total_pages:
            return f"Error: Page {page_number} does not exist. PDF has {self.total_pages} pages."

        # Page indices are 0-based in pdfplumber
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            text = page.extract_text()
            self.pages_text[page_number] = text or ""
            self.pages_not_searched.remove(page_number)
            return text

    def get_random_page_not_searched(self):
        return random.choice(self.pages_not_searched)

if __name__ == "__main__":
    pdf_loader = PDFLoader("relativity.pdf")
    print(pdf_loader.search_page(260))
