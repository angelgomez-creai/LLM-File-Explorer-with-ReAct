class PDFSearchTool:
    def __init__(self, pdf_loader):
        self.pdf_loader = pdf_loader
    
    def __call__(self, params):
        page = int(params)
        return self.pdf_loader.search_page(page)