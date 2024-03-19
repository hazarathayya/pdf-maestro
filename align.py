import os
from pypdf import PdfMerger, PdfWriter, PdfReader

merger = PdfWriter()


def get_pdfs():
    pdfs = []
    for file in os.listdir():
        if file.endswith('.pdf'):
            pdfs.append(file)
    return pdfs


def count_pages(pdfs):
    num_pages = []
    pdf_file = open(pdfs, 'rb')
    pdf_reader = PdfReader(pdf_file)
    num_pages.append(len(pdf_reader.pages))
    return num_pages


pdfs = get_pdfs()
num_pages = count_pages(pdfs[0])


def merge():
    reader1 = PdfReader(pdfs[0])
    for i in range(num_pages[0]):
        merger.add_page(reader1.pages[i])
        merger.pages[i].rotate(90)
    return merger


def download_merged_pdf():
    merger = merge()
    merger.write("aligned.pdf")
    merger.close()


download_merged_pdf()