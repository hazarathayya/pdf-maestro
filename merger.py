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
    for pdf in pdfs:
        pdf_file = open(pdf, 'rb')
        pdf_reader = PdfReader(pdf_file)
        num_pages.append(len(pdf_reader.pages))
    return num_pages


pdfs = get_pdfs()
num_pages = count_pages(pdfs)
max_len = max(num_pages[0], num_pages[1])


def merge():
    reader1 = PdfReader(pdfs[0])
    reader2 = PdfReader(pdfs[1])
    for i in range(max_len):
        if i<num_pages[0]:
            merger.add_page(reader1.pages[i])
        if i<num_pages[1]:
            merger.add_page(reader2.pages[i])
    return merger


def download_merged_pdf():
    merger = merge()
    merger.write("merged.pdf")
    merger.close()


download_merged_pdf()