import os
import re
from pypdf import PdfReader, PdfWriter

def get_pdfs(data_dir):
    pdfs = []
    for i in os.listdir(data_dir):
        if i.endswith('.pdf'):
            pdfs.append(i)
    return pdfs

data_dir = os.path.join(os.getcwd(), "data/qa_pdfs")
pdfs = get_pdfs(data_dir)
print(pdfs)
# print(os.getcwd())

""" To cut the indexes and other waste pages in the books """
def preprocessing(data_dir, pdf, n):
    os.chdir(data_dir)
    reader = PdfReader(pdf)
    writer = PdfWriter()
    for i in range(len(reader.pages)):
        if n<=i:
            writer.add_page(reader.pages[i])
    
    with open(pdf, "wb") as f:
        writer.write(f)
    os.chdir("..")
    os.chdir("..")

# preprocessing(data_dir, pdfs[0], 19)

def write_txt(work_dir, pdfs):
    txt = ""
    for i in pdfs:
        os.chdir(work_dir)
        reader = PdfReader(i)
        for page in reader.pages:
            txt = txt +  page.extract_text()

        os.chdir("..")
        target_dir = os.getcwd()
        file_name = f"{target_dir}/qa_txts/{i[:-4]}.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(txt)
        print(file_name)
    os.chdir("..")


txt_dir = os.path.join(os.getcwd(), "data/qa_pdfs")
write_txt(txt_dir, pdfs)
print(f"current working directory", os.getcwd())