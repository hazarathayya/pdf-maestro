import os
import time
import gc
from pypdf import PdfReader, PdfWriter, PdfMerger
from pdf2image import convert_from_path


files = [f for f in os.listdir('./data/pdfs')]
num_files = len(files)
print(num_files)

#we have the pdf's names with spaces which is invalid
def rename_files(pdfs):
    count = 0
    os.chdir("./data/pdfs")
    for i in range(num_files):
        print(pdfs[i], "before")
        os.rename(pdfs[i], f"{count}.pdf")
        print(pdfs[i])
        count = count + 1


# Just Execute once
# rename_files(files)

def count_pages(pdfs):
    num_pages = []
    os.chdir('./data/pdfs')
    for pdf in pdfs:
        pdf_file = open(pdf, 'rb')
        pdf_reader = PdfReader(pdf_file)
        num_pages.append(len(pdf_reader.pages))
        pdf_file.close()
    os.chdir("..")
    os.chdir("..")
    return num_pages


# print(os.getcwd())
file_pages = count_pages(files)



def extract_images(cnt):
    cwd = os.getcwd()
    # print(cwd)
    pdfs_dir = os.path.join(cwd, "pdfs")
    images_dir = os.path.join(cwd, "Images")
    pdf_path = os.path.join(pdfs_dir, f"{cnt}.pdf")
    images = convert_from_path(pdf_path, output_folder=images_dir, fmt="jpeg")
    # print(images)
    os.chdir(images_dir)
    # print(os.getcwd())
    
    for i, image in enumerate(images):
        image.save(f"page_{cnt}_{i}.jpg", "JPEG")

    for file_name in os.listdir(images_dir):
        if file_name.count('-') == 5 and file_name.endswith('.jpg'):
            os.remove(os.path.join(images_dir, file_name))
    os.chdir("..")
    # print(os.getcwd())


im_dir = "./data/Images"



def extract_all_pdfs(pdfs):
    os.chdir("./data")
    cnt = 0
    for pdf in pdfs:
        extract_images(cnt)
        cnt = cnt + 1

extract_all_pdfs(files)


def check_pages(files):
    for pdf in range(len(files)):
        print(f"{files[pdf]} count -> {file_pages[pdf]}")

# check_pages(files)