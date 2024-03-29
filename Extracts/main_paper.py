import os
import re
from pypdf import PdfReader

def get_pdfs():
    pdfs = []
    for i in os.listdir(os.path.join(os.getcwd(), "papers")):
        if i.endswith('.pdf'):
            pdfs.append(i)
    return pdfs

pdfs = get_pdfs()
# print(pdfs)
# print(os.getcwd())

def write_txt(work_dir):
    os.chdir('./papers')
    for i in pdfs:
        reader = PdfReader(i)
        page = reader.pages[0]
        file_name = f"{work_dir}/texts/{i[:-4]}"
        txt = page.extract_text()
        result = re.search(r'(.+?)(?=\bI\. I NTRODUCTION\b)', txt, re.DOTALL)
        if result:
            extracted_text = result.group(1).strip()
            # print(extracted_text)
        else:
            print("Pattern not found.")

        change_text(extracted_text)

        with open(file_name, "w", encoding="utf-8") as file:
            file.write(extracted_text)
        print(file_name)
    os.chdir("..")


def change_text(txt):
    # Add "\Chapter{" at the start of the text and "}{}{}" before the first new line
    txt = "\\Chapter{" + txt.replace("\n", "}\n\n", 1)

    # Replace "Abstract —" with "\bigskip\n\end{center}\n\n\noindent{"
    txt = txt.replace("Abstract —", "\n\bigskip\n\end{center}\n\n\noindent{", 1)

    # Replace "Index Terms —" with "}\n\n\noindent\textbf{Keywords:}"
    txt = txt.replace("Index Terms —", "}\n\n\noindent\textbf{Keywords:}")

    # Print or use the modified extracted text
    # print(txt)
    

    return txt

write_txt(os.getcwd())
print(f"current working directory", os.getcwd())