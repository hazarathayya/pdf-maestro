import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def make_df(txt_dir, chunk_size):
    os.chdir(txt_dir)
    chunks = []
    for i in os.listdir(txt_dir):
        with open(i, "r", encoding="utf-8") as f:text=f.read()
        clean_txt = preprocess_text(text)
        chunks  = chunks + [clean_txt[i:i+chunk_size] for i in range(0, len(clean_txt), chunk_size)]
        f.close()
    df = pd.DataFrame({'text':chunks})
    return df


txt_dir = os.path.join(os.getcwd(), 'data/qa_txts')
chunk_size = 256
df = make_df(txt_dir=txt_dir, chunk_size=chunk_size)
print(df.head())
os.chdir("..")
df.to_csv("text.csv")
os.chdir("..")
print(os.getcwd())
