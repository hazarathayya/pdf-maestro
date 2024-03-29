import os
import pandas as pd

df = pd.read_csv(os.path.join(os.getcwd(), "data/text.csv"))
print(df.head())