# %% import libraries and modules:
import re  # regular expression operations
import pandas as pd  # data manipulation and analysis
import seaborn as sns  # data visualization
from langchain.text_splitter import (
    SentenceTransformersTokenTextSplitter,
)  # split text into tokens
import chromadb  # vector database
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)  # embedding funcs for ChromaDB
from pprint import (
    pprint,
)  # pretty-printing for better readability of complex data structures


# %% determine the max word count from a list of text strings:
def max_word_count(txt_list: list):
    max_length = 0
    for txt in txt_list:  # loop over each text string
        word_count = len(
            re.findall(r"\w+", txt)
        )  # count the number of words in the string `txt`
        if word_count > max_length:
            max_length = word_count
    return f"Max Word Count: {max_length} words"


# %% split text into tokens:

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, max input length: 256 characters
model_max_chunk_length = 256

# init a text splitter object to split text into tokens:
token_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=model_max_chunk_length,
    chunk_overlap=0,  # overlap between adjacent chunks
    model_name="all-MiniLM-L6-v2",  # a SentenceTransformers model name
)


# %% import data:
text_path = "data/movies.csv"  # resouce: https://www.kaggle.com/datasets
# df = pd.read_csv(text_path)
# print(df)


# %% read the CSV file into a pandas DataFrame:
df_movies_raw = pd.read_csv(
    text_path, parse_dates=["release_date"]
)  # specify cols that should be parsed as dates (without specifying `parse_dates`, the `release_date` col will be read as strings)
print(df_movies_raw.shape)  # (38998, 20)
print(df_movies_raw.head())  # first 5 entries


# %% filter enties with missing info and duplicate ids:

# filter movies for missing title or overview:
df_movies_filt = df_movies_raw.dropna(subset=["title", "overview"])
print(df_movies_filt.shape)  # (38371, 20)

# drop duplicate ids:
df_movies_filt = df_movies_filt.drop_duplicates(subset=["id"])
print(df_movies_filt.shape)  # (36976, 20)


# %% call the helper func to check the max word count from a list of text strings:
max_word_count(df_movies_filt["overview"])  # 193 words
# 🥳 The maximum description length is 193 words, which is below the threshold which is coming from our model (256).

# %% calculate the word count for each movie description & store these counts in a list:
description_len = []
for txt in df_movies_filt.loc[
    :, "overview"
]:  # selecting all rows and the "overview" col
    description_len.append(len(re.findall(r"\w+", txt)))

print(len(description_len))  # 36976

"""
Iteration 1
txt = 'This is the first movie.'
re.findall(r"\w+", txt) returns ['This', 'is', 'the', 'first', 'movie']
len(['This', 'is', 'the', 'first', 'movie']) is 5
description_len.append(5)

Iteration 2
txt = 'An exciting new adventure awaits.'
re.findall(r"\w+", txt) returns ['An', 'exciting', 'new', 'adventure', 'awaits']
len(['An', 'exciting', 'new', 'adventure', 'awaits']) is 5
description_len.append(5)
After both iterations, description_len will be [5, 5].
"""


# %% visualize the distribution of word counts in overviews:
sns.histplot(description_len, bins=100)
# 👀 Most overviews fall below 100 words.


# %% embedding func:
embedding_fn = SentenceTransformerEmbeddingFunction()
