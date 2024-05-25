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


# %% call the func to check the max word count from a list of text strings:
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


# %% embedding func to be used for encoding text data into vectors:
embedding_fn = SentenceTransformerEmbeddingFunction()


# %%
# # create a client object to connect to and interact with a Chroma database:
# chroma_db = chromadb.Client()

# create a client object to interact with a Chroma database named 'db':
chroma_db = chromadb.PersistentClient("db")  # create a local chroma db called 'db'

chroma_db.list_collections()  # list the tables in `chroma_db`
# [], no tables as we just started


# %% create (if the collection does not exist) or retrieve (if the collection already exists) the collection (table) called "movies":
chroma_collection = chroma_db.get_or_create_collection(
    "movies",
    embedding_function=embedding_fn,  # for encoding text data into vectors
)

chroma_db.list_collections()  # [Collection(name=movies)]


# %% convert the vals in the specified cols to str & store them in python lists:
ids = df_movies_filt["id"].astype(str).tolist()  # int64 -> str
# print(ids)  # [ '615656', '758323', ... ]
documents = df_movies_filt["overview"].tolist()  # pandas series -> python list
titles = df_movies_filt["title"].tolist()
metadatas = [{"title": title} for title in titles]
# print(
#     metadatas[:5]
# )  # [ {'title': 'Meg 2: The Trench'}, {'title': "The Pope's Exorcist"}, ... ]


# %% add data to the movies collection in the ChromaDB database in batches:
batch_size = 5000
for i in range(0, len(ids), batch_size):
    print(i)  # i = 0, 5000, 10000, ...
    chroma_collection.add(
        ids=ids[i : i + batch_size],
        documents=documents[i : i + batch_size],
        metadatas=metadatas[i : i + batch_size],
    )


# %% retrieve the IDs of all documents stored within the chroma_collection collection in the ChromaDB database
print(chroma_collection.get()["ids"])  # ['100', '10000', '10001', ...]
print(len(chroma_collection.get()["ids"]))  # 36976


# %% get title by description:
def get_title_by_description(query_text: str):
    n_results = 3
    res = chroma_collection.query(query_texts=[query_text], n_results=n_results)
    # print("👀", res)
    """
    {'ids': [['320288', '127585', '121133']], 'distances': [[0.8340994119644165, 0.8993083238601685, 0.902590811252594]], 'metadatas': [[{'title': 'Dark Phoenix'}, {'title': 'X-Men: Days of Future Past'}, {'title': 'X-Men: The Legend of Wolverine'}]], 'embeddings': None, 'documents': [["The X-Men face their most formidable and powerful foe when one of their own Jean Grey starts to spiral out of control. During a rescue mission in outer space Jean is nearly killed when she's hit by a mysterious cosmic force. Once she returns home this force not only makes her infinitely more powerful but far more unstable. The X-Men must now band together to save her soul and battle aliens that want to use Grey's new abilities to rule the galaxy.", 'The ultimate X-Men ensemble fights a war for the survival of the species across two time periods as they join forces with their younger selves in an epic battle that must change the past – to save our future.', "The most popular Super Hero Team in history is ready for action in a spectacular series of thrilling adventures. When a familiar face from Wolverine's former life resurfaces he must wage a war he never intended. Ultimately the X-Men must join forces with Magneto in a fight to save all mutants from annihilation.  Discover the truth of Wolverine's secret past and watch his decisive battle as he is forced to make a choice that will forever affect the fate of the X-Men."]], 'uris': None, 'data': None}
    """
    for i in range(n_results):
        # NOTE: metadatas = [{"title": title} for title in titles]
        pprint(f"Title: {res['metadatas'][0][i]['title']}")
        # NOTE: documents = df_movies_filt["overview"].tolist()
        pprint(f"Description: {res['documents'][0][i]}")
        pprint("---------------------------------")


# %% call the func:
get_title_by_description(query_text="X, superheros")

# %%
