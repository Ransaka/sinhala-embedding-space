"""
This file is used to search the most similar vectors in the database using the faiss library.
used indexer class grabbed from daily-llama repo (https://github.com/Ransaka/daily-llama)
"""
import numpy as np
import pandas as pd
from embeddings.embeddings import load_model, model_id

# from daily llama repo
import faiss

class Indexer:
  def __init__(self, embed_vec):
    self.embeddings_vec = embed_vec
    self.build_index()

  def build_index(self):
    """
    Build the index for the embeddings.

    This function initializes the index for the embeddings. It calculates the dimension (self.d)
    of the embeddings vector and creates an IndexFlatL2 object (self.index) for the given dimension.
    It then adds the embeddings vector (self.embeddings_vec) to the index.

    Parameters:
    - None

    Return:
    - None
    """
    self.d = self.embeddings_vec.shape[1]
    self.index = faiss.IndexFlatL2(self.d)
    self.index.add(self.embeddings_vec)

  def topk(self, vector, k = 4):
    """
        A function that takes in a vector and an optional parameter k and returns the indices of the k nearest neighbors in the index.

        Parameters:
            vector: A numpy array representing the input vector.
            k (optional): An integer representing the number of nearest neighbors to retrieve. Defaults to 4 if not specified.

        Returns:
            I: A numpy array containing the indices of the k nearest neighbors in the index.
    """
    # vec = self.retreaver.encode(text)['embeddings'].detach().cpu().numpy()
    _, I = self.index.search(vector, k)
    return I
  

def get_embeddings_vec(file_path):

    """
    This function loads the embeddings from the given file path.
    
    Parameters:
    - file_path: A string representing the path to the embeddings file.
    
    Return:
    - embeddings_vec: A numpy array containing the embeddings.
    """
    return np.load(file_path)

def get_similar(indexer, text_embeddings, top_k = 5):
    """
    This function returns the top k similar sentences for the given query.
    
    Parameters:
    - indexer: An Indexer object representing the indexer for the embeddings.
    - text_embeddings: A np.array representing the query embeddings.
    - top_k (optional): An integer representing the number of nearest neighbors to retrieve. Defaults to 4 if not specified.
    
    Return:
    - top_results: A numpy array containing the indices of the k nearest neighbors in the index.
    """
    return indexer.topk(text_embeddings,k=top_k).flatten()

def search_demo(test_queries:list=None,top_k:int=1):
    """
    This function returns the top k similar sentences for the given query.
    """
    model = load_model(model_id)
    embeddings_vec = get_embeddings_vec(r"data\top_cluster_embeddings.npy")
    indexer = Indexer(embeddings_vec)

    cluster_dataset = pd.read_csv(r"data\top_cluster_dataset.csv",usecols=['Headline'])
    search_space = cluster_dataset['Headline'].values.tolist()
    if test_queries is None:
        test_queries = [
            "ක්ෂය රෝග මර්දන ව්යාපාරයේ පී.සී.ආර්. යන්ත්ර 36 භාවිතයට ගන්නැයි ඉල්ලීමක්",
            "පොළොන්නරුව මහරෝහලේ අකුරට වැඩ කිරීමේ වෘත්තීය ක්රියාමාර්ගයක්",
            "අංගොඩ අයි ඩී එච් රෝහලේ ඩෙංගු විශේෂ ප්රතිකාර ඒකකය තවම නැහැ ",
            "කමිටු ගැන විශ්වාසයක් නැහැ - මාළඹේ පෞද්ගලික වෛද්ය විද්යාලයීය දෙමාපිය සංසදය"
        ]

    for query in test_queries:
        query_embeddings = model.encode(query).reshape(1,-1)
        print("Query: ", query)
        print("Results: ")
        for index in get_similar(indexer, query_embeddings, top_k = top_k):
            print("\t-",search_space[index])
        print()