"""
This file contains the code for the embeddings. 
    Tested models as follows:
        - Ransaka/SinhalaRoberta
        - keshan/SinhalaBERTo
This file used Ransaka/SinhalaRoberta model for the embeddings.

You can download the model from huggingface.co
    - https://huggingface.co/Ransaka/SinhalaRoberta
    - https://huggingface.co/keshan/SinhalaBERTo

You can download dataset from kaggle.com
    - https://www.kaggle.com/datasets/ransakaravihara/hiru-news-set3

"""
import random
import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer, models,util

model_id = "Ransaka/SinhalaRoberta"

def load_and_process_data(file_path:str)->list:
    """
    This function loads the data from the file path and process it.
    """
    def processor(text:str)->str:
        """Only addresses the most common issues in the dataset"""
        return text\
            .replace("\u200d","")\
            .replace("Read More..","")\
            .replace("ඡායාරූප","")\
            .replace("\xa0","")\
            .replace("වීඩියෝ","")\
            .replace("()","")

    def basic_processing(series:pd.Series)->pd.Series:
        """Applies the processor function to a pandas series"""
        return series\
        .apply(processor)
    
    df  = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['Headline'] = basic_processing(df['Headline'])
    # df['fullText'] = basic_processing(df['fullText'])

    #only headlines used for the embeddings
    sentences = df['Headline'].values.tolist()
    random.shuffle(sentences)
    return sentences

def load_model(model_id:str)->SentenceTransformer:
    """
    This function loads the model from the huggingface.co
    """
    word_embedding_model = models.Transformer(model_id, max_seq_length=514)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def get_embeddings(model: SentenceTransformer, sentences: list)->list:
    """
    This function returns the embeddings for the given sentences.
    """
    return model.encode(sentences)

def save_embeddings(embeddings: list, file_path: str):
    """
    This function saves the embeddings to the given file path.
    """
    np.save(file_path, embeddings)

def load_embeddings(file_path: str)->list:
    """
    This function loads the embeddings from the given file path.
    """
    return np.load(file_path)

def get_similar(model:SentenceTransformer,embeddings: list, query: str, top_k: int = 5)->list:
    """
    This function returns the top k similar sentences for the given query.
    """
    query_embedding = model.encode([query])[0]
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return top_results

if __name__ == "__main__":  
    file_path = r"data\top_cluster_dataset.csv"

    #load and process data
    sentences = load_and_process_data(file_path)
    model = load_model(model_id)

    #get embeddings
    embeddings = get_embeddings(model, sentences)
    save_embeddings(embeddings, r"data\embeddings.npy")