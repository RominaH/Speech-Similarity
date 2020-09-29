import pandas as pd
import numpy as np
import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Location of Michelle Obama's Speech
MO = r"C:\Users\romha\Desktop\Michelle Obama 2008 DNC Speech.txt"
# Location of Melania Trump's Speech
MT = r"C:\Users\romha\Desktop\Melania Trump 2016 RNC Speech.txt"

def similarity(path1, path2):
    '''
    prints the cosine and similiarity of the text objects located
    at path1 and path2
    '''
    # opens the file and saves them as readable objects
    one = open(path1).read().replace('\n','')
    two = open(path2).read().replace('\n','')
    
    #creates a list containing the objects 
    lst = [one, two]
    
    #drops the stop words --------------------------------------------------------
    vectorizer = CountVectorizer(stop_words='english')
        
    #converts the text documents to a (sparse) matrix of token counts
    sparse_matrix = vectorizer.fit_transform(lst)
    
    #creates dense matrix
    dense_matrix = sparse_matrix.todense()
    df = pd.DataFrame(dense_matrix, 
                      columns=vectorizer.get_feature_names(), 
                      index=['1','2'])
    
    #returns the cosine similarity
    cos_similarity = cosine_similarity(df, df)[0][1]
    return cos_similarity
    
    
#prints the cosine similarity of Michelle Obama's speech and Melania Trump's speech
similarity(MO, MT)

