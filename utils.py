import numpy as np
import pandas as pd

import pickle

from praw.models import MoreComments

from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec

from urllib.parse import urlparse

EMBEDDING_SIZE = 150

def generate_csv(posts, PATH='data/data.csv'):
    # posts: Dictionary object
    columns = ['url', 'title', 'body', 'comments', 'flair']
    data = pd.DataFrame.from_dict(posts, orient='index', columns=columns)
    data.index.name = 'id'
    data.fillna('', inplace=True)
    data['descriptor'] = data['title'] + ' ' + data['body'] + ' ' + data['comments']
    data = data[['url', 'title', 'body', 'comments', 'descriptor', 'flair']]
    
    kept_flairs = set([x for (x, v) in dict(data['flair'].value_counts()).items() if v>50])
    data['flair'] = data['flair'].apply(lambda x: x if x in kept_flairs else 'Other')
    
    data.to_csv(PATH)

def load_csv(PATH='data/data.csv'):
    data = pd.read_csv(PATH, index_col='id')
    data.fillna('', inplace=True)
    return data

def expand_data(sections):
    with open('data/data.pkl', 'rb') as handle:
        posts = pickle.load(handle)
        
    print('Previously:', len(posts))
    for section in sections:
        for post in section:
            if post.id not in posts:
                comments = ''
                for comment in post.comments[:min(3, len(post.comments))]:
                    if isinstance(comment, MoreComments):
                        continue
                    comments += ' ' + comment.body
                posts[post.id] = [
                    post.url,
                    post.title,
                    post.selftext,
                    comments,
                    post.link_flair_text
                ]
                
    with open('data/data.pkl', 'wb') as handle:
        pickle.dump(posts, handle)
    print('Now:', len(posts))

    generate_csv(posts)

def preprocess_docs(docs):
    corpus = []
    for doc in docs:
        processed_sentence = simple_preprocess(doc, max_len=20)
        corpus.append(processed_sentence)
    return corpus

def doc_into_vector(doc, model, process_as_document=False):
    if process_as_document and isinstance(model, Doc2Vec):
        return model.infer_vector(doc)
    vec = np.zeros(EMBEDDING_SIZE)
    word_count = len(doc)
    for word in doc:
        if word in model.wv.vocab:
            vec += model.wv.get_vector(word)
    return np.divide(vec, word_count+1e-8)

def docs_into_vectors(docs, model):
    embeddings = []
    for doc in docs:
        doc_vector = doc_into_vector(doc, model)
        embeddings.append(doc_vector)
    return np.array(embeddings)

a = 'http://www.cwi.nl:80/%7Eguido/Python.html'
b = '/data/Python.html'
c = 532
d = u'dkakasdkjdjakdjadjfalskdjfalk'

def url_validator(given_url):
    try:
        result = urlparse(given_url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
