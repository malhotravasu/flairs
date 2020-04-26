import numpy as np

import pickle

import praw
from praw.models import MoreComments

from gensim.models import Doc2Vec
from gensim.utils import simple_preprocess

from tensorflow import keras

from utils import url_validator, preprocess_docs, docs_into_vectors
from creds import *

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Initialize required models
classification_model = keras.models.load_model('models/classification_model.h5')
embedding_model = Doc2Vec.load('models/embedding_model.pkl')
with open('models/le.pkl', 'rb') as handle:
    le = pickle.load(handle)

def get_descriptor_from_urlid(urlid):
    if url_validator(urlid):
        post = reddit.submission(url=urlid)
    else:
        post = reddit.submission(id=urlid)
    comments = ''
    for comment in post.comments[:min(3, len(post.comments))]:
        if isinstance(comment, MoreComments):
            continue
        comments += ' ' + comment.body
    return post.title + ' ' + post.selftext + ' ' + comments

def get_flair_from_urlids(urlids):
    docs = []
    for urlid in urlids:
        docs.append(get_descriptor_from_urlid(urlid))
    docs = preprocess_docs(docs)
    doc_vectors = docs_into_vectors(docs, embedding_model)
    prediction = classification_model.predict(doc_vectors)
    flairs = np.argmax(prediction, axis=1)
    confidence = np.max(prediction, axis=1)
    return (le.inverse_transform(flairs), confidence)