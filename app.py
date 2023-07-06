import math
import os
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import GloVe
import pickle
import requests, io, zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# URL to change
curr_url = "www.yahoo.com"
basepath = '.'

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: 
    description = ''
  return description

def scrape_description(url):
  if not url.startswith('http'):
    url = 'http://' + url
  response = requests.get(url, timeout=10)
  html = response.text
  description = get_description_from_html(html)
  return description

def get_descriptions_from_data(data):
  descriptions = []
  for site in tqdm(data):
    url, html, label = site
    descriptions.append(get_description_from_html(html))
  return descriptions

train_descriptions = get_descriptions_from_data(train_data)
train_urls = [url for (url, html, label) in train_data]
val_descriptions = get_descriptions_from_data(val_data)

vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(descriptions, vectorizer):
  X = vectorizer.transform(descriptions).todense()
  return X

bow_X_train = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_y_train = [label for url, html, label in train_data]

bow_X_val = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_y_val = [label for url, html, label in val_data]

bow_X_train, bow_X_val = np.array(bow_X_train), np.array(bow_X_val)

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
        if found_words > 0:
            X[i] /= found_words

    return X

glove_X_train = glove_transform_data_descriptions(train_descriptions)
glove_y_train = [label for (url, html, label) in train_data]

glove_X_val = glove_transform_data_descriptions(val_descriptions)
glove_y_val = [label for (url, html, label) in val_data]

def train_model(X_train, y_train, X_val, y_val):
  model = LogisticRegression(solver='liblinear')
  model.fit(X_train, y_train)

  return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val):
  model = train_model(X_train, y_train, X_val, y_val)
  
  return model

def prepare_data(data, featurizer):
    X = []
    y = []
    for datapoint in data:
        url, html, label = datapoint
        html = html.lower()
        y.append(label)

        features = featurizer(url, html)

        feature_descriptions, feature_values = zip(*features.items())

        X.append(feature_values)

    return X, y, feature_descriptions

def get_normalized_count(html, phrase):
    return math.log(1 + html.count(phrase.lower()))


def keyword_featurizer(url, html):
    features = {}

    features['.com domain'] = url.endswith('.com')
    features['.org domain'] = url.endswith('.org')
    features['.net domain'] = url.endswith('.net')
    features['.info domain'] = url.endswith('.info')
    features['.org domain'] = url.endswith('.org')
    features['.biz domain'] = url.endswith('.biz')
    features['.ru domain'] = url.endswith('.ru')
    features['.co.uk domain'] = url.endswith('.co.uk')
    features['.co domain'] = url.endswith('.co')
    features['.tv domain'] = url.endswith('.tv')
    features['.news domain'] = url.endswith('.news')

    keywords = ['trump', 'biden', 'clinton', 'sports', 'finance']

    for keyword in keywords:
      features[keyword + ' keyword'] = get_normalized_count(html, keyword)

    return features


keyword_X_train, y_train, _ = prepare_data(train_data, keyword_featurizer)
keyword_X_val, y_val, _ = prepare_data(val_data, keyword_featurizer)

train_and_evaluate_model(keyword_X_train, y_train, keyword_X_val, y_val)
vectorizer = CountVectorizer(max_features=300)

vectorizer.fit(train_descriptions)

def vectorize_data_descriptions(data_descriptions, vectorizer):
  X = vectorizer.transform(data_descriptions).todense()
  return X

bow_X_train = vectorize_data_descriptions(train_descriptions, vectorizer)
bow_X_val = vectorize_data_descriptions(val_descriptions, vectorizer)
bow_X_train, bow_X_val = np.array(bow_X_train), np.array(bow_X_val)

train_and_evaluate_model(bow_X_train, y_train, bow_X_val, y_val)

VEC_SIZE = 300
glove = GloVe(name='6B', dim=VEC_SIZE)

def get_word_vector(word):
    try:
      return glove.vectors[glove.stoi[word.lower()]].numpy()
    except KeyError:
      return None

def glove_transform_data_descriptions(descriptions):
    X = np.zeros((len(descriptions), VEC_SIZE))
    for i, description in enumerate(descriptions):
        found_words = 0.0
        description = description.strip()
        for word in description.split():
            vec = get_word_vector(word)
            if vec is not None:
                found_words += 1
                X[i] += vec
            X[i] /= found_words

    return X


glove_X_train = glove_transform_data_descriptions(train_descriptions)
glove_X_val = glove_transform_data_descriptions(val_descriptions)

train_and_evaluate_model(glove_X_train, y_train, glove_X_val, y_val)

def combine_features(X_list):
  return np.concatenate(X_list, axis=1)

combined_X_train = combine_features([keyword_X_train, bow_X_train, glove_X_train])
combined_X_val = combine_features([keyword_X_val, bow_X_val, glove_X_val])

model = train_and_evaluate_model(combined_X_train, y_train, combined_X_val, y_val)

def get_data_pair(url):
  if not url.startswith('http'):
      url = 'http://' + url
  url_pretty = url
  if url_pretty.startswith('http://'):
      url_pretty = url_pretty[7:]
  if url_pretty.startswith('https://'):
      url_pretty = url_pretty[8:]

  response = requests.get(url, timeout=10)
  htmltext = response.text

  return url_pretty, htmltext



url, html = get_data_pair(curr_url)

def dict_to_features(features_dict):
  X = np.array(list(features_dict.values())).astype('float')
  X = X[np.newaxis, :]
  return X
def featurize_data_pair(url, html):
  keyword_X = dict_to_features(keyword_featurizer(url, html))
  description = get_description_from_html(html)
  bow_X = vectorize_data_descriptions([description], vectorizer)
  glove_X = glove_transform_data_descriptions([description])

  X = combine_features([keyword_X, bow_X, glove_X])

  return X

curr_X = np.array(featurize_data_pair(url, html))

model = train_model(combined_X_train, y_train, combined_X_val, y_val)

curr_y = model.predict(curr_X)[0]


if curr_y < .5:
  print(curr_url, 'appears to be real.')
else:
  print(curr_url, 'appears to be fake.')