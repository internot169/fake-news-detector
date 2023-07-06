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
basepath = '.'

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

with open(os.path.join(basepath, 'train_val_data.pkl'), 'rb') as f:
  train_data, val_data = pickle.load(f)

def get_description_from_html(html):
  soup = bs(html)
  description_tag = soup.find('meta', attrs={'name':'og:description'}) or soup.find('meta', attrs={'property':'description'}) or soup.find('meta', attrs={'name':'description'})
  if description_tag:
    description = description_tag.get('content') or ''
  else: # If there is no description, return empty string.
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