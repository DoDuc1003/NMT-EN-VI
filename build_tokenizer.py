import numpy as np                      # Numerical computing library
import pandas as pd                     # Data manipulation library
import matplotlib.pyplot as plt        # Data visualization library
import seaborn as sns                   # Statistical data visualization library

from sklearn import preprocessing      # Data preprocessing
from sklearn.model_selection import train_test_split   # Data splitting
from sklearn.metrics import accuracy_score, precision_score, recall_score   # Model evaluation metrics

from sklearn.linear_model import LinearRegression, LogisticRegression   # Linear models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor   # Decision tree models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor   # Ensemble models
from sklearn.svm import SVC, SVR         # Support Vector Machines
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor   # K-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB   # Naive Bayes


import nltk                             # Natural Language Processing toolkit
from nltk.corpus import stopwords       # Stopwords for text processing

import spacy

from datasets import load_dataset

dataset = load_dataset("mt_eng_vietnamese", "iwslt2015-vi-en")
print(len(dataset))

def get_training_english(dataset):
    for i in range(len(dataset['train'])):
        yield dataset['train']['translation'][i]['en']

def get_training_vietnamese(dataset):
    for i in range(len(dataset['train'])):
        yield dataset['train']['translation'][i]['vi']
        
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

def build_tokenizer(training_corpus):
    """
    normalization
    pre-tokenization
    model
    postprocessor
    """
    my_tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))
    my_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    # WhitespaceSplit() Whitespace()
    my_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
    my_tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    return my_tokenizer

tokenizer_english = build_tokenizer(training_corpus=get_training_english(dataset))
tokenizer_vietnamese = build_tokenizer(training_corpus=get_training_vietnamese(dataset))
tokenizer_english.save('tokenizer_english.json')
tokenizer_vietnamese.save('tokenizer_vietnamese.json')

