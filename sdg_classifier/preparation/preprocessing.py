# Import necessary modules
import subprocess
import numpy as np
import spacy
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re

from unicodedata import normalize, combining
from tqdm import tqdm


def load_nltk_tools():
    # download necessary NLP models and tools
    try:
        _ = spacy.load("en_core_web_lg")
    except OSError:
        print("The 'en_core_web_lg' model is not installed. Installing now...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
        _ = spacy.load("en_core_web_lg")
        print("The 'en_core_web_lg' model has been successfully installed.")

    nltk.download('punkt')
    nltk.download('stopwords')


def get_stopwords():
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    spacy_en = spacy.load("en_core_web_lg")
    spacy_stopwords = spacy_en.Defaults.stop_words

    stopwords = list(set(spacy_stopwords).union(set(nltk_stopwords)))

    return stopwords


def preprocess_data(X, y, truncation="lemma"):
    """
    Text preprocessing is divided into two stages:

    1. In the first stage, advanced NLP operations are performed that cannot be performed with native TensorFlow functions, such as:
    - Removing accents, and special characters;
    - Removing stopwords;
    - Lemmatization or Stemming;
    - Filtering.

    2. The second stage consists of converting `numpy` datasets to the TensorFlow standard format `tf.data.Dataset`.

    Before feeding the model with this dataset, we need to vectorize the text sequences. This step is performed only in the model training notebook. It involves passing the data through a native TensorFlow `TextVectorization` layer. This layer performs:
    - Padding of text sequences;
    - Word encoding/vectorization.
    :param X: raw titles
    :param y: target labels
    :param truncation: truncation mode ("lemma", "stem" or None)
    :return: output dataset with processed titles
    """

    # Convert text to lowercase
    Z = [text.lower() for text in X]

    # Remove special characters
    Z = [text.translate(str.maketrans("", "", string.punctuation + "123456789")) for text in Z]

    # Remove numbers
    Z = [re.sub(r"^\d+\s|\s\d+\s|\s\d+$|\d+\)", ' ', text) for text in Z]

    # Remove double spaces
    Z = [re.sub(r"\s+[a-zA-Z]\s+", ' ', text) for text in Z]

    # Remove accents
    Z = ["".join([char for char in normalize("NFKD", text) if not combining(char)]) for text in Z]

    # Tokenize text
    Z = [word_tokenize(text) for text in Z]

    # Remove stopwords
    stopwords = get_stopwords()
    Z = [list((word for word in tokens if ((word not in stopwords) and (len(word) > 1)))) for tokens in Z]

    # Lemmatizing
    if truncation == "lemma":
        # Concatenate tokens
        Z = [" ".join(tokens) for tokens in Z]

        # Lemmatize sentences
        nlp = spacy.load("en_core_web_lg")
        lemmatize = lambda sentence: " ".join([token.lemma_ for token in nlp(sentence)])
        Z = [lemmatize(text) for text in tqdm(Z)]

    # Stemming
    if truncation == "stem":
        stemmer = SnowballStemmer("english")
        Z = [" ".join([stemmer.stem(token) for token in tokens]) for tokens in Z]

    if truncation is None:
        Z = [" ".join(tokens) for tokens in Z]

    # Convert back to np.array
    Z = np.array(Z)

    # Discard empty sentences
    non_empty_sentences = Z != ""
    y = y[non_empty_sentences]
    Z = Z[non_empty_sentences]

    return Z, y
