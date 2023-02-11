# Import necessary modules
import subprocess
import numpy as np
import string
import spacy

from nltk.stem import SnowballStemmer

from unicodedata import normalize, combining
from tqdm import tqdm
from sdg_classifier.utils.monitor import LOGGER


def load_nltk_tools():
    # download necessary NLP models and tools
    try:
        _ = spacy.load("en_core_web_lg")
    except OSError:
        print("The 'en_core_web_lg' model is not installed. Installing now...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
        _ = spacy.load("en_core_web_lg")
        print("The 'en_core_web_lg' model has been successfully installed.")


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

    # Remove special characters and numbers
    Z = [text.translate(str.maketrans("", "", string.punctuation + "123456789")) for text in Z]

    # Remove double spaces
    Z = [' '.join(text.split()) for text in Z]

    # Remove accents
    Z = ["".join([char for char in normalize("NFKD", text) if not combining(char)]) for text in Z]

    # Tokenize text, remove stopwords and punctuation
    LOGGER.info("[Preprocessing - 1/2] Tokenizing texts, removing stopwords and punctuation.")
    nlp = spacy.load("en_core_web_lg")
    Z = [[token.text for token in nlp(sentence) if (not token.is_stop) and (not token.is_punct)]
         for sentence in tqdm(Z)]

    # Remove titles with less than 3 words at the end of the code
    has_more_than_two_words = [len(text_list) > 2 for text_list in Z]

    LOGGER.info("[Preprocessing - 2/2] Standardizing texts.")
    # Lemmatizing
    if truncation == "lemma":
        # Concatenate tokens
        Z = [" ".join(tokens) for tokens in Z]

        # Lemmatize sentences
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
    y = y[has_more_than_two_words]
    Z = Z[has_more_than_two_words]

    return Z, y
