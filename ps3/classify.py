import math
import os
import random
import numpy as np
from collections import Counter

import spacy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nlp = spacy.load("en_core_web_sm")


def load_documents(base_dir):
    docs = []
    labels = []

    for label in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, label)

        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    #sport 199 problem
                    text = file.read()
                    docs.append(text)
                    labels.append(label)

    return docs, labels


def clear_document(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def data_split(docs, labels, test_size=0.5):
    indexes = list(range(len(docs)))

    random.shuffle(indexes)
    split_index = int(len(docs) * (1 - test_size))

    train_indexes = indexes[:split_index]
    test_indexes = indexes[split_index:]

    train_docs = [docs[i] for i in train_indexes]
    train_labels = [labels[i] for i in train_indexes]

    test_docs = [docs[i] for i in test_indexes]
    test_labels = [labels[i] for i in test_indexes]

    return train_docs, test_docs, train_labels, test_labels


def compute_tf(document: str):
    tokens = document.split()
    term_count = Counter(tokens)
    total_terms = len(tokens)
    tf = {term: count / total_terms for term, count in term_count.items()}
    return tf


def compute_idf(documents: list[str]):
    num_documents = len(documents)
    idf_values = {}

    all_tokens_set = set([term for doc in documents for term in doc.split()])

    for term in all_tokens_set:
        docs_with_term = sum([1 for doc in documents if term in doc.split()])
        idf_values[term] = math.log(num_documents / docs_with_term)
    return idf_values


def compute_tfidf(documents: list[str], unique_words: list[str]):
    idf_values = compute_idf(documents)
    docs_count = len(documents)
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    tfidf_matrix = np.zeros((docs_count, len(unique_words)))
    for doc_idx, document in enumerate(documents):
        tf_values = compute_tf(document)
        for term, val in tf_values.items():
            if term not in unique_words:
                continue
            tfidf_matrix[doc_idx, word_to_index[term]] = val * idf_values[term]

    return tfidf_matrix


print('Loading data...')
# 1. Wczytanie danych - wcześniejsza lematyzacja, usuwanie stop-words
documents, labels = load_documents("bbc")
docs_cleaned = []
for doc in documents:
    docs_cleaned.append(clear_document(doc))

print('Data load and cleaning done.')

# 2. Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = data_split(docs_cleaned, labels, test_size=0.5)

train_words = set()
for doc in X_train:
    train_words.update(doc.split())
train_words = list(train_words)

print('Calculating tfidf...')
# 3. Utwórz wektorowe reprezentacje dokumentów stosując algorytm TF-IDF
tfidf_train_result = compute_tfidf(X_train, train_words)
print('Tfidf for train data - computing done.')
tfidf_test_result = compute_tfidf(X_test, train_words)
print('Tfidf for test data - computing done.')

print('Training model...')
# 4. Trening klasyfikatora
model = MultinomialNB()
model.fit(tfidf_train_result, y_train)

print('Model trained.')

print('Predicting...')
# 5. Predykcja na zbiorze testowym
y_pred = model.predict(tfidf_test_result)
print('Prediction done. \n')

# 6. Ewaluacja modelu
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
