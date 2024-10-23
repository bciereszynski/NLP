import os
import random

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


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

print('Loading data...')
# 1. Wczytanie danych - wcześniejsza lematyzacja, usuwanie stop-words
documents, labels = load_documents("bbc")
docs_cleaned = [clear_document(doc) for doc in documents]

print('Data load and cleaning done.')

# 2. Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = data_split(docs_cleaned, labels, test_size=0.5)

print('Calculating tfidf...')
# 3. Utwórz wektorowe reprezentacje dokumentów stosując algorytm TF-IDF
vectorizer = TfidfVectorizer()
tfidf_train_result = vectorizer.fit_transform(X_train)
tfidf_test_result = vectorizer.transform(X_test)

print('Training model...')
# 4. Trening klasyfikatora
model = MultinomialNB()
model.fit(tfidf_train_result, y_train)

print('Model trained.')

print('Predicting...')
# 5. Predykcja na zbiorze testowym
y_pred = model.predict(tfidf_test_result)
print('Prediction done. \n')

print("Test data:")
# 6. Ewaluacja modelu testowego
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Ewaluacja modelu treningowego
print("Train data:")
y_pred = model.predict(tfidf_train_result)
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred))
