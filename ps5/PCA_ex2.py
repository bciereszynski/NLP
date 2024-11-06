import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# Zadanie 2
# Użyj algorytmu PCA z pakietu scikit-learn celem redukcji z 50 do 2 wymiarów dla każdego słowa ze słownika modelu

word2Vec = Word2Vec.load("model-w2v-cbow.bin")
model = word2Vec.wv

words = list(model.index_to_key)
vectors = [model[word] for word in words]

frequent_words = words[:100]

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

#Frequent
frequent_vectors = []

for i in range(100):
    frequent_vectors.append(list(reduced_vectors[words.index(frequent_words[i])]))

x_values = []
y_values = []
for vector in frequent_vectors:
    x_values.append(vector[0])
    y_values.append(vector[1])

plt.figure(figsize=(10, 8))
plt.scatter(x_values, y_values)

for i in range(100):
    plt.text(x_values[i], y_values[i], frequent_words[i], fontsize=9)

plt.title("100 najczęściej występujących słów")
plt.show()


#Random
random_indices = random.sample(range(len(reduced_vectors)), 100)
random_vectors = reduced_vectors[random_indices]
random_words = [words[i] for i in random_indices]

plt.figure(figsize=(10, 8))
plt.scatter(random_vectors[:, 0], random_vectors[:, 1])

for i, word in enumerate(random_words):
    plt.text(random_vectors[i, 0], random_vectors[i, 1], word, fontsize=9)

plt.title("100 losowych słów")
plt.show()
