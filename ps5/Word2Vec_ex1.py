import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords

from gensim.models import Word2Vec

# ZADANIE 1
# Wytrenuj model Word2Vec (CBoW) tak aby uzyskać 50 wymiarowe wektory reprezentujące słowa.

urls = ['https://en.wikipedia.org/wiki/Biology',
        'https://en.wikipedia.org/wiki/Computer_science',
        'https://en.wikipedia.org/wiki/Physics',
        'https://en.wikipedia.org/wiki/History']
all_words = []

# korpus językowy
for url in urls:
    scrapped_data = urllib.request.urlopen(url)
    article = scrapped_data.read()
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = " ".join([p.text for p in paragraphs])


    processed_article = article_text.lower()
    processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
    processed_article = re.sub(r'\s+', ' ', processed_article)

    all_sentences = nltk.sent_tokenize(processed_article)
    all_words = all_words + [nltk.word_tokenize(sent) for sent in all_sentences]

all_words = [[w for w in words if w not in stopwords.words('english')] for words in all_words]

word2vec = Word2Vec(all_words, vector_size=50)
word2vec.save('model-w2v-cbow.bin')

# tests
model = word2vec.wv

print(model.most_similar('intelligence'))
print("energy <--> power: " + str(model.similarity('energy','power')))

word2vec.save('model-w2v-cbow.bin')
