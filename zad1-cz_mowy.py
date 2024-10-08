import spacy
import sys
import os

nlp = spacy.load("en_core_web_lg")

def analyzeFile(text, outFilePath):
    my_doc = nlp(text)
    tokens_count = len(my_doc)
    sents_count = len(list(my_doc.sents))
    words_count = len([token for token in my_doc if not token.is_punct and not token.is_space])

    avg_words_per_sent = round(words_count / sents_count, 2)
    avg_tokens_per_sent = round(tokens_count / sents_count, 2)

    verb_count = 0
    noun_count = 0
    adj_count = 0
    adv_count = 0

    for token in my_doc:
        if token.pos_ == 'VERB':
            verb_count += 1
        elif token.pos_ == 'NOUN':
            noun_count += 1
        elif token.pos_ == 'ADJ':
            adj_count += 1
        elif token.pos_ == 'ADV':
            adv_count += 1

    noun_lemma_count = {}

    for token in my_doc:
        if token.pos_ == "NOUN":
            lemma = token.lemma_
            if lemma in noun_lemma_count:
                noun_lemma_count[lemma] += 1
            else:
                noun_lemma_count[lemma] = 1

    noun_lemma_count = sorted(noun_lemma_count.items(), key=lambda x: x[1], reverse=True)
    freq_nouns = noun_lemma_count[:5]

    adjs_count_for_noun = {}
    for noun, count in freq_nouns[:2]:
        adjs = []
        for chunk in my_doc.noun_chunks:
            if chunk.root.text == noun:
                adjs.extend([token for token in chunk if token.pos_ == "ADJ"])
        adj_lemma_count = {}
        for token in adjs:
            lemma = token.lemma_
            if lemma in noun_lemma_count:
                adj_lemma_count[lemma] += 1
            else:
                adj_lemma_count[lemma] = 1

        adjs_count_for_noun[noun] = sorted(adj_lemma_count.items(), key=lambda x: x[1], reverse=True)

    out = open(outFilePath, "w", encoding="utf-8")
    out.write(f"Liczba zdań: {sents_count}\n")
    out.write(f"Liczba tokenów: {tokens_count}\n")
    out.write(f"Liczba słów: {words_count}\n")

    out.write(f"Średnia liczba słów w zdaniu: {avg_words_per_sent}\n")
    out.write(f"Średnia liczba tokenów w zdaniu: {avg_tokens_per_sent}\n")
    out.write(f"Liczba czasowników: {verb_count}\n")
    out.write(f"Liczba rzeczowników: {noun_count}\n")
    out.write(f"Liczba przymiotników: {adj_count}\n")
    out.write(f"Liczba przysłówków: {adv_count}\n")

    out.write("5 najczęściej występujących rzeczowników:\n")
    for noun, count in freq_nouns:
        out.write(f"{noun}: {count}\n")

    out.write("Dla 2 najczęściej występujących rzeczowników - wszystkie określające je przymiotniki występujące w tekście:\n")
    for noun, count in freq_nouns[:2]:
        out.write(f"{noun}:\n")
        for adj, count in adjs_count_for_noun[noun]:
            out.write(f"{adj} --- {count} \n")
    out.close()


directory = sys.argv[1]
for file in os.listdir(directory):
    filePath = os.path.join(directory, file)
    if os.path.isfile(os.path.join(directory, file)):
        os.makedirs(directory + "_out", exist_ok=True)
        outFilePath = os.path.join(directory + "_out", file)
        with open(filePath, "r", encoding="utf-8") as f:
            text = f.read()
            analyzeFile(text, outFilePath)
            f.close()
