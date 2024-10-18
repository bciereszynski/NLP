import spacy
import random
nlp = spacy.load("en_core_web_sm")

# usage: python distance.py
# przykład.txt has to be in the same directory as this script

filePath = "przyklad.txt"
min_length = 3

def levenshtein(s, t):
        ''' From Wikipedia article; Iterative with two matrix rows. '''
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]
                
        return v1[len(t)]

def find_closest_word(word, dictionary):
    closest_word = "placeholder"
    min_val = levenshtein(word, closest_word)
    for candidate in dictionary:
        val = levenshtein(word, candidate)
        if val < min_val:
            min_val = val
            closest_word = candidate
    return closest_word

def get_words(text):
    words = []

    doc = nlp(text)
    for token in doc:
        if token.is_alpha:
            words.append(token.text)
    return words

def create_errors(text, min_length):
    words = get_words(text)
    number_of_modifications = int(len(words) * 0.2)
    words_to_modify = random.sample(words, number_of_modifications)

    modified_text = []
    doc = nlp(text)
    for token in doc:
        if token.text in words_to_modify:
            modified_text.append(modify_word(token.text, min_length))
        else:
            modified_text.append(token.text)
    return modified_text
def modify_word(word, min_length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if random.random() < 0.33 and len(word) >= min_length:
        idx = random.randint(0, len(word) - 1)
        new_char = random.choice(alphabet)
        return word[:idx] + new_char + word[idx + 1:]
    elif random.random() < 0.33 and len(word) >= min_length:
        idx = random.randint(0, len(word))
        new_char = random.choice(alphabet)
        return word[:idx] + new_char + word[idx:]
    elif random.random() < 0.33 and len(word) >= min_length:
        idx = random.randint(0, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    
    return word

with open(filePath, "r", encoding="utf-8") as f:
    text = f.read()
    words = get_words(text)
    f.close()

words_single = list(set(words)) # remove duplicates
with open("slownik.txt", "w", encoding="utf-8") as f:
    for w in words_single:
        f.write(w + "\n")

modified_text = create_errors(text, min_length)
with open("przyklad_z_bledami.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(modified_text))

# ------------------

with open("slownik.txt", 'r', encoding='utf-8') as file:
    dictionary = [line.strip() for line in file]

with open("przyklad_z_bledami.txt", "r", encoding="utf-8") as file:
    text_with_errors = file.read()

words_to_correct = get_words(text_with_errors)
corrected_text = []
doc = nlp(text_with_errors)
for token in doc:
    if token.text in words_to_correct:
        corrected_text.append(find_closest_word(token.text, dictionary))
    else:
        corrected_text.append(token.text)

with open("przyklad_poprawiony.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(corrected_text))

with open("przyklad_poprawiony.txt", "r", encoding="utf-8") as file:
    text = file.read()
    words_corrected = get_words(text)

with open("przyklad.txt", "r", encoding="utf-8") as file:
    text = file.read()
    words_original = get_words(text)

diff = 0
for i in range(len(words_original)):
    if words_original[i].lower() != words_corrected[i].lower():
        diff += 1
        print(f"Original: {words_original[i]}, Corrected: {words_corrected[i]} ---- {words_to_correct[i]}")

print("Diff: " + str(diff))
