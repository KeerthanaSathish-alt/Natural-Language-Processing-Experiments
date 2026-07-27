import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# User input
text = input("Enter a paragraph: ")

# Sentence Tokenization
sentences = sent_tokenize(text)

# Word Tokenization for full text
words = word_tokenize(text)

# Create stemmer and lemmatizer objects
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Stemming and Lemmatization for full word tokens
stemmed_words = [stemmer.stem(word) for word in words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Display results
print("\nOriginal Text:")
print(text)

print("\nSentence Tokens:")
print(sentences)

print("\nWord Tokens:")
print(words)

print("\nStemmed Word Tokens:")
print(stemmed_words)

print("\nLemmatized Word Tokens:")
print(lemmatized_words)

print("\nSentence-wise Word Tokenization, Stemming, and Lemmatization:")
for i, sentence in enumerate(sentences, start=1):
    sentence_words = word_tokenize(sentence)

    stemmed_sentence_words = [
        stemmer.stem(word)
        for word in sentence_words
    ]

    lemmatized_sentence_words = [
        lemmatizer.lemmatize(word)
        for word in sentence_words
    ]

    print(f"\nSentence {i}:")
    print("Original Sentence:", sentence)
    print("Word Tokens:", sentence_words)
    print("Stemmed Words:", stemmed_sentence_words)
    print("Lemmatized Words:", lemmatized_sentence_words)

# Simple comparison
print("\nComparison:")
print("Sentence tokenization splits a paragraph into separate sentences.")
print("Word tokenization splits text into individual words and punctuation.")
print("Stemming reduces words to root forms, which may not always be meaningful.")
print("Lemmatization converts words to meaningful base dictionary forms.")