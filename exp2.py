import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import hmm
from nltk.corpus import brown

# Download required resources
nltk.download('punkt')
nltk.download('brown')

# Get multiline input from user
print("Enter text:")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
text = "\n".join(lines)
if not text.strip():
    print("No input provided.")
    sys.exit(0)

# Tokenize text
tokens = word_tokenize(text)

# Train an HMM tagger on a small corpus
train_sents = brown.tagged_sents(categories='news', tagset='universal')
trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(train_sents)

tagged_words = model.tag(tokens)

# Display tokens
print("\nTokens:")
print(tokens)

# Display POS tags
print("\nPOS Tags:")
for word, tag in tagged_words:
    print(word, "->", tag)

# Simple tag meanings
print("\nTag Meanings:")
print("NN -> Noun")
print("VB -> Verb")
print("JJ -> Adjective")
print("RB -> Adverb")
print("PRP -> Pronoun")
print("DT -> Determiner")

# Count tagged words
print("\nTotal Words:", len(tokens))