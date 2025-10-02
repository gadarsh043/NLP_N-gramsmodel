import nltk
import itertools
import re
import math

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def preprocess(corpus):
    corpus = [[word.lower() for word in sentence] for sentence in corpus]
    corpus = [[re.sub(r'[^\w\s]', '', word) for word in sentence] for sentence in corpus]
    lemmatizer = WordNetLemmatizer()
    corpus = [[lemmatizer.lemmatize(word) for word in sentence] for sentence in corpus]

    return corpus

def handle_unknown_words(corpus, min_freq=2):
    flat_corpus = list(itertools.chain(*corpus))
    word_counts = Counter(flat_corpus)
    
    unknown_token = "<UNK>"
    processed_corpus = []
    
    for sentence in corpus:
        processed_sentence = []
        for word in sentence:
            if word_counts[word] < min_freq:
                processed_sentence.append(unknown_token)
            else:
                processed_sentence.append(word)
        processed_corpus.append(processed_sentence)
    
    return processed_corpus, unknown_token

# Load Dataset
def load_corpus(file_path):
    with open(file_path, 'r') as f:
        corpus = f.readlines()
    return [sentence.strip().split() for sentence in corpus]

class UnigramModel:
    def __init__(self, corpus, k=0):
        self.corpus = corpus
        self.word_counts = defaultdict(int)
        self.total_words = 0
        self.vocab = set()
        self.unknown_token = "<UNK>"
        self.add_k = k
        self.vocab_size = 0
        self.build_unigram_model()

    def build_unigram_model(self):
        for word in self.corpus:
            if word not in self.vocab:
                self.vocab.add(word)
            self.word_counts[word] += 1
            self.total_words += 1
        self.vocab_size = len(self.vocab)

    def get_count(self, word):
        return self.word_counts.get(word, 0)

    def get_probability(self, word):
        count = self.get_count(word)
        prob = (count + self.add_k) / (self.total_words + self.add_k * self.vocab_size)
        return max(prob, 1e-10)

    def unigram_perplexity(self, test_corpus):
        log_prob_sum = 0
        test_words = len(test_corpus)

        for word in test_corpus:
            prob = self.get_probability(word)
            log_prob_sum += math.log(prob)

        perplexity = math.exp(-log_prob_sum / test_words)
        return perplexity

class BigramModel:
    def __init__(self, corpus, k):
        self.corpus = corpus
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self.add_k = k
        self.build_bigram_model()

    def build_bigram_model(self):
        for i in range(len(self.corpus) - 1):
            word1 = self.corpus[i]
            word2 = self.corpus[i + 1]
            self.vocab.add(word1)
            self.vocab.add(word2)
            self.bigram_counts[(word1, word2)] += 1
            self.unigram_counts[word1] += 1

        self.unigram_counts[self.corpus[-1]] += 1
        self.vocab.add(self.corpus[-1])

    def get_probability(self, word1, word2):
        bigram_count = self.bigram_counts.get((word1, word2), 0)
        firstword_count = self.unigram_counts.get(word1, 0)
        vocab_size = len(self.vocab)
        
        if firstword_count == 0:
            return 1e-10
        
        prob = (bigram_count + self.add_k) / (firstword_count + self.add_k * vocab_size)
        return max(prob, 1e-10)

    def bigram_perplexity(self, test_corpus):
        log_prob_sum = 0
        test_bigrams = len(test_corpus) - 1

        for i in range(test_bigrams):
            word1 = test_corpus[i]
            word2 = test_corpus[i + 1]
            prob = self.get_probability(word1, word2)
            log_prob_sum += math.log(prob)

        perplexity = math.exp(-log_prob_sum / test_bigrams)
        return perplexity


# Function to evaluate the Bigram model


if __name__ == "__main__":
    file_path = 'train.txt'
    test_path = 'validation.txt'

    train_data = load_corpus(file_path)
    test_data = load_corpus(test_path)

    train_corpus = preprocess(train_data)
    test_corpus = preprocess(test_data)

    train_corpus_flat = list(itertools.chain(*train_corpus))
    test_corpus_flat = list(itertools.chain(*test_corpus))

    train_data_split, validation_data = train_test_split(train_data, train_size=0.9, random_state=192)
    val_corpus = preprocess(validation_data)
    val_corpus_flat = list(itertools.chain(*val_corpus))

    print("=== N-gram Language Model Evaluation ===")
    print("Testing different smoothing values and unknown word thresholds\n")

    for min_freq in [1, 2, 3]:
        print(f"--- Unknown Word Threshold: {min_freq} ---")
        
        train_processed, unknown_token = handle_unknown_words(train_corpus, min_freq)
        val_processed, _ = handle_unknown_words(val_corpus, min_freq)
        test_processed, _ = handle_unknown_words(test_corpus, min_freq)
        
        train_flat = list(itertools.chain(*train_processed))
        val_flat = list(itertools.chain(*val_processed))
        test_flat = list(itertools.chain(*test_processed))
        
        for k in [0, 0.5, 1.0]:
            print(f"\nSmoothing k = {k}:")
            
            unigram_model = UnigramModel(train_flat, k)
            bigram_model = BigramModel(train_flat, k)

            unigram_perplexity_val = unigram_model.unigram_perplexity(val_flat)
            bigram_perplexity_val = bigram_model.bigram_perplexity(val_flat)
            
            unigram_perplexity_test = unigram_model.unigram_perplexity(test_flat)
            bigram_perplexity_test = bigram_model.bigram_perplexity(test_flat)

            print(f"  Validation Set - Unigram: {unigram_perplexity_val:.4f}, Bigram: {bigram_perplexity_val:.4f}")
            print(f"  Test Set - Unigram: {unigram_perplexity_test:.4f}, Bigram: {bigram_perplexity_test:.4f}")
        
        print()
