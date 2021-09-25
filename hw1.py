import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    #pass
    textLen = len(text)
    for i in range(n-1):
        text.insert(0, '<s>')
    else:
        text.append('</s>')
    for ind in range(textLen+1):
        yield tuple([
            text[(ind+n-1)], tuple(
                text[ind: (ind+n-1)]
                )
            ])




# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    #pass
    f = open(corpus_path, 'r')
    paras = f.read().split('\n\n')
    sent_tokens = []
    for p in paras:
        sent_tokens.extend(sent_tokenize(p))
    word_tokens = []
    for s in sent_tokens:
        word_tokens.append(word_tokenize(s))
    return word_tokens


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    #pass
    corpus = load_corpus(corpus_path)
    nGramLM = NGramLM(n)
    for sentence in corpus:
        nGramLM.update(sentence)
    return nGramLM


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()
        
    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        #pass
        for ngram in get_ngrams(self.n, text):
            if ngram in self.ngram_counts:
                self.ngram_counts[ngram]+=1
            else:
                self.ngram_counts[ngram]=1
            context = ngram[1]
            if context in self.context_counts:
                self.context_counts[context]+=1
            else:
                self.context_counts[context]=1
            self.vocabulary.add(ngram[0])

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        #pass
        pmle = 0
        #if len(self.vocabulary)!=0:
        #    pmle = 1.0/len(self.vocabulary)
        ngram = (word, context)
        if (context in self.context_counts):
            if (ngram in self.ngram_counts):
                if(delta==0):
                    pmle = self.ngram_counts[ngram]/self.context_counts[context]
                else:
                    pmle = (self.ngram_counts[ngram] + delta)/(self.context_counts[context] + (delta * len(self.vocabulary)))
            else:
                pmle = delta / (self.context_counts[context] + (delta * len(self.vocabulary)))
        else:
            if len(self.vocabulary)!=0:
                pmle = 1.0/len(self.vocabulary)
        return pmle


    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        #pass
        logProb = 0
        for ngram in get_ngrams(self.n, sent):
            prob = self.get_ngram_prob(ngram[0], ngram[1], delta)
            #logProb+=math.log2(prob)
            try:
                logProb+=math.log2(prob)
            except ValueError:
                logProb-=math.inf
            except:
                print('Other Error')
        return logProb

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]], delta=.0) -> float:
        #pass
        corpus_log_prob = 0
        token_counts = [len(wordList) for wordList in corpus]
        total_tokens = sum(token_counts)
        all_words=[]
        for sent in corpus:
            for word in sent:
                all_words.append(word)
        corpus_log_prob = self.get_sent_log_prob(word_tokenize(' '.join(all_words)),delta)
        avg_log_prob = corpus_log_prob/total_tokens
        pp = math.pow(2, -1 * avg_log_prob)
#        if math.isinf(avg_log_prob):
#            pp = math.inf
#        else:
#            pp = math.pow(2, (-1*avg_log_prob))
        return pp

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        #pass
        vocab = sorted(self.vocabulary)
        r = random.random()
        lb = ub = 0
        for word in vocab:
            prob = self.get_ngram_prob(word, context, delta)
            lb=ub
            ub+=prob
            if (lb<=r) and (r<ub):
                return word


    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        #pass
        textGen=[]
        context=[]
        eos = False
        for i in range(self.n - 1):
            context.append('<s>')
        for j in range(max_length):
            rand_word = self.generate_random_word(tuple(context), delta)
            textGen.append(rand_word)
            if (rand_word=='</s>'):
                break
            if (self.n!=1):
                del context[0]
                context.append(rand_word)
        finText = ' '.join(textGen)
        return finText




def main(corpus_path: str, delta: float, seed: int):
    alt_path = 'shakespeare.txt'
    unigram_lm = create_ngram_lm(1, alt_path)
    trigram_lm = create_ngram_lm(3, alt_path)
    pentagram_lm = create_ngram_lm(5, alt_path)
    print("\nRandom Sentences generated by unigram model")
    for i in range(1,6):
        print(i, ": ", unigram_lm.generate_random_text(10, 0.1))
    print("\nRandom Sentences generated by trigram model")
    for i in range(1,6):
        print(i, ": ", trigram_lm.generate_random_text(10, 0.1))
    print("\nRandom Sentences generated by 5-gram model")
    for i in range(1,6):
        print(i, ": ", pentagram_lm.generate_random_text(10, 0.1))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=0.01, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
