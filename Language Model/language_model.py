import nltk
from collections import Counter
from model_utils.smoothing import smoothing
from model_utils.perplexity import unigram_perplexity, bigram_perplexity, trigram_perplexity
from model_utils.generate_sentences import generate_random_unigram, generate_random_bigram, generate_random_trigram
from preprocessing.process import EOS, EOW,postprocess

class LanguageModel():
    def __init__(self, tokens, output_folder):
        self.tokens = tokens
        self.OUTPUT_FOLDER = output_folder
        
        self.unigrams, self.smoothed_unigram_tables, self.unigram_zero_counts = self._init_ngram(1)
        self.bigrams, self.smoothed_bigram_tables, self.bigram_zero_counts = self._init_ngram(2)
        self.trigrams, self.smoothed_trigram_tables, self.trigram_zero_counts = self._init_ngram(3)

    def _init_ngram(self, n):
        print(f"Calculating {n}-grams...\n")
        ngrams = list(nltk.ngrams(self.tokens, n))
        ngram_table = Counter(ngrams)
        
        print(f"Smoothing {n}-grams...\n")
        smoothed_ngram_table, zero_counts = smoothing(ngram_table)
        
        return ngrams, smoothed_ngram_table, zero_counts
    
    def calculate_perplexity(self, test_cases):
        with open(self.OUTPUT_FOLDER+"/perplexity", "w", encoding='utf-8') as f:
            i = 1
            for sentence in test_cases:
                test_tokens = sentence.split()
                if len(test_tokens) > 2:
                    f.write("Test Sentence {}: {}\n".format(i, postprocess(sentence)))
                    f.write("Unigram perplexity: {}\n".format(unigram_perplexity(test_tokens, self.smoothed_unigram_tables, self.unigrams, self.unigram_zero_counts)))
                    f.write("Bigram perplexity: {}\n".format(bigram_perplexity(test_tokens, self.smoothed_bigram_tables, self.smoothed_unigram_tables, self.unigram_zero_counts)))
                    f.write("Trigram perplexity: {}\n\n".format(trigram_perplexity(test_tokens, self.smoothed_trigram_tables, self.smoothed_bigram_tables, self.bigram_zero_counts)))  
                    i += 1
        print("Perplexity calculation complete. See results in the appropriate model's output folder.")

    def generate_random_sentences(self, no_of_sentences=10):
        with open(self.OUTPUT_FOLDER+"/random_sentences", "w", encoding='utf-8') as file:
            for sentence_no in range(1, no_of_sentences+1):
                file.write(f"#{sentence_no}:\n")
                file.write(f"Unigram: {generate_random_unigram(self.smoothed_unigram_tables, no_of_sentences)}\n")
                file.write(f"Bigram: {generate_random_bigram(self.smoothed_bigram_tables, no_of_sentences)}\n")
                file.write(f"Trigram: {generate_random_trigram(self.smoothed_trigram_tables, no_of_sentences)}\n\n")
    
        print("Random sentences have been generated. See results in the appropriate model's output folder.")