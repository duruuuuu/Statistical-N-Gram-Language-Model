import random
from preprocessing.process import EOS, EOW

def generate_random_unigram(unigram_table, max_length):
    sentence = []
    
    for _ in range(max_length):
        top_unigrams = [ngram for ngram in sorted(unigram_table.items(), key=lambda x: x[1], reverse=True)[:5] 
                        if ngram[0][0] != EOS and ngram[0][0] != EOW]
        if not top_unigrams:
            break
        next_word = random.choice(top_unigrams)[0][0]
        sentence.append(next_word)

    return ' '.join(sentence)


def generate_random_bigram(bigram_table, max_length):
    sentence = []
    start_word = EOS
    sentence.append(start_word)

    for _ in range(max_length):
        next_word_choices = [
            (key[1], bigram_table[key]) 
            for key in bigram_table.keys() 
            if key[0] == start_word
        ]

        if not next_word_choices:
            break
        
        top_5_bigrams = sorted(next_word_choices, key=lambda x: x[1], reverse=True)[:5]
        words, weights = zip(*top_5_bigrams)
        
        next_word = random.choices(words, weights=weights, k=1)[0]
        
        sentence.append(next_word)
        start_word = next_word

    sentence.append(EOS)
    return ' '.join(sentence)

def generate_random_trigram(trigram_tables, max_length):
    sentence = []
    
    start_words = random.choice([key for key in trigram_tables.keys() if key[0] == EOS and key[1] != EOS])
    sentence.extend(start_words)

    for _ in range(max_length - 2):
        prev_words = tuple(sentence[-2:])
        
        next_word_choices = [
            (key[2], trigram_tables[key]) 
            for key in trigram_tables.keys() 
            if key[:2] == prev_words and key[2] != EOS
        ]

        if not next_word_choices:
            break
        
        top_5_trigrams = sorted(next_word_choices, key=lambda x: x[1], reverse=True)[:5]
        words, weights = zip(*top_5_trigrams)
        
        next_word = random.choices(words, weights=weights, k=1)[0]
        
        sentence.append(next_word)

    sentence.append(EOS)
    return ' '.join(sentence)