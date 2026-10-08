import math

def _unigram_probability(syllable, smoothed_unigram_tables, unigrams, unigram_zero_counts):
    numerator = smoothed_unigram_tables.get(syllable, unigram_zero_counts)
    denominator = len(unigrams)
    return float(numerator) / float(denominator)

def _bigram_probability(prev, curr, smoothed_bigram_tables, smoothed_unigram_tables, unigram_zero_counts):
    numerator = smoothed_bigram_tables.get((prev, curr), unigram_zero_counts)
    denominator = smoothed_unigram_tables.get((prev,), unigram_zero_counts)
    return float(numerator) / float(denominator)

def _trigram_probability(prev1, prev2, curr, smoothed_trigram_tables, smoothed_bigram_tables, bigram_zero_counts):
    numerator = smoothed_trigram_tables.get((prev1, prev2, curr), bigram_zero_counts)
    denominator = smoothed_bigram_tables.get((prev1, prev2), bigram_zero_counts)
    return float(numerator) / float(denominator)

def unigram_perplexity(tokens, smoothed_unigram_tables, unigrams, unigram_zero_counts):
    number_of_tokens = len(tokens)
    prob_sum_logs = 0
    for syllable in tokens:
        syllable_probability = _unigram_probability(syllable, smoothed_unigram_tables, unigrams, unigram_zero_counts)
        prob_sum_logs += -math.log(syllable_probability)
    return math.exp(prob_sum_logs / number_of_tokens)

def bigram_perplexity(tokens, smoothed_bigram_tables, smoothed_unigram_tables, unigram_zero_counts):
    number_of_tokens = len(tokens)
    prob_sum_logs = 0
    prev = None
    for curr in tokens:
        if prev is not None:
            syllable_probability = _bigram_probability(prev, curr, smoothed_bigram_tables, smoothed_unigram_tables, unigram_zero_counts)
            prob_sum_logs += -math.log(syllable_probability)
        prev = curr
    return math.exp(prob_sum_logs / number_of_tokens)

def trigram_perplexity(tokens, smoothed_trigram_tables, smoothed_bigram_tables, bigram_zero_counts):
    number_of_tokens = len(tokens)
    prob_sum_logs = 0
    prev1, prev2 = None, None
    for curr in tokens:
        if prev1 is not None and prev2 is not None:
            syllable_probability = _trigram_probability(prev1, prev2, curr, smoothed_trigram_tables, smoothed_bigram_tables, bigram_zero_counts)
            prob_sum_logs += -math.log(syllable_probability)
        prev1, prev2 = prev2, curr
    return math.exp(prob_sum_logs / number_of_tokens)
