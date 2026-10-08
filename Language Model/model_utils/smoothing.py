def smoothing(table):
    frequencies_of_counts = _count_counts(table)

    smoothed_table = {}
    
    for gram, ngram_count in table.items():
        N_c = frequencies_of_counts.get(ngram_count, 1)
        N_c_plus_1 = frequencies_of_counts.get(ngram_count + 1, 1)
        smoothed_count = (((ngram_count + 1) * N_c_plus_1) / N_c)
        smoothed_table[gram] = smoothed_count
    
    N_1 = frequencies_of_counts.get(1, 1)
    P_zero_frequency = N_1 / sum(table.values())
    return smoothed_table, P_zero_frequency

def _count_counts(table):
    frequency_of_counts = {}
    for i in table:
        if table[i] in frequency_of_counts:
            frequency_of_counts[table[i]] += 1
        else:
            frequency_of_counts[table[i]] = 1
    return frequency_of_counts