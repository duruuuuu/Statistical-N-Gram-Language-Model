def characterize(word):
    word = word.lower()

    if len(word) < 2:
        return word
    
    return " ".join(list(word))