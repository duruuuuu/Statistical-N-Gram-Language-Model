from preprocessing.process import preprocess, get_syllable_test_data, get_syllable_train_data, get_character_test_data, get_character_train_data
from language_model import LanguageModel

if __name__ == '__main__':
    preprocess(percentage=100)

    # syllable model
    print("SYLLABLE MODEL\n")
    
    print("Processing training and testing data\n")
    train_data, test_data = get_syllable_train_data(), get_syllable_test_data()
    train_tokens = train_data.split()
    sentences = test_data.split(r'</s>')

    syllable_model = LanguageModel(train_tokens, 'syllableModel_output')
    
    print("Calculating perplexity of test sentences...\n")
    syllable_model.calculate_perplexity(sentences)
    
    print("Generating random sentences...\n")
    syllable_model.generate_random_sentences()

    # character model
    print("\nCHARACTER MODEL")
    
    print("Processing training and testing data\n")
    train_data, test_data = get_character_train_data(), get_character_test_data()
    train_tokens = train_data.split()
    sentences = test_data.split(r'</s>')

    character_model = LanguageModel(train_tokens, 'characterModel_output')

    print("Calculating perplexity of test sentences...\n")
    character_model.calculate_perplexity(sentences)
    
    print("Generating random sentences...\n")
    character_model.generate_random_sentences()