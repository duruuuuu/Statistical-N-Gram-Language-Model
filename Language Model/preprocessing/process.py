import re
import nltk
from preprocessing.syllabilize import syllabilize
from preprocessing.characterize import characterize
from html2text import html2text
nltk.download('punkt')

CORPORA_FILE = "data/wiki_00"
TRAINING_SET = "data/train_data"
TESTING_SET = "data/test_data"

SYLLABILIZED_TRAIN_SET = "data/train_processed_data_SYL"
SYLLABILIZED_TEST_SET = "data/test_processed_data_SYL"
CHARACTERIZED_TRAIN_SET = "data/train_processed_data_CHAR"
CHARACTERIZED_TEST_SET = "data/test_processed_data_CHAR"

WORD_SPLIT_PATTERN = re.compile(r'\w+|[^\w\s]')
EOS_PUNCTUATIONS = ['.', '!', '?']
EOS = "</s>"
EOW = "</w>"
EOW_V2 = " </w> "
TOTAL_CORPUS_LINE_COUNT = 4547965

def preprocess(percentage=100):
    train_number_of_lines, test_number_of_lines = _split_train_test(percentage)
    _process_data(TRAINING_SET, SYLLABILIZED_TRAIN_SET, train_number_of_lines, "Preprocessing syllable train data", "syllable")
    _process_data(TESTING_SET, SYLLABILIZED_TEST_SET, test_number_of_lines, "Preprocessing syllable test data","syllable")

    _process_data(TRAINING_SET, CHARACTERIZED_TRAIN_SET, train_number_of_lines, "Preprocessing character train data", "character")
    _process_data(TESTING_SET, CHARACTERIZED_TEST_SET, test_number_of_lines, "Preprocessing character test data", "character")

def get_syllable_train_data():
    with open(SYLLABILIZED_TRAIN_SET, "r", encoding="utf-8") as file:
        train_text = file.read()
    return train_text

def get_syllable_test_data():
    with open(SYLLABILIZED_TEST_SET, "r", encoding="utf-8") as file:
        test_text = file.read()
    return test_text

def get_character_train_data():
    with open(CHARACTERIZED_TRAIN_SET, "r", encoding="utf-8") as file:
        train_text = file.read()
    return train_text

def get_character_test_data():
    with open(CHARACTERIZED_TEST_SET, "r", encoding="utf-8") as file:
        test_text = file.read()
    return test_text

def _process_data(file_path, output_path, number_of_lines, description, which_model):
    with open(file_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as output_file:
        for line_number, line in enumerate(f, 1):
            if line.isspace() or line_number > number_of_lines:
                continue
            
            line = line.strip()

            if which_model == "syllable":
                text = _syllabilize_text(html2text(line).lower().rstrip())

            if which_model == "character":
                text = _characterize_text(html2text(line).lower().rstrip())

            output_file.write(text)

def _split_train_test(corpora_usage_percentage, test_percentage=5):
    print("Splitting train and test data...")
    total_lines = int(TOTAL_CORPUS_LINE_COUNT * (corpora_usage_percentage / 100))
    test_lines_count = int(total_lines * test_percentage / 100)
    train_lines_count = total_lines - test_lines_count

    with open(CORPORA_FILE, "r", encoding="utf-8") as file:
        with open(TRAINING_SET, "w", encoding="utf-8") as train_file:
            with open(TESTING_SET, "w", encoding="utf-8") as test_file:
                for line_number, line in enumerate(file, 1):
                    if line_number <= train_lines_count:
                        train_file.write(line)
                    else:
                        test_file.write(line)
                        if line_number == total_lines:
                            break

    return train_lines_count, test_lines_count
        
def _syllabilize_text(text):
    words = WORD_SPLIT_PATTERN.findall(text)
    syllabified_text = ''

    for word in words:
        if word.isalpha():
            syllables = syllabilize(word)
            syllabified_text += syllables + EOW_V2
        elif word in EOS_PUNCTUATIONS:
            last_occurrence = syllabified_text.rfind(EOW_V2)
            if last_occurrence != -1:
                syllabified_text = syllabified_text[:last_occurrence] + ' ' + EOS + ' '

    return syllabified_text

def _characterize_text(text):
    words = WORD_SPLIT_PATTERN.findall(text)
    characterized_text = ''

    for word in words:
        if word.isalpha():
            characters = characterize(word)
            characterized_text += characters + EOW_V2
        elif word in EOS_PUNCTUATIONS:
            last_occurrence = characterized_text.rfind(EOW_V2)
            if last_occurrence != -1:
                characterized_text = characterized_text[:last_occurrence] + ' ' + EOS + ' '

    return characterized_text

def postprocess(tokens):
    processed_text = ''.join(tokens)
    processed_text = processed_text.replace(' ', '')
    processed_text = processed_text.replace('</s>', '')
    processed_text = processed_text.replace('</w>', ' ')
    return processed_text