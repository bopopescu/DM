from collections import Counter
from helpers import prepare_text_for_lda_2

def get_frequent_words(source_file, no_of_words):
    text_data = []
    with open(source_file) as file:
        for line in file:
            tokens = prepare_text_for_lda_2(line)
            text_data.append(tokens)

    flatten_text_data = [item for sublist in text_data for item in sublist]
    occurence_count = Counter(flatten_text_data)
    print (occurence_count.most_common(no_of_words))


get_frequent_words('dataset.csv',5)
