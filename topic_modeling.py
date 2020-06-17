from gensim import corpora
from helpers import prepare_text_for_lda

import gensim
import pickle

def get_topics(source_file, no_of_topics):
    text_data = []
    with open(source_file) as file:
        for line in file:
            tokens = prepare_text_for_lda(line)
            text_data.append(tokens)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data ] ## it might do the trick here
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=no_of_topics, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')

    topics = ldamodel.print_topics(num_words=4)

    topics_parsed = []
    for topic in topics:
        topic_parsed = str(topic).split('"')[1].split('"')[0]
        if topic_parsed not in topics_parsed:
            topics_parsed.append(topic_parsed)

    print(topics_parsed)


try:
    get_topics('dataset.csv',10)
except:
    print('error getting the topics of the twitt')

