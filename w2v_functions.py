from normalization_functions import *
from analytics_functions import *
import gensim

def list_builder(list_of_urls):
    all_tokens = list()
    for url in list_of_urls:
        text = text_reader(url)
        list_of_sentences = syntok_list_of_sentences(text)
        for sentence in list_of_sentences:
            sentence_tokens = expand_contractions(sentence, False, True, True)
            for token in sentence_tokens:
                if token in stopwords or token in ['"',"'",'.',',','/','-']:
                    sentence_tokens.remove(token)
            all_tokens.append(sentence_tokens)
    return all_tokens
                
            