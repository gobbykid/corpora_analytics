from normalization_functions import *
import pandas as pd

def token_count(text, remove_punctuation=True):
    tokens = word_tokenization(text, remove_punctuation, True)
    return len(tokens)

def lexical_richness(list_of_urls, remove_punctuation=True):
    res = dict()
    for url in list_of_urls:
        text = text_reader(url)
        tokens = word_tokenization(text, remove_punctuation, True)
        res[url] = (len(set(tokens))/len(tokens))
    return res

def corpus_size(corpus):
    number_of_files = 0
    for url in corpus.fileids():
        if url != '.DS_Store':
            number_of_files += 1
    return number_of_files
            
# To define in terms of tokens the length of our corpus
def corpus_dimension(directory_path):
    list_of_files_urls = list()
    total_tokens_in_corpus = 0
    for url in os.listdir(directory_path):
        if url != '.DS_Store':
            if os.path.isfile(os.path.join(directory_path, url)):
                list_of_files_urls.append(url)
    for file_url in list_of_files_urls:
        path = directory_path + file_url
        text = text_reader(path)
        single_text_tokens = token_count(text)
        total_tokens_in_corpus += single_text_tokens
    return total_tokens_in_corpus

# This function is useful to see in which percentage a specific token appears in a text
def word_percentage(token, text, remove_punctuation=True):
    list_of_tokens = word_tokenization(text, remove_punctuation, True)
    single_token_count = list_of_tokens.count(token)
    total_tokens_count = len(list_of_tokens)
    # "Round" is useful to limit to 2 digits the float
    return str(round(100*single_token_count/total_tokens_count, 2))+"%"

# Concordances are best computed for raw or clean texts?
def word_concordances(word, text, remove_punctuation=False):
    nltk_text = text_object_creator(text, remove_punctuation, False)
    return nltk_text.concordance(word)

def concordances_list(word, text, remove_punctuation=True):
    res = list()
    nltk_text = text_object_creator(text, remove_punctuation)
    concordances_list = nltk_text.concordance_list(word)
    for obj in concordances_list:
        res.append((obj[0], obj[1], obj[2]))
        # Returns a list of lists with [0]sx, [1]word, [2]dx
    return res

def word_similarities(word, text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.similar(word)

def word_common_contexts(list_of_words, text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.common_contexts(list_of_words)

def hapaxes(text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    f_distribution = FreqDist(nltk_text)
    return f_distribution.hapaxes()

def collocations(text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.collocations()

def common_words_df(word_dict, percentage_dict, output_set, direct_percentage=True):
    store_dict = {
        "word":[],
        "ratio":[],
        "f_corpus raw count":[],
        "m_corpus raw count":[],
        "f_corpus percentage":[],
        "m_corpus percentage":[]
    }
    # With direct_percentage==True, we mean we are working on male words, sorted from the higher 
    # to the lower (in terms of frequencies) thanks to "reverse=True"
    if direct_percentage:
        for word in sorted(percentage_dict,key=percentage_dict.get,reverse=True)[:50]:
            try:
                ratio = percentage_dict[word]/(1-percentage_dict[word])
            except:
                #That is the situation in which a word only appears in male sentences
                ratio = 100
            if ratio >= 3:
                output_set.add(word)
                store_dict["word"].append(word)
                store_dict["ratio"].append(round(ratio,2))
                store_dict["f_corpus raw count"].append(word_dict['female'].get(word,0))
                store_dict["m_corpus raw count"].append(word_dict['male'].get(word,0))
                store_dict["f_corpus percentage"].append(round((1-percentage_dict.get(word,0)),4))
                store_dict["m_corpus percentage"].append(round(percentage_dict.get(word,0),4))
    # There, we work with female words, in a from low to high sorted dictionary
    else:
        for word in sorted(percentage_dict,key=percentage_dict.get,reverse=False)[:50]:
            try:
                ratio=(1-percentage_dict[word])/percentage_dict[word]
            except:
                ratio = 100
            if ratio >= 3:
                output_set.add(word)
                store_dict["word"].append(word)
                store_dict["ratio"].append(round(ratio,2))
                store_dict["f_corpus raw count"].append(word_dict['female'].get(word,0))
                store_dict["m_corpus raw count"].append(word_dict['male'].get(word,0))
                store_dict["f_corpus percentage"].append(round((1-percentage_dict.get(word,0)),4))
                store_dict["m_corpus percentage"].append(round(percentage_dict.get(word,0),4))
    df = pd.DataFrame(store_dict)
    return df

"""
# Classifiers functions
def gender_feature_last_char(name):
    feature = {"last_character" : name[-1],
                "first_character" : name[0]}
    return feature
"""