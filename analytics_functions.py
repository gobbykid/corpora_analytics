from normalization_functions import *

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

# Lexical dispersion of words in text
def word_dispersion_plot(list_of_words, text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.dispersion_plot(list_of_words)

def frequency_distribution(text, number_of_words_to_display=50, show_plot=True, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    f_distribution = FreqDist(nltk_text)
    if show_plot:
        f_distribution.plot(number_of_words_to_display, cumulative=False)
    return f_distribution.most_common(number_of_words_to_display)

def hapaxes(text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    f_distribution = FreqDist(nltk_text)
    return f_distribution.hapaxes()

def collocations(text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.collocations()

def conditional_frequency_distribution(list_of_words, corpus, cumulative_counts=False):
    c_f_distribution = nltk.ConditionalFreqDist(
        (target, fileid[:4]) # The "[:-4]" is useful to take the year of publication of each text
        for fileid in corpus.fileids()
        if fileid != '.DS_store'
        for word in corpus.words(fileid)
        for target in list_of_words 
        if word.lower().startswith(target)) # The "startswith()" method is useful to take all the words (for example if in targets we have "girl", the function will take also "girls")
    c_f_distribution.plot(cumulative=cumulative_counts)
    #c_f_distribution.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)
    return c_f_distribution

def common_words_list(word_dict, percentage_dict, output_set, direct_percentage=True):
    if direct_percentage:
        for word in sorted(percentage_dict,key=percentage_dict.get,reverse=True)[:50]:
            try:
                ratio = percentage_dict[word]/(1-percentage_dict[word])
            except:
                #That is the situation in which a word only appears in male sentences
                ratio = 10
            if ratio >= 3:
                output_set.add(word)
                print('%.1f\t%02d\t%02d\t%s' % (ratio, word_dict['male'].get(word,0), word_dict['female'].get(word,0), word))
    else:
        for word in sorted (percentage_dict,key=percentage_dict.get,reverse=False)[:50]:
            try:
                ratio=(1-percentage_dict[word])/percentage_dict[word]
            except:
                ratio = 10
            if ratio >= 3:
                output_set.add(word)
                print('%.1f\t%01d\t%01d\t%s' % (ratio, word_dict['male'].get(word,0), word_dict['female'].get(word,0), word))


#NOT USED
"""
# Function to extract the proper nouns 
def ProperNounExtractor(text):
    res = list()
    tagged = pos_tagging(text)
    for (word, tag) in tagged:
        if tag == 'NNP': # If the word is a proper noun
            res.append(word.lower().capitalize())
    return res

# Classifiers functions
def gender_feature_last_char(name):
    feature = {"last_character" : name[-1],
                "first_character" : name[0]}
    return feature
"""