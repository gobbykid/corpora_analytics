from lib2to3.pgen2.pgen import DFAState
from normalization_functions import *
import pandas as pd
from random import sample
from visualizations_functions import *

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
    nltk_text = text_object_creator(text, remove_punctuation, True)
    return nltk_text.concordance(word)

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

def check_distribution (sample, sample_check=False):
    # Turn all the scores into positive (or == 0) by adding 1 to them
    # Useful to adopt the Boxcox transformation, that will regularize the 
    # distribution of our values into a normal distribution.
    positive_scores = list()
    for el in sorted(sample):
    # Remove extremes
        if el != 1 and el != -1:
            positive_scores.append(el+1)
    positive_scores, param = stats.boxcox(positive_scores)
    # Computing p-value
    normality_result = stats.normaltest(positive_scores)
    p_value = normality_result[1]
    # Computing a Q-Q-Plot
    if not sample_check:
        qq_plot(positive_scores)
        if p_value > 0.05:
            print("According to a D'Agostino-Pearson normality test, we cannot reject the null hypothesis, in fact:")
            print("The p-value is:", p_value)
            print("The mean is:", np.mean(Series(positive_scores)))
            print("The std is:", np.std(Series(positive_scores)))
            print("So, data follow a normal distribution.")
            return (np.mean(Series(positive_scores)), np.std(Series(positive_scores)))
        else:
            print("According to a D'Agostino-Pearson normality test, data do not follow a normal distribution.")
            print("The p-value is:", p_value)
            return False
    else:
        return p_value

def f_test(data_1, data_2, sample_check=False):
    positive_scores_1 = list()
    positive_scores_2 = list()
    for el in sorted(data_1):
    # Remove extremes
        if el != 1 and el != -1:
            positive_scores_1.append(el+1)
    positive_scores_1, param = stats.boxcox(positive_scores_1)
    for el in sorted(data_2):
    # Remove extremes
        if el != 1 and el != -1:
            positive_scores_2.append(el+1)
    positive_scores_2, param = stats.boxcox(positive_scores_2)
    num = np.array(positive_scores_1)
    den = np.array(positive_scores_2)
    var_num = np.var(num)
    var_den = np.var(den)
    if var_num < var_den:
        var_num, var_den = var_den, var_num
        num, den = den, num
    f_stats = np.var(num)/np.var(den) #calculate F test statistic 
    dfn = num.size-1 #define degrees of freedom numerator 
    dfd = den.size-1 #define degrees of freedom denominator 
    p_value = 1-stats.f.cdf(f_stats, dfn, dfd) #find p-value of F test statistic 
    if p_value > 0.05:
        if not sample_check:
            print("According to an F-test, we cannot reject the null hypothesis, in fact:")
            print("The F-statistics is:", f_stats)
            print("The p-value is:", p_value)
            print("The two variances do not differ in a significant way")
    else:
        if not sample_check:
            print("According to an F-test, we can reject the null hypothesis, in fact:")
            print("The F-statistics is:", f_stats)
            print("The p-value is:", p_value)
            print("The two variances differ in a significant way")
    return (f_stats, p_value)

def stratified_random_sampling(populations, sample_size):
    pop_samples = []
    for population in populations:
        normal = False
        population_df = DataFrame(population)
        while not normal:
            fraction = sample_size/len(population_df.index)
            sample_df = population_df.groupby('labels', group_keys=False).apply(lambda x: x.sample(frac=fraction))
            pop_sample = sample_df["scores"].to_list()
            p_value = check_distribution(pop_sample, True)
            if p_value > 0.05:
                pop_samples.append(pop_sample)
                normal = True
    f_test_results = f_test(pop_samples[0], pop_samples[1], True)
    if f_test_results[1] > 0.05:
        return (pop_samples[0], pop_samples[1])
    else:
        sample_res = stratified_random_sampling(populations, sample_size)
        return sample_res
        

def t_test_independent(sample_1, sample_2):
    variances_stats = round(f_test(sample_1, sample_2, True)[0],4)
    same_variances = False
    if variances_stats == 1.0000:
        same_variances = True
    res = stats.ttest_ind(sample_1, sample_2, equal_var=same_variances)
    p_value = res[1]
    if p_value > 0.05:
        print("According to a T-Test for independent samples we can not reject the null hypothesis, in fact:")
        print("The p-value is:", p_value)
        print("The two means (" + str(np.mean(Series(sample_1))), str(np.mean(Series(sample_2))) + ") do not differ in a significant way.")
    else:
        print("According to a T-Test for independent samples we can accept the alternative hypothesis, in fact:")
        print("The p-value is:", p_value)
        print("The two means (" + str(np.mean(Series(sample_1))), str(np.mean(Series(sample_2))) + ") differ in a significant way.")
    return res



"""
# Classifiers functions
def gender_feature_last_char(name):
    feature = {"last_character" : name[-1],
                "first_character" : name[0]}
    return feature
"""