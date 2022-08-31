import matplotlib.pyplot as plt
from wordcloud import WordCloud
from notebookjs import execute_js
import numpy as np
from PIL import Image
from os import path
import os
import scipy.stats as stats
import pylab
from normalization_functions import *
from analytics_functions import *

def wordcloud_generator(freq_dict, male_author=True):
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    if male_author:
        mask = np.array(Image.open(path.join(d, "Useful elements and texts/boy.png")))
        wordcloud = WordCloud(mask=mask).generate_from_frequencies(freq_dict)
        plt.figure(figsize=(18,10))
        plt.imshow(wordcloud)
    else:
        mask = np.array(Image.open(path.join(d, "Useful elements and texts/girl.png")))
        wordcloud = WordCloud(mask=mask).generate_from_frequencies(freq_dict)
        plt.figure(figsize=(18,10))
        plt.imshow(wordcloud)

def radial_bar_chart_generator(csv_path):
    d3_lib_url = "https://d3js.org/d3.v3.min.js"
    with open("assets/Visualizations/radial_bar.css", "r") as f:
        radial_bar_css = f.read()
    with open ("assets/Visualizations/radial_bar_lib.js", "r") as f:
        radial_bar_lib = f.read()

    energy = pd.read_csv(csv_path)

    execute_js(library_list=[d3_lib_url, radial_bar_lib], main_function="radial_bar", 
             data_dict=energy.to_dict(orient="records"), css_list=[radial_bar_css])

# Lexical dispersion of words in text
def word_dispersion_plot(list_of_words, text, remove_punctuation=True):
    nltk_text = text_object_creator(text, remove_punctuation)
    return nltk_text.dispersion_plot(list_of_words)

def frequency_distribution(list_of_urls, number_of_words_to_display=50, show_plot=True, remove_punctuation=True):
    res = ""
    for url in list_of_urls:
        text = text_reader(url)
        res += text
    nltk_text = text_object_creator(res, remove_punctuation)
    f_distribution = FreqDist(nltk_text)
    if show_plot:
        f_distribution.plot(number_of_words_to_display, cumulative=False)
    return f_distribution.most_common(number_of_words_to_display)

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

def distribution_graph(series, left_limit, right_limit):
    # Calculating mean and Stdev of AGW
    mean = np.mean(series)
    std = np.std(series)
    # Calculating probability density function (PDF)
    pdf = stats.norm.pdf(series.sort_values(), mean, std)
    # Drawing a graph
    plt.plot(series.sort_values(), pdf)
    plt.xlim([left_limit,right_limit])
    plt.xlabel("Score", size=12)    
    plt.ylabel("Frequency", size=12)                
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()

def qq_plot(data):
    return stats.probplot(data,plot=pylab)