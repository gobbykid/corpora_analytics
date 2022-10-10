import matplotlib.pyplot as plt
from wordcloud import WordCloud
from notebookjs import execute_js
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import scipy.stats as stats
import pylab
import seaborn as sns
import altair as alt
from functions.normalization_functions import *
from functions.analytics_functions import *

def wordcloud_generator(freq_dict, male_author=True):
    if male_author:
        mask = np.array(Image.open("assets/Useful elements and texts/boy.png"))
        wordcloud = WordCloud(mask=mask).generate_from_frequencies(freq_dict)
        plt.figure(figsize=(18,10))
        plt.imshow(wordcloud)
    else:
        mask = np.array(Image.open("assets/Useful elements and texts/girl.png"))
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

def analogies_plot(orig_data, labels):
    pca = PCA(n_components=2)
    data = pca.fit_transform(orig_data)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.xlabel("PCA Component 2")
    plt.ylabel("PCA Component 1")
    plt.plot(data[:,0], data[:,1], '.')
    
    for i in range(len(data)//2):
        plt.annotate("",
                    xy=data[i],
                    xytext=data[i+len(data)//2],
                    arrowprops=dict(arrowstyle="<-",
                                    connectionstyle="arc3, rad=0",
                                    relpos=(1., 1.), fc='w')
            )
    for i in range(len(data)):
        if i < len(data)//2:
            plt.annotate(labels[i], xy=data[i], color='#2081C3',size='small')
        else:
            plt.annotate(labels[i], xy=data[i], color='#FF729F',size='small')

def plot_embedding_space(list_of_words, model, female_words=True): 
    plt.figure(figsize=(13,7))
    pca = PCA(n_components=2)
    data = list()   
    for w in list_of_words:
        if w in model.wv:
            data.append(model.wv[w])
    pca_data = pca.fit_transform(data)
    if female_words:
        plt.scatter(pca_data[:,0],pca_data[:,1],linewidths=10,color='#FEC9F1')
        plt.title("Word Embedding Space - Female related words",size=20)
    else:
        plt.scatter(pca_data[:,0],pca_data[:,1],linewidths=10,color='#A3C4BC')
        plt.title("Word Embedding Space - Male related words",size=20)
    plt.xlabel("PC1",size=12)
    plt.ylabel("PC2",size=12)
    for i, pca in enumerate(pca_data):
        plt.annotate(list_of_words[i], xy=(pca_data[i,0],pca_data[i,1]))


def plot_final_embedding_space(f_words, m_words, model, female_corpus=True): 
    plt.figure(figsize=(13,7))
    pca = PCA(n_components=2)
    list_of_words = list()
    f_len = len(f_words)
    list_of_words.extend(f_words)
    m_len = len(m_words)
    list_of_words.extend(m_words)
    data = list()   
    for i,w in enumerate(list_of_words):
        if w in model.wv:
            data.append(model.wv[w])
        else:
            if i < f_len:
                f_len -= 1
            else:
                m_len -= 1    
    pca_data = pca.fit_transform(data)
    plt.scatter(pca_data[:f_len,0],pca_data[:f_len,1],linewidths=10,color='#FEC9F1')
    plt.scatter(pca_data[m_len:,0],pca_data[m_len:,1],linewidths=10,color='#A3C4BC')
    if female_corpus:
        plt.title("Word Embedding Space - Female authors corpus", size=20)
    else:
        plt.title("Word Embedding Space - Male authors corpus", size=20)
    plt.xlabel("PC1",size=12)
    plt.ylabel("PC2",size=12)
    for i, pca in enumerate(pca_data):
        plt.annotate(list_of_words[i], xy=(pca_data[i,0],pca_data[i,1]))


def tfidf_heatmap(df_top_scores, red_list, female_authors=True):
    # adding a little randomness to break ties in term ranking
    top_tfidf_plusRand = df_top_scores.copy()
    top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(df_top_scores.shape[0])*0.0001

    # base for all visualizations, with rank calculation
    base = alt.Chart(top_tfidf_plusRand).encode(
        x = 'rank:O',
        y = 'document:N'
    ).transform_window(
        rank = "rank()",
        sort = [alt.SortField("tfidf", order="descending")],
        groupby = ["document"],
    )
    # heatmap specification
    if female_authors:
        heatmap = base.mark_rect().encode(
            color=alt.Color('tfidf:Q',scale=alt.Scale(range=['#ffdde1','#89216B']))
        )
    else:
        heatmap = base.mark_rect().encode(
            color=alt.Color('tfidf:Q',scale=alt.Scale(range=['#FFFDE4','#005AA7']))
        )
    # red circle over terms in above list
    circle = base.mark_circle(size=100).encode(
        color = alt.condition(
            alt.FieldOneOfPredicate(field='term', oneOf=red_list),
            alt.value('red'),
            alt.value('#FFFFFF00')        
        )
    )
    # text labels, white for darker heatmap colors
    text = base.mark_text(baseline='middle').encode(
        text = 'term:N',
        color = alt.condition(alt.datum.tfidf >= 0.30, alt.value('white'), alt.value('black'))
    )
    # display the three superimposed visualizations
    (heatmap + circle + text).properties(width = 600)