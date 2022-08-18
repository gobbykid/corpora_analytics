import matplotlib.pyplot as plt
from wordcloud import WordCloud
from notebookjs import execute_js
import numpy as np
from PIL import Image
from os import path
import os
import pandas as pd

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
    with open("Visualizations/radial_bar.css", "r") as f:
        radial_bar_css = f.read()
    with open ("Visualizations/radial_bar_lib.js", "r") as f:
        radial_bar_lib = f.read()

    energy = pd.read_csv(csv_path)

    execute_js(library_list=[d3_lib_url, radial_bar_lib], main_function="radial_bar", 
             data_dict=energy.to_dict(orient="records"), css_list=[radial_bar_css])

