import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from normalization_functions import *
from sklearn.datasets import fetch_20newsgroups

text = text_reader("Raw/F/1869_little_women.txt")
#dataset = fetch_20newsgroups(shuffle=True, random_state=5, remove=('headers', 'footers', 'quotes'))
df = DataFrame.eval(text)
