import nltk
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
import nltk.data
from nltk.text import Text
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import os
from os.path import exists
import re
import pandas as pd
from pandas import *
import gensim
from pprint import *
from matplotlib import *
import contractions
import syntok.segmenter as segmenter
from syntok.tokenizer import Tokenizer
from tqdm import tqdm

# In the following, the "gaps=True" parameter is useful in order to specify that we want to find the spaces
#marked thanks to the special regex "\s+" as delimiters and not as tokens
# By setting this parameter ""=False" we basically ask to find the tokens that match the regex
tokenizer = RegexpTokenizer(pattern= "\s+", gaps=True) 
# This "clean_tokenizer" avoid to consider punctuation but takes into consideration ' and - and _
clean_tokenizer = RegexpTokenizer(pattern= "[a-zA-Z']+")
sentence_tokenizer = nltk.data.load('assets/Tokenizers/english.pickle')

# Seen that some of the stopwords are also contained in these 2 sets that we need to
# gender sentences, I will first remove all of them.
male_words=set(['guy','dr','spokesman','chairman',"men's",'men','him',"he's",'his','boy',
'boyfriend','boyfriends','boys','brother','brothers','dad','dads','dude','father',
'fathers','fiance','gentleman','gentlemen','god','grandfather','grandpa',
'grandson','groom','he','himself','his','husband','pastor','husbands','king','male','man',
'mr','nephew','nephews','priest','prince','son','sons','uncle','uncles',
'waiter','widower','widowers','he','he is','lord'])
female_words=set(['heroine','drss','spokeswoman','chairwoman',"women's",'actress','women',
"she's",'her','aunt','aunts','bride','daughter','daughters','female','fiancee','girl',
'girlfriend','girlfriends','girls','goddess','granddaughter','grandma','grandmother',
'herself','ladies','lady','mom','moms','mother','mothers','mrs','ms','niece',
'nieces','priestess','princess','queens','she','sister','sisters','waitress',
'widow','widows','wife','wives','woman','she','she is','lady'])

stopwords = stopwords.words('english')
for word in stopwords:
    if word in male_words or word in female_words:
        stopwords.remove(word)

lemmatizer = WordNetLemmatizer()
# Text reading and normalization functions

# Reads text
def text_reader(url):
    file = open(url).read()
    return file

def word_tokenization(text, remove_punctuation=True, for_analysis=False):
    if for_analysis:
        text = text.lower()
    if not remove_punctuation:
        tokens = tokenizer.tokenize(text)
    elif remove_punctuation:
        tokens = clean_tokenizer.tokenize(text)
    return tokens

# Sentence tokenization is not so accurate with this tokenizer, but it is the best since it mess up only dialogues.
def sentence_tokenization(text):
    return sentence_tokenizer.tokenize(text)

def create_list_of_sentences(text):
    list_of_sentences = sentence_tokenization(text)
    clean_sentences = list()
    for sentence in list_of_sentences:
        words_in_sentence = list()
        list_of_words = sentence.split(" ")
        for word in list_of_words:
            splitted_words = word.split("\n")
            for splitted in splitted_words:
                words_in_sentence.append(splitted)   
        clean_sentence = " ".join(words_in_sentence)
        clean_sentences.append(clean_sentence)
    return clean_sentences

def syntok_list_of_sentences(text):
    res = []
    for paragraph in segmenter.process(text):
        for sent in paragraph:
            temp = []
            token_list = []
            for token in sent:
                space_token_tuple = token.spacing, token.value
                if space_token_tuple[0] != '\n':
                    token_list.append(space_token_tuple[0])
                else:
                    token_list.append(' ')
                token_list.append(space_token_tuple[1])
            if token_list[0] == ' ':
                token_list.remove(token_list[0])
            temp = ''.join(token_list)
            res.append(temp)
    return res

# A function that returns a list of lists of tokens of each sentence of a corpus
def list_builder(list_of_urls):
# Must be removed all proper names!!!
    commons = text_reader("assets/Useful elements and texts/common_ws_list.txt")
    common_ws_list = list()
    for row in commons.split():
        common_ws_list.append(row.lower())
    common_ws_list = set(common_ws_list)
    all_tokens = list()
    for url in tqdm(list_of_urls):
        text = text_reader(url)
        list_of_sentences = syntok_list_of_sentences(text)
        for sentence in list_of_sentences:
            sentence_tokens = expand_contractions(sentence, False, True)
            for token in sentence_tokens:
                if token in stopwords or token in ['"',"'",'.',',','/','-'] or token in common_ws_list:
                    sentence_tokens.remove(token)
            all_tokens.append(sentence_tokens)
    return all_tokens

def tokenize_texts(text):
    tok = Tokenizer()
    return [token.value for token in tok.tokenize(text)]

def lemmatization(tokens):
    list_of_lemmas = list()
    for token in tokens:
        list_of_lemmas.append(lemmatizer.lemmatize(token))
    return list_of_lemmas

# Creates nltk.text object
def text_object_creator(text, remove_punctuation=True, for_analysis=True):
    tokens = word_tokenization(text, remove_punctuation, for_analysis)
    clean_text = nltk.Text(tokens)
    return clean_text

# Expand the contractions inside a text and creates a new expanded text
#Remember to give the same name to the new txt file, it will be modified inside the function
def expand_contractions(input_url_or_text, is_url=False, remove_punctuation=True, for_analysis=False):
    expanded_words = list()
    if is_url:
        splitted_url = input_url_or_text.split("/")
        if splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "F":
            new_url = "assets/Normalized corpora/F/"+splitted_url[3]
        elif splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "M":
            new_url = "assets/Normalized corpora/M/"+splitted_url[3]
        if not exists(new_url):
            raw = text_reader(input_url_or_text)
            # using contractions.fix to expand the shortened words
            for word in raw.split():
                expanded_words.append(contractions.fix(word))
            expanded_text = ' '.join(expanded_words)
            new_file = open(new_url,'w')
            new_file.write(expanded_text)
            new_file.close()
            return new_url
    else:
        tokens = word_tokenization(input_url_or_text, remove_punctuation, for_analysis)
        for token in tokens:
            expanded_words.append(contractions.fix(token))
        return expanded_words
    

# Expand contractions and DO NOT remove punctuation
# Remember to give the same name to the new txt file, it will be modified inside the function
def text_cleaner(url):
    splitted_url = url.split("/")
    if splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "F":
        new_url = "assets/Normalized corpora/F/"+splitted_url[3]
    elif splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "M":
        new_url = "assets/Normalized corpora/M/"+splitted_url[3]
    if not exists(new_url):
        book = text_reader(url)
        book = re.sub('\[[Ii]llustration(.+)?\]|\[[Pp]icture(.+)?\]', '', book)
        expanded_words = expand_contractions(book, False, False, False)
        # Cleans from stopwords
        new_file = open(new_url,'a')
        for word in expanded_words: 
            if not word in stopwords:
                new_file.write(" " + word)
        new_file.close()
        return new_url
    return new_url

# Probably not useful-----------------------------------
def text_normalizer(url):
    splitted_url = url.split("/")
    if splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "F":
        new_url = "assets/Normalized corpora/F/"+splitted_url[3]
    elif splitted_url[0] == "assets" and splitted_url[1] == "Raw corpora" and splitted_url[2] == "M":
        new_url = "assets/Normalized corpora/M/"+splitted_url[3]
    if not exists(new_url):
        book = text_reader(url)
        # Siamo sicuri di sta roba?
        book = re.sub('\w+\n', '. ', book) #we replace the newlines with a dot and a space, so that the syntok segmenter can work properly
        book = re.sub('\n', ' ', book) #and so on...
        book = re.sub('--', ' ', book)
        book = re.sub('-', ' ', book)
        book = re.sub('_', ' ', book)
        book = re.sub('- -', ' ', book)
        book = re.sub('\*', ' ', book)
        book = re.sub('\s+', ' ', book)
        book = re.sub('\[[Ii]llustration(.+)?\]|\[[Pp]icture(.+)?\]', '', book)
        new_file = open(new_url,'w')
        new_file.write(book)
        new_file.close()
        return new_url
# -------------------------------------------------------

def create_corpus(corpus_directory_path):
    newcorpus = PlaintextCorpusReader(corpus_directory_path, '.*')
    return newcorpus

def pos_tagging(text_or_list_of_words, is_list=False):
    if not is_list:
        tokens = word_tokenization(text_or_list_of_words)
        return pos_tag(tokens, tagset='universal')
    else:
        return pos_tag(text_or_list_of_words, tagset='universal')