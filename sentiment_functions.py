from normalization_functions import *
from analytics_functions import *
from textblob import TextBlob
from nrclex import NRCLex

# GENDER THE SENTENCE

#The function below takes a work list and returns the gender of the person being 
# talked about, if any, based on the number of words a sentence has in common with 
# either the male or female word lists.
def gender_the_sentence(sentence_words, male_words, female_words):
    male_w = len(male_words.intersection(sentence_words))
    female_w = len(female_words.intersection(sentence_words))
    #male_n = len(male_names_heidi.intersection(sentence_words))
    #female_n = len(female_names_heidi.intersection(sentence_words))
    #male_w += male_n
    #female_w += female_n
    if male_w > 0 and female_w == 0:
        gender = 'male'
    elif male_w == 0 and female_w > 0: 
        gender = 'female'
    elif male_w > female_w:
        gender = 'mainly_male'
    elif male_w < female_w:
        gender = 'mainly_female'
    elif male_w == female_w and male_w != 0: 
        gender = 'both'
    else:
        gender = 'none'
    return gender

def increment_gender(sentence_tokens, gender, sentence_dict, words_dict, freq_dict):
    # words_dict -> contains the total number of words used in the gendered sentences (ex. 20 male sentences of 10 words: words_dict["male"]=200)
    # freq_dict -> contains the total number of occurrences of a specific word in each gendered sentence
    sentence_dict[gender] += 1
    words_dict[gender] += len(sentence_tokens)
    for token in sentence_tokens:
        #This script has the aim to find (if exists) the word inside the word_freq
        # dictionary and to increment it by one, if the key (that is the word) does
        # not exist, then it will add the word with value = 0 and increment the value
        # by 1, ex: .get(key, default_value_if_key_not_found)
        freq_dict[gender][token] = freq_dict[gender].get(token,0) + 1  

def sentiment_analysis(sentence):
    return TextBlob(sentence).sentiment.polarity

# remove punctuation before words
def is_it_proper(word, proper_nouns):
        if word[0] == word[0].upper() and word[0] not in ['"',"'",'.',',','/','-']:
            if len(word) > 1:
                if word[0] != "I" and word[1] != "'":
                    case = 'upper'
                elif word[0] != "I" and word[1] != ' ':
                    case = 'upper'
                else:
                    case = 'lower'
            else:
                case = 'lower'
        else:
            case = 'lower'
        word_lower = word.lower()
        
        try:
            proper_nouns[word_lower][case] = proper_nouns[word_lower].get(case,0) + 1
        except:
            proper_nouns[word_lower] = {case:1}

def emotion_frequencies(url, emotion_dict):
    text = text_reader(url)
    emo_analyzer = NRCLex(text) 
    emotions = emo_analyzer.affect_frequencies
    for emo in emotions:
        if emo not in emotion_dict:
            emotion_dict[emo] = emotions[emo]
        else:
            emotion_dict[emo] = (emotion_dict[emo] + emotions[emo])/2
    return emotion_dict

def gender_analysis(text, sentences_dict_df, sentence_dict, words_dict, raw_words_dict, freq_dict, proper_nouns_dict, male_words, female_words):
    #create list of sentences
    list_of_sentences = syntok_list_of_sentences(text)
    #tokenization not for analysis
    for sentence in list_of_sentences:
        sentences_dict_df[sentence] = {
                                        "number_of_tokens":0,
                                        "gender":"",
                                        "polarity":"",
                                        "score":""
                                        }
        for word in word_tokenization(sentence, True, False)[1:]:
            is_it_proper(word, proper_nouns_dict)
        #With "expand_contractions" I also tokenize the text
        sentence_tokens = expand_contractions(sentence, False, True, True)
        sentence_tokens = lemmatization(sentence_tokens)
        for token in sentence_tokens:
            if token in stopwords or token in ['"',"'",'.',',','/','-']:
                sentence_tokens.remove(token)
            else:
                if token not in raw_words_dict:
                    raw_words_dict[token] = 1
                else:
                    raw_words_dict[token] += 1
        sentences_dict_df[sentence]["number_of_tokens"] = len(sentence_tokens)
        
        #gender the sentence
        gender = gender_the_sentence(sentence_tokens, male_words, female_words)
        sentences_dict_df[sentence]["gender"] = gender
        increment_gender(sentence_tokens, gender, sentence_dict, words_dict, freq_dict)
        polarity_score = sentiment_analysis(sentence)
        sentences_dict_df[sentence]["score"] = polarity_score

        if polarity_score < 0:
            polarity = 'NEG'
        elif polarity_score > 0:
            polarity = 'POS'
        else:
            polarity = 'NEU'
        sentences_dict_df[sentence]["polarity"] = polarity