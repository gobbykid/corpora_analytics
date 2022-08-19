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

def gender_analysis(text, sentence_dict, words_dict, raw_words_dict, freq_dict, sentiment_dict, proper_nouns_dict, male_words, female_words):
    #create list of sentences
    list_of_sentences = syntok_list_of_sentences(text)
    #tokenization not for analysis
    for sentence in list_of_sentences:
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
            
        #gender the sentence
        gender = gender_the_sentence(sentence_tokens, male_words, female_words)
        increment_gender(sentence_tokens, gender, sentence_dict, words_dict, freq_dict)
        polarity_score = sentiment_analysis(sentence)

        if polarity_score < 0:
            polarity = 'NEG'
        elif polarity_score > 0:
            polarity = 'POS'
        else:
            polarity = 'NEU'

        if gender == 'male':
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'female': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'mainly_male': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'mainly_female': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'both': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        else:
            sentiment_dict[gender][polarity][sentence] = polarity_score

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

def sentence_emotion_analyzer(sentence, sentences_dict_df):
    emo_analyzer = NRCLex(sentence) 
    emotions = emo_analyzer.affect_frequencies
    for emo in emotions:
        if emo != "anticip":
            sentences_dict_df[sentence][emo] = emotions[emo]
    

def test_gender_analysis(text, sentences_dict_df, sentence_dict, words_dict, raw_words_dict, freq_dict, sentiment_dict, proper_nouns_dict, male_words, female_words):
    #create list of sentences
    list_of_sentences = syntok_list_of_sentences(text)
    #tokenization not for analysis
    for sentence in list_of_sentences:
        sentences_dict_df[sentence] = {
                                        "gender":"",
                                        "polarity":"",
                                        "score":"",
                                        'fear': 0,
                                        'anger': 0,
                                        'trust': 0,
                                        'surprise': 0,
                                        'positive': 0,
                                        'negative': 0,
                                        'sadness': 0,
                                        'disgust': 0,
                                        'joy': 0,
                                        'anticipation': 0
                                        }
        sentence_emotion_analyzer(sentence, sentences_dict_df)
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

        if gender == 'male':
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'female': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'mainly_male': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'mainly_female': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        elif gender == 'both': 
            sentiment_dict[gender][polarity][sentence] = polarity_score
        else:
            sentiment_dict[gender][polarity][sentence] = polarity_score

sentences_dict_df = {
    "sentence":{
        "gender":"",
        "polarity":"",
        "score":"",
        'fear': 0.06581840024547972,
        'anger': 0.04010545517783605,
        'trust': 0.11695044598034351,
        'surprise': 0.0732976982030891,
        'positive': 0.17503607155019787,
        'negative': 0.17390092703781612,
        'sadness': 0.08731623909076737,
        'disgust': 0.04257919349290297,
        'joy': 0.11072519834706926,
        'anticipation': 0.11427037087449804
    }
}