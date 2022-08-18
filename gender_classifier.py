from normalization_functions import *
from analytics_functions import *
from textblob import TextBlob
from csv import DictWriter

"""
names_corpus = create_corpus("Classifier/Names/")
# This list will contain tuples with a name[0] and its gender[1] (properly male or female)
names_list = ([(name,'male') for name in names.words("male.txt")] +
                        [(name,'female') for name in names.words("female.txt")] +
                        [(name.lower().capitalize(),'male') for name in names_corpus.words("male_names.txt")] +
                        [(name.lower().capitalize(),'female') for name in names_corpus.words("female_names.txt")])
names_list = set(names_list)

# Even though I've looked for other peculiar features for gender classification in names
# I found that the most representative is this one, in fact adding other (not too much complex)
# features (such as number and order of "a" in a name) does not increase accuravy and sometimes it
# even decrease accuracy level...

feature_list = [(gender_feature_last_char(name),gender) for (name,gender) in names_list]

total_names = len(feature_list) - 1
mid = total_names - 1
train_list,test_list = feature_list[:mid],feature_list[mid:]

# this is a test to see the percentage of correct answers given by the classifier
# In reality we train our classifier with the whole of names
test_classifier = nltk.NaiveBayesClassifier.train(train_list)
print(nltk.classify.accuracy(test_classifier, test_list))

#gender_classifier = nltk.NaiveBayesClassifier.train(feature_list)

#print(test_classifier.classify(gender_feature_last_char("Luca")))

"""

#---------------------------------------------------------
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
    #here could be better to rejoin the sentence and tokenize it again with contractions mainteined and punctuation remove???
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
    #should check [To be done as first operation] if there is a proper 
    # name in the sentence to which the sentence is referred

    #create list of sentences
    list_of_sentences = syntok_list_of_sentences(text)
    #tokenization NOT for analysis
    for sentence in list_of_sentences:
        for word in word_tokenization(sentence, True, False)[1:]:
            is_it_proper(word, proper_nouns_dict)
        #With "expand_contractions" I also tokenize the text
        sentence_tokens = expand_contractions(sentence, False, True, True)
        sentence_length = len(sentence_tokens)
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
        
