from argparse import RawDescriptionHelpFormatter
from analytics_functions import *

def text_to_list(url):
    raw = text_reader(url)
    res = [word for word in raw.split()]
    return res

def list_to_text(new_url, words_list):
    new_file = open(new_url,'a')
    for word in words_list: 
        new_file.write("\n" + word)
    new_file.close()
    return True

def list_cleaner(sentiment_list):
    res = list()
    for word in sentiment_list:
        word = re.sub(r"[-]", " ", word)
        res.append(word)
    return res 


positive_list = text_to_list("Classifier/Sentiment/RawSentiment/positive_words.txt")
positive_list = list_cleaner(positive_list)
negative_list = text_to_list("Classifier/Sentiment/RawSentiment/negative_words.txt")
negative_list = list_cleaner(negative_list)
neutral_list = text_to_list("Classifier/Sentiment/RawSentiment/neutral_words.txt")
neutral_list = list_cleaner(neutral_list)

scored_sentiment = open("Classifier/Sentiment/RawSentiment/AFINN-111.txt", "r")

# I strip lines and fix contractions in order to add to the respective list the word/s
raw_scored = list()
for line in scored_sentiment:
    stripped_line = line.strip()
    raw_line = stripped_line.split()
    line_obj = list()
    for word in raw_line:
        line_obj.append(contractions.fix(word))
    raw_scored.append(line_obj)
scored_sentiment.close()

for temp_list in raw_scored:
    word = " ".join(temp_list[:-1])
    if int(temp_list[-1]) < 0:
        negative_list.append(word)
    elif int(temp_list[-1]) > 0:
        positive_list.append(word)
    elif int(temp_list[-1]) == 0:
        neutral_list.append(word)

negative = set(negative_list)
neutral = set(neutral_list)
positive = set(positive_list)
negative = list_cleaner(negative)
neutral = list_cleaner(neutral)
positive = list_cleaner(positive)

list_to_text("Classifier/Sentiment/SentimentCorpus/clean_positive.txt", positive)
list_to_text("Classifier/Sentiment/SentimentCorpus/clean_negative.txt", negative)
list_to_text("Classifier/Sentiment/SentimentCorpus/clean_neutral.txt", neutral)


sentiment_corpus = create_corpus("Classifier/Sentiment/SentimentCorpus/")

connotations_list = ([(word.lower(), "POS") for word in sentiment_corpus.words("clean_positive.txt")] +
                        [(word.lower(), "NEG") for word in sentiment_corpus.words("clean_negative.txt")] +
                        [(word.lower(), "NEU") for word in sentiment_corpus.words("clean_neutral.txt")])
