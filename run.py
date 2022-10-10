from functions.normalization_functions import *
from functions.analytics_functions import *
from functions.visualizations_functions import *
from functions.sentiment_functions import *

# Creation of corpora
f_directory = "assets/Raw corpora/F/"
m_directory = "assets/Raw corpora/M/"
f_corpus = create_corpus(f_directory)
m_corpus = create_corpus(m_directory)

# Storing titles and URLs of texts
f_authors_texts = list()
f_titles = list()
for url in f_corpus.fileids():
    if url != '.DS_Store':
        f_authors_texts.append(f_directory+url)
        f_titles.append(url[:-4])

m_authors_texts = list()
m_titles = list()
for url in m_corpus.fileids():
    if url != '.DS_Store':
        m_authors_texts.append(m_directory+url)
        m_titles.append(url[:-4])

# Useful elements

# As a first step we will need two different sets of words associated 
# with female and male context.
# Such sets have been taken from the Jailbreak-the-Patriarchy Project
# (https://github.com/DanielleSucher/Jailbreak-the-Patriarchy)
# and enhanced where necessary, according to our view and to our context.
m_words = set(['guy','dr','spokesman','chairman',"men's",'men','him',"he's",'his','boy',
'boyfriend','boyfriends','boys','brother','brothers','dad','dads','dude','father',
'fathers','fiance','gentleman','gentlemen','god','godfather','grandfather','grandpa',
'grandson','groom','he','himself','his','husband','pastor','husbands','king','male','man',
'mr','nephew','nephews','priest','prince','son','sons','uncle','uncles',
'waiter','widower','widowers','he','he is','lord','master','earl','don'])
male_words = set()
for word in m_words:
    male_words.add(word.lower())
    male_words.add(word.capitalize())

f_words=set(['heroine','drss','spokeswoman','chairwoman',"women's",'actress','women',
"she's",'her','aunt','aunts','bride','daughter','daughters','female','fiancee','girl',
'girlfriend','girlfriends','girls','goddess','godmother','granddaughter','grandma','grandmother',
'herself','ladies','lady',"lady's",'miss','mom','moms','mother','mothers','mrs','ms','niece',
'nieces','priestess','princess','queens','she','sister','sisters','waitress',
'widow','widows','wife','wives','woman','she','she is','lady','mistress','queen',
'grandmama','mamma','dame','countess','missie'])
female_words = set()
for word in f_words:
    female_words.add(word.lower())
    female_words.add(word.capitalize())

# Seen that we are working with children's literature, one possible issue is the 
# recognition of animals and the possible imbalance that may derive in the statistics.
# To avoid the problem, we took this animals name txt file 
# (https://gist.github.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54) 
# and compiled a list from it.
# Such list will be useful to clear the words that we will analyze.
animals = text_reader("assets/Useful elements and texts/animals_list.txt")
animals_list = list()
for row in animals.split():
    animals_list.append(row.lower())
    animals_list.append(row.capitalize())
animals_list = set(animals_list)

# The same will be performed for a list of common words that we have 
# adapted from the reddit analysis project
# (https://github.com/rhiever/reddit-analysis/blob/master/redditanalysis/words/common-words.txt).
commons = text_reader("assets/Useful elements and texts/common_ws_list.txt")
common_ws_list = list()
for row in commons.split():
    common_ws_list.append(row.lower())
    common_ws_list.append(row.capitalize())
common_ws_list = set(common_ws_list)

# Now we extract all the not-names words that are in general capitalized
capital_not_names = list()

geo_names = text_reader("assets/Useful elements and texts/common_geographical_names.txt")
for row in geo_names.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

countries = text_reader("assets/Useful elements and texts/countries.txt")
for row in countries.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

exclamations = text_reader("assets/Useful elements and texts/exclamations.txt")
for row in exclamations.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

nationalities = text_reader("assets/Useful elements and texts/nationalities.txt")
for row in nationalities.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

religious = text_reader("assets/Useful elements and texts/religious_words.txt")
for row in religious.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

cities_uk = text_reader("assets/Useful elements and texts/UK_cities.txt")
for row in cities_uk.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

countries_uk = text_reader("assets/Useful elements and texts/UK_counties.txt")
for row in countries_uk.split():
    capital_not_names.append(row.lower())
    capital_not_names.append(row.capitalize())

capital_not_names = set(capital_not_names)


# Creating some basic elements useful to store and analyze the information 
# that we will extract from our corpora and to compute some statistics.
# The reason behind the addition of "mainly" classes is for a possible future 
# upgrade of the project
list_of_gender = ['male','female','mainly_male','mainly_female','none','both']

f_sentence_counter = {sex:0 for sex in list_of_gender}
f_word_counter = {sex:0 for sex in list_of_gender}
f_word_freq = {sex:{} for sex in list_of_gender}
f_sent_dict = {}

m_sentence_counter = {sex:0 for sex in list_of_gender}
m_word_counter = {sex:0 for sex in list_of_gender}
m_word_freq = {sex:{} for sex in list_of_gender}
m_sent_dict = {}

male_authors_raw_count = dict()
female_authors_raw_count = dict()

f_emotions = dict()
m_emotions = dict()

#For female authors
f_male_general = set()
f_male_nouns = set()
f_male_verbs = set()
f_male_adjectives = set()

f_female_general = set()
f_female_nouns = set()
f_female_verbs = set()
f_female_adjectives = set()

#For male authors
m_male_general = set()
m_male_nouns = set()
m_male_verbs = set()
m_male_adjectives = set()

m_female_general = set()
m_female_nouns = set()
m_female_verbs = set()
m_female_adjectives = set()