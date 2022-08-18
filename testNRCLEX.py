from normalization_functions import *
from nrclex import NRCLex

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
