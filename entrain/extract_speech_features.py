import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

import matplotlib.pyplot as plt
from matplotlib.pyplot import errorbar, boxplot
import seaborn

from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score 

from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize


def get_praat_features(sound, *meta_info):
    
    # initialize segments info, default list of dicts, each dict keys: start, end, *speaker/dialog/turn
    # if no timestamps provided, we look at entire sound
    # meta_info is a tuple
    
    if len(meta_info)==0:
        segments_info = [{'start': 0.0, 'end': 0.0}]
    else:
        segments_info = meta_info[0]
    
    # praat objects
    pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    intensity = call(sound, "To Intensity", 100.0, 0.0)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    
    praat_features = []
    
    for seg_info in segments_info:
        
        start, end = seg_info['start'], seg_info['end']
        
        #pitch
        min_pitch = call(pitch, "Get minimum", start, end, "Hertz", "Parabolic")
        max_pitch = call(pitch, "Get maximum", start, end, "Hertz", "Parabolic")
        mean_pitch = call(pitch, "Get mean", start, end, "Hertz")
        sd_pitch = call(pitch, "Get standard deviation", start, end, "Hertz")
        
        #intensity
        min_intensity = call(intensity, "Get minimum", start, end, "Parabolic")
        max_intensity = call(intensity, "Get maximum", start, end, "Parabolic")
        mean_intensity = call(intensity, "Get mean", start, end, "energy")
        sd_intensity = call(intensity, "Get standard deviation", start, end)

        # jitter, shimmer
        jitter = call(point_process, "Get jitter (local)", start, end, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", start, end, 0.0001, 0.02, 1.3, 1.6)

        # HNR
        hnr = call(harmonicity, "Get mean", start, end)

        #speaking rate
        speaking_rate = get_word_per_sec(seg_info)


        praat_features.append({
            **seg_info,
            'min_pitch': min_pitch, 'max_pitch': max_pitch,
            'mean_pitch': mean_pitch, 'sd_pitch': sd_pitch,
            'min_intensity': min_intensity, 'max_intensity':max_intensity, 
            'mean_intensity': mean_intensity, 'sd_intensity': sd_intensity,
            'jitter': jitter, 'shimmer': shimmer, 'hnr': hnr,
            'speaking_rate': speaking_rate,
            })
    return praat_features


def extract_features(df, audio_path, feature_func):
    '''extract features given timestamps for an audio file'''
    sound = parselmouth.Sound(audio_path)
    extracted_features = feature_func(sound, df.to_dict('records'))
    df_features = pd.DataFrame(extracted_features)

    return df_features



def get_word_per_sec(row):
    s = row['transcript']
    tokenized = word_tokenize(s)
    word_count = len(tokenized)
    
    time = row['end'] - row['start']
    
    return word_count/time

if __name__ == "__main__ ":
    
    pass

# ##### EXTRACT PRAAT FEATURES ###############

# praat_extracted = extract_features(df, WAV_FILE_PATH, get_praat_features)

# df_praat = pd.DataFrame(praat_extracted)

# df_praat
# ##### EXTRACT PRAAT FEATURES ###############