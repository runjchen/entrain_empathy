import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

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

SPEECH_FEATURES = ['min_pitch', 'max_pitch','mean_pitch', 'sd_pitch',
            		'min_intensity', 'max_intensity', 
            		'mean_intensity', 'sd_intensity',
            		'jitter', 'shimmer', 'hnr', 'speaking_rate']

# combine all normed feature csv files
folder = 'gender_normed_feature'
raw_feature_files = [file for file in os.listdir(folder) if file.endswith(".csv")]


def proximity(df, feature_names, context_lengh=3, save_as="output.pkl"):
	
	output = []
	for i in range(context_lengh, len(df)):

		# join the previous turns
		context = " ".join(df['transcript'][i-context_lengh:i].str.strip())
		sentence = df['transcript'][i].strip()

		# get local feature differences
		average = df.iloc[i-context_lengh:i+1].groupby('speaker').mean()
		try:
			diff = np.abs( (average.loc['A'] - average.loc['B'])[feature_names].values )
			output.append((context, sentence, diff))
		except:
			pass

	# save the list of tuples (context, sentence, partner_differences)
	with open(save_as, 'wb') as f:
		pickle.dump(output, f)

	print(f'saved as {save_as}.')

	pass

def proximity_multiple(folder,feature_names):
	raw_feature_files = [file for file in os.listdir(folder) if file.endswith(".csv")]

	# Check whether the specified path exists or not
	if not os.path.exists('entrain_data'):
		os.makedirs('entrain_data')

	for f in raw_feature_files:
		proximity(pd.read_csv(folder+'/'+f),
	 		feature_names, context_lengh=3, 
	 		save_as=f"entrain_data/{f}.pkl")
	pass

if __name__ == '__main__':
	proximity_multiple(folder, SPEECH_FEATURES)