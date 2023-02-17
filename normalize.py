import os
import numpy as np
import pandas as pd
from collections import defaultdict

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


'''
	Input: directory containing raw feature csv files
	Output: z-score normalized features

'''
SPEECH_FEATURES = ['min_pitch', 'max_pitch','mean_pitch', 'sd_pitch',
            		'min_intensity', 'max_intensity', 
            		'mean_intensity', 'sd_intensity',
            		'jitter', 'shimmer', 'hnr', 'speaking_rate']

def get_distribution(df, feature_list, by):
	
	assert by in ['speaker', 'gender'], "'by' needs to 'speaker' or 'gender'"

	distribution = defaultdict(dict)
	for group in df[by].unique():
		filter = (df[by]==group)
		distribution[group] = {'mean': df[filter][feature_list].mean(axis=0),
								'std': df[filter][feature_list].std(axis=0)}


	return distribution

def get_z_score(filtered_data, mean, std):
	'''x = (x-mean)/std'''
	return (filtered_data - mean)/std
	

def main():

	# combine all raw feature csv files
	raw_feature_files = [file for file in os.listdir('.') if file.endswith(".csv")]
	df_all = pd.concat(map(pd.read_csv, raw_feature_files), ignore_index=True)

	# calculate mean and stds
	# distribution_by_gender = get_distribution(df, SPEECH_FEATURES, 'gender')
	# distribution_by_speaker = get_distribution(df, SPEECH_FEATURES, 'speaker')

	# z-score normalize features
	for by in ['gender', 'speaker']:
		# calculate mean and stds
		distribution = get_distribution(df_all, SPEECH_FEATURES, by)

		# Check whether the specified path exists or not
		path = f"{by}_normed_feature"
		if not os.path.exists(path):
			os.makedirs(path)

		for file in tqdm(raw_feature_files):
			df = pd.read_csv(file, index_col=0)

			for group in df[by].unique():
				filter = (df[by]==group)
				normed_feature = get_z_score(df[filter][SPEECH_FEATURES],
							distribution[group]['mean'],
							distribution[group]['std'])
				df.loc[filter, SPEECH_FEATURES] = normed_feature

			df.to_csv(f"{path}/{by}_normed_{file}")


if __name__ == '__main__':
	main()


