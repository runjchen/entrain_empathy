import os
import pandas as pd
import numpy as np
from collections import defaultdict

from extract_speech_features import get_praat_features, extract_features



speakerid2gender = defaultdict(str)
gender2speakerid = defaultdict(list)
with open('speaker.txt', 'r') as file:
	speaker_info = file.readlines()
	for i in range(1, len(speaker_info)):
		speaker, gender = speaker_info[i].strip().split(',')
		speaker = speaker.strip()
		gender = gender.strip()

		speakerid2gender[speaker] = gender
		gender2speakerid[gender].append(speaker)
# print(speakerid2gender) 
# print()
# print(gender2speakerid)

speakerid2session = defaultdict(list)
session2speaker = defaultdict(dict)
with open('meta.txt', 'r') as file:
	meta = file.readlines()
	for i in range(1, len(meta)):
		session, A, B = meta[i].strip().split(',')
		session = session.strip()
		A = A.strip()
		B = B.strip()
		speakerid2session[A].append((session,'A'))
		speakerid2session[B].append((session,'B'))
		session2speaker[session] = {'A':A, 'B':B}
# print(speakerid2session)
# print(session2speaker)


def load_df(data_dir, game_name, speaker, extension, **kwargs):
	''' given full file name, load the file into a dataframe'''
	file_name = f"{game_name}.{speaker}.{extension}"
	# print(file_name)
	file_path = os.path.join(data_dir, file_name)
	df = pd.read_csv(file_path, names=["start", "end", str(extension)]
				 ,delimiter=" ")
	df["speaker"] = speaker

	for keyword in kwargs:
		df[keyword] = kwargs[keyword]

	return df


def get_trans(df_words, df_turns):
	'''get the transcript from .words files and attach to .turns files '''
	turn_trans = []
	for index, row in df_turns.iterrows():
		s, e, _ = row['start'], row['end'], row['turns']
		try:
			first_word_ind = df_words[df_words['start']==s].index[0]
			last_word_ind = df_words[df_words['end']==e].index[0]
			segment = df_words['words'][first_word_ind:last_word_ind+1]
			# trans = " ".join(segment) # remove pauses
			trans = " ".join([w for w in segment if w!="#"]) # remove pauses

		except:
			trans = ""
			print(s, e, _)
#			 print(df_words["words"][last_word_ind+1:])

		turn_trans.append(trans)
		
	df_turns['transcript'] = turn_trans

	#filter out bad transcripts and turns
	filter_trans = ~( df_turns["transcript"].isin(["#", ""]) )
	filter_turns = ~( df_turns["turns"].isin(["L", "N"]) )
	df_turns = df_turns[ filter_trans & filter_turns ]

	# df_turns = df_turns[~(df_turns["transcript"]=="#")] # remove pauses
	# df_turns = df_turns[~(df_turns["transcript"]=="L")] # remove L, N turn labels, which occur outside of conv
	# df_turns = df_turns[~(df_turns["transcript"]=="N")]
	# df_turns.dropna(inplace=True)
	return df_turns


def combine_turns(data_dir, game_name):
	'''get transcripts, extract features, combine turns from speaker A and B'''
	
	s_session, game_type, game_j = game_name.split('.')
	session = s_session[1:]
	# print('debugging',game_name)
	######################
	# speaker A
	######################
	speaker_A_id = session2speaker[session]['A']
	# print(session2speaker[session])
	gender_A = speakerid2gender[speaker_A_id]

	df_words_A = load_df(data_dir, game_name, "A", "words")
	df_turns_A = load_df(data_dir, game_name, "A", "turns", 
						**{'session':session, 'game': game_type, 'game_id': game_j,
						'speaker_id':speaker_A_id, 'gender':gender_A})
	print(game_name, '.A', sep='')
	df_A = get_trans(df_words_A, df_turns_A)

	wav_path_A = os.path.join(data_dir, f"{game_name}.A.wav")
	# print(wav_path_A)
	df_A_praat = extract_features(df_A, wav_path_A, get_praat_features)

	######################
	# speaker B
	######################
	speaker_B_id = session2speaker[session]['B']
	gender_B = speakerid2gender[speaker_B_id]

	df_words_B = load_df(data_dir, game_name, "B", "words")
	df_turns_B = load_df(data_dir, game_name, "B", "turns",
						**{'session':session, 'game':game_type, 'game_id': game_j,
						'speaker_id':speaker_B_id, 'gender':gender_B})
	print(game_name, '.B', sep='')
	df_B = get_trans(df_words_B, df_turns_B)
	
	wav_path_B = os.path.join(data_dir, f"{game_name}.B.wav")
	df_B_praat = extract_features(df_B, wav_path_B, get_praat_features)


	# combine both speakers
	df = pd.concat([df_A_praat, df_B_praat]).sort_values(by=["start", "end"], ignore_index=True)
	# return None
	return df 



def main():
	data_dir = "games"

	for i in range(1, 13): #session 0-12
		session_i = str(i).rjust(2, '0')
		
		for game_type in ["cards", "objects"]:
			
			if game_type == "cards":
				for game_j in range(1,4): # card game #1, 2, 3
					game_name = f"s{session_i}.{game_type}.{game_j}"
					print(game_name)
					df = combine_turns(data_dir, game_name)
					# if df.isnull().values.any():
					# 	print(game_name)
					df.to_csv(f"{game_name}.csv")
					
			elif game_type == "objects":
				game_j=1 # objects game #1
				game_name = f"s{session_i}.{game_type}.{game_j}"
				# print(game_name)
				
				df = combine_turns(data_dir, game_name)
				df.to_csv(f"{game_name}.csv")


if __name__ == "__main__":
	main()
			
		
