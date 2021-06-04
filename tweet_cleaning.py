
import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
import string
import tqdm






def clean(tweets):

	cleaned_tweets = []

	for tweet in tqdm.tqdm(tweets):

		# The next 7 lines of code were obtained from 
		# https://machinelearningmastery.com/clean-text-machine-learning-python/
		# These are for text cleaning
		###
		tokens = word_tokenize(tweet)
		# convert to lower case
		tokens = [w.lower() for w in tokens]
		# remove punctuation from each word
		table = str.maketrans('', '', string.punctuation)
		stripped = [w.translate(table) for w in tokens]
		# remove remaining tokens that are not alphabetic
		words_l = [word for word in stripped if word.isalpha()]
		# filter out stop words
		stop_words = set(stopwords.words('english'))
		words_l = [w for w in words_l if not w in stop_words]
		###

		words_l = [w for w in words_l if w in words.words()]

		words_l = ' '.join(words_l)

		if words_l != ' ':
			cleaned_tweets.append(words_l)

	return cleaned_tweets






def main():

	with open('training_tweets.txt') as f:
		lines = f.readlines()

	list_of_dicts = []

	for line in tqdm.tqdm(lines):

		json_line = json.loads(line)

		list_of_dicts.append(json_line)

	d = pd.DataFrame(list_of_dicts)

	tweets = list(d['text'])

	cleaned_tweets = clean(tweets)

	df = pd.DataFrame(cleaned_tweets)

	df.to_csv('cleaned_training_tweets.csv', header = False, index = False)








main()
