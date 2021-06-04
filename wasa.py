
import pandas as pd
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import scipy.stats
import sklearn
import tqdm
import json
from collections import Counter




TRAINING_CSV_FILE_PATH = 'cleaned_training_tweets.csv'
TEST_CSV_FILE_PATH = 'crowdflower-apple-twitter-sentiment/data/apple_twitter_sentiment_dfe.csv'
LIWC_FILE_PATH = 'cleaned_test_tweets_LIWC_RESULTS.csv'
RADIUS = 1





def convert_to_trinary(sent, method):

	to_return = []

	for s in sent:

		if s > 0:

			to_return.append(1)

		elif s < 0:

			to_return.append(-1)

		else:

			to_return.append(0)	

	return to_return




def convert_to_trinary_baseline(sent):

	to_return = []

	for s in sent:

		if s == '5':

			to_return.append(1)

		elif s == '1':

			to_return.append(-1)

		else:

			to_return.append(0)

	return to_return







def get_afinn_sentiments(cleaned_tweets, afinn):

	sentiments = []

	for tweet in cleaned_tweets:

		try:

			s = 0
			denom = 0

			for word in tweet.split():

				if word in afinn.keys():

					sentiment_afinn = afinn[word]

				else:

					sentiment_afinn = 0

				s += sentiment_afinn
				denom += 1

			sentiments.append(s)

		except:

			sentiments.append(0)

	return sentiments







def get_mac_sentiments(cleaned_tweets, G, afinn):

	degree_dict = dict(list(G.degree(G.nodes, weight = 'weight')))

	sentiments = []

	for tweet in cleaned_tweets:

		try:
			s = 0

			denom = 0

			for word in tweet.split():

				if word in G:

					word_degree = degree_dict[word]

				else:

					word_degree = 0

				if word in afinn.keys():

					sentiment_afinn = afinn[word]

				else:

					sentiment_afinn = 0

				s += word_degree * sentiment_afinn
				denom += word_degree

			sentiments.append(s)

		except:

			sentiments.append(0)

	return sentiments








def get_new_edges(relevant_words):

	comb = combinations(relevant_words, 2)

	return(list(comb))






def form_network(tweets):

	node_pos_dict = {}
	edge_weight_dict = {}

	G = nx.Graph()

	for tweet in tqdm.tqdm(tweets):

		try:

			tweet_split = tweet.split()

			tweet_split_pos_tags_tuples = nltk.pos_tag(tweet_split)

			for index, word in enumerate(tweet_split):

				word_pos_tag_tuple = tweet_split_pos_tags_tuples[index]
				word_pos_tag = word_pos_tag_tuple[1]

				if word not in G and (word_pos_tag == 'NN' or word_pos_tag == 'JJ' or word_pos_tag[0:1] == 'VB'):

					G.add_node(word)

					node_pos_dict[word] = word_pos_tag

			for word_index in range(RADIUS, len(tweet_split) - RADIUS):

				left_index  = word_index - RADIUS
				right_index = word_index + RADIUS

				relevant_words = tweet_split[left_index:right_index + 1]

				new_edges = get_new_edges(relevant_words)

				for e in new_edges:

					if e[0] in G and e[1] in G and e[0] != e[1]:

						if not G.has_edge(*e):

							G.add_edge(*e)
							edge_weight_dict[e] = 1

						else:

							edge_weight_dict[e] += 1



		except: pass

	nx.set_node_attributes(G, node_pos_dict, 'pos')

	nx.set_edge_attributes(G, edge_weight_dict, 'weight')

	G.remove_nodes_from(list(nx.isolates(G)))

	return G







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







def get_nltk_sentiments(tweets):

	sentiments = []

	sia = SentimentIntensityAnalyzer()

	for tweet in tweets:

		try:

			scores = sia.polarity_scores(tweet)

			sentiments.append(scores['compound'])

		except:

			sentiments.append(0)

	return sentiments






def main():

	print('Reading uncleaned training tweets...')

	with open('training_tweets.txt') as f:
		lines = f.readlines()

	list_of_dicts = []

	for line in tqdm.tqdm(lines):

		json_line = json.loads(line)

		list_of_dicts.append(json_line)

	d = pd.DataFrame(list_of_dicts)

	training_tweets = list(d['text'])

	print('Done!')

	print('\n')
	print('Reading AFINN dictionary...')

	afinn = pd.read_table("AFINN/AFINN-111.txt", header = None)
	afinn = dict(zip(afinn[0], afinn[1]))

	print('Done!')

	print('\n')
	print('Reading raw test data...')

	test_data = pd.read_csv(TEST_CSV_FILE_PATH, header = 0)

	print('Done!')

	print('\n')
	print('Gathering baseline human ratings from raw test data...')

	baseline = convert_to_trinary_baseline(list(test_data['sentiment']))
	baseline_sentiment = [a*b for a,b in zip(list(test_data['sentiment_confidence']),baseline)]

	print('Done!')

	print('\n')
	print('Gathering cleaned test tweets...')

	# Only use if cleaning new test tweets
	#cleaned_test_tweets = clean(test_tweets)
	#df = pd.DataFrame(cleaned_test_tweets)
	#df.to_csv('cleaned_test_tweets.csv', header = False, index = False)	

	# Only use if reading cleaned test tweets from CSV file
	cleaned_test_tweets = pd.read_csv('cleaned_test_tweets.csv', header = None)
	cleaned_test_tweets = list(cleaned_test_tweets[0])

	print('Done!')

	#print('\n')
	#print('Gathering cleaned training data...')

	# Only use if have new training tweets to clean
	#cleaned_training_tweets = clean(training_tweets)

	# Only use if already have clean training tweets in a CSV file
	##cleaned_training_tweets = pd.read_csv(TRAINING_CSV_FILE_PATH, header = None)
	##cleaned_training_tweets = list(cleaned_training_tweets[0])

	#print('Done!')

	print('\n')
	print('Forming network...')

	G = form_network(training_tweets)

	print('Done')

	print('\n')
	print('Gathering J/A, AFINN, NLTK, LIWC sentiments...')

	mac_sentiments = get_mac_sentiments(cleaned_test_tweets, G, afinn)
	afinn_sentiments = get_afinn_sentiments(cleaned_test_tweets, afinn)
	nltk_sentiments = get_nltk_sentiments(cleaned_test_tweets)

	liwc_df = pd.read_csv(LIWC_FILE_PATH, header = 0)	
	liwc_sentiments = list(liwc_df['posemo'].sub(liwc_df['negemo']))

	print('Done!')

	print('\n')
	print('Gathering predicted scores from various methods...')

	predicted_liwc = convert_to_trinary(liwc_sentiments, 'liwc')
	predicted_nltk = convert_to_trinary(nltk_sentiments, 'nltk')
	predicted_mac = convert_to_trinary(mac_sentiments, 'mac')
	predicted_afinn = convert_to_trinary(afinn_sentiments, 'afinn')

	print('Done!')

	print('\n')
	print('Counts:')
	print(Counter(baseline).keys())
	print(Counter(baseline).values())
	print('\n')


	final_liwc = []
	final_nltk = []
	final_mac = []
	final_afinn = []

	for i, gt in enumerate(baseline):

		if gt == predicted_mac[i]:

			final_mac.append(1)

		else:

		 	final_mac.append(0)

		if gt == predicted_afinn[i]:

		 	final_afinn.append(1)

		else:

		 	final_afinn.append(0)

		if gt == predicted_liwc[i]:

		 	final_liwc.append(1)

		else:

		 	final_liwc.append(0)

		if gt == predicted_nltk[i]:

		 	final_nltk.append(1)

		else:

		 	final_nltk.append(0)

	print('\n')
	print('Pct Correct:')
	print('liwc', sum(final_liwc) / len(final_liwc))
	print('nltk', sum(final_nltk) / len(final_nltk))
	print('afinn', sum(final_afinn) / len(final_afinn))
	print('joseph,alex', sum(final_mac) / len(final_mac))
	print('\n')

	print('\n')
	print('F1 Scores:')
	print('liwc', sklearn.metrics.f1_score(predicted_liwc, baseline, average = 'weighted'))
	print('nltk', sklearn.metrics.f1_score(predicted_nltk, baseline, average = 'weighted'))
	print('afinn', sklearn.metrics.f1_score(predicted_afinn, baseline, average = 'weighted'))
	print('joseph,alex', sklearn.metrics.f1_score(predicted_mac, baseline, average = 'weighted'))
	print('\n')








main()