from textblob import TextBlob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet['text'] for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet['id'] for tweet in tweets])
        df['len'] = np.array([len(tweet['text']) for tweet in tweets])
        df['date'] = np.array([tweet['created_at'] for tweet in tweets])
        df['source'] = np.array([tweet['source'] for tweet in tweets])
        df['likes'] = np.array([tweet['favorite_count'] for tweet in tweets])
        df['retweets'] = np.array([tweet['retweet_count'] for tweet in tweets])

        return df

tweet_analyzer = TweetAnalyzer()

#Take in data that was saved and make JSON to analyze
data = {}
data['tweets'] = []

# Using readlines()
file1 = open('tweets.txt', 'r')
Lines = file1.readlines()

# Strips the newline character
for line in Lines:
    a = line.strip()
    data['tweets'].append(json.loads(a))

tweets = data['tweets']

df = tweet_analyzer.tweets_to_data_frame(tweets)
df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])

print(df.head(1000))
