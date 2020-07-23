"""
This file is used to retrieve the tweets by id using the Twitter's APIs with the tweepy library.
After a tweet is retrieved it is saved saved as a set of characteristics:
[Id,Tweet,is_retweet,following,followers,actions,label]

NOTE: there is a limit of 900 requests each 15 minutes (the library will automatically wait)
"""

import csv
import tweepy as tw

# The keys have been omitted for obvious reasons
consumer_key = "//"
consumer_secret = "//"
access_token= "//"
access_token_secret= "//"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


def read_and_write_dataset():
  with open('./Datasets/dataset_3.csv', "a", newline='') as csv_file_output:
    csv_writer = csv.writer(csv_file_output, delimiter=',')
    with open('./Datasets/tweets_id_dataset_3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_row = True
        for row in csv_reader:
          if not first_row:
            try:
              tweet_id = row[0]
              tweet_label = row[1]
              tweet = api.get_status(tweet_id)
              csv_writer.writerow([tweet_id, tweet.text.replace('\n', ' '), 1 if tweet.retweeted else 0, tweet.user.friends_count, tweet.user.followers_count, tweet.retweet_count + tweet.favorite_count, tweet_label])
              csv_file_output.flush()
            except BaseException as error:
              None
          first_row = False


read_and_write_dataset()