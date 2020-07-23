"""With this third analysis I wanted to try something as real as possible. I found online a list of labelled tweet IDs with no other information.
The referenced paper and dataset are available here: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182487
To work with these I had to subscribe to the Twitter Developers site, obtain the API keys, use them to recover the tweets text and information,
preprocess them creating the feature sets and then train and test the ML models.
"""

# Various imports
import math
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Setting the path of the dataset
dataset_location = './Datasets/dataset_3.csv'


# Creation of the spamwords array, this will be used to evaluate the "number of spamwords" feature of the tweets
spam_words_arr = ['$$$', '£££', 'accounts', 'additional', 'bank', 'bonus', 'cash', 'cost', 'credit', 'earn', '$', 'finance', 'advice', 'freedom', 'investment', 'insurance',
                  'decision', 'invoice', '£', 'million', 'account', 'potential', 'earnings', 'refund', 'risk', 'save', 'stock', 'thousands', 'trade', 'Dollars', 'ad',
                  'amazing', 'bargain', 'offer', 'cheap', 'clearance', 'congratulations', 'dear', 'market', 'marketing', 'delete', 'fantastic', 'free', 'trial', 'gift',
                  'increase', 'incredible', 'sales', 'traffic', 'junk', 'member', 'easy', 'expires', 'extended', 'fast', 'opportunity', 'performance', 'promise', 'sale',
                  'spam', 'special', 'promotion', 'subscribe', 'promo', 'form', 'quick', 'urgent', 'unbeatable', 'unsubscribe', 'visit', 'website', 'win', 'winner', 'boss',
                  'hosting', 'paid', 'business', 'income', 'profit', 'sleep', 'advertisement', 'mlm', '$$$', '@mlm', '100%', 'satisfied', 'cards', 'money', 'guarantee', 'info',
                  'order', 'now!', '18+', 'adult', 'adults', 'over', '18', '21', 'xxx']


# Reading the dataset and dropping useless columns
def read_dataset(filename):
    data = pd.read_csv(filename)
    print('____________________________________________\n_______________ Dataset Info _______________\n____________________________________________')
    print('Number of features: {}'.format(data.shape[1]))
    print('Number of examples: {}'.format(data.shape[0]))
    print(data['label'].value_counts())
    print('\nDataset read:')
    print(data.head())
    return data.drop(['Id'], axis=1)

dataset = read_dataset(dataset_location)
print('\n')


# Method used to create the feature set of a given tweet

def extract_features(tweet):
  words_arr = tweet[0].split(" ")
  tweet_length = len(words_arr)

  urls = 0;
  hashtags = 0;
  spamword_count = 0;
  digits = 0;
  for word in words_arr:
    word = word.lower()
    if word.startswith('#'):
      hashtags += 1
      if word[1:] in spam_words_arr:
        spamword_count += 1
      if isDigit(word[1:]):
        digits += 1
      if 'http' in word or '.com' in word:
        urls += 1

    elif word in spam_words_arr:
      spamword_count += 1

    elif isDigit(word):
      digits += 1

    elif word.startswith('http') or '.com' in word:
      urls += 1

  return [spamword_count, hashtags/tweet_length, urls/tweet_length, tweet_length, digits, urls, hashtags, tweet[1] if not math.isnan(tweet[1]) else 0, tweet[2] if not math.isnan(tweet[2]) else 0, tweet[3] if not math.isnan(tweet[3]) else 0, tweet[4] if not math.isnan(tweet[4]) else 0]

def isDigit(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


# Creating the feature set for each tweet and splitting the dataset in train and test set
Xtrain = []
Ytrain = []
dataset_list = dataset.values.tolist()

for tweet in dataset_list:
  Xtrain.append(extract_features(tweet))
  Ytrain.append(tweet[5])

def split_dataset(dataset, labels, percentage):
  index = int(round((len(dataset)/100) * percentage, 0))
  return (dataset[0: index], labels[0: index], dataset[index:], labels[index:])

Xtrain, Ytrain, Xtest, Ytest =  split_dataset(Xtrain, Ytrain, 80)

print('Training set size: ', len(Xtrain), '\nTest set size: ', len(Xtest), '\n')


print('____________________________________________\n____________ SVM classification ____________\n____________________________________________')
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(Xtrain, Ytrain)
predictions = clf.predict(Xtest)
print('Predictions: \n', predictions)
print('\nConfusion matrix: \n', confusion_matrix(Ytest, predictions))
print('\nEvaluation metrics: \n',classification_report(Ytest, predictions))


print('____________________________________________\n____________ KNN classification ____________\n____________________________________________')
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)
clf = make_pipeline(StandardScaler(), neigh)
clf.fit(Xtrain, Ytrain)
predictions = clf.predict(Xtest)
print('Predictions: \n', predictions)
print('\nConfusion matrix: \n', confusion_matrix(Ytest, predictions))
print('\nEvaluation metrics: \n',classification_report(Ytest, predictions))


print('____________________________________________\n_______ Random Forest classification _______\n____________________________________________')
rand_for = RandomForestClassifier(criterion='entropy', n_estimators=100)
clf = make_pipeline(StandardScaler(), rand_for)
clf.fit(Xtrain, Ytrain)
predictions = clf.predict(Xtest)
print('Predictions: \n', predictions)
print('\nConfusion matrix: \n', confusion_matrix(Ytest, predictions))
print('\nEvaluation metrics: \n',classification_report(Ytest, predictions))