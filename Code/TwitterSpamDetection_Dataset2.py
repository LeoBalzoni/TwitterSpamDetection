"""Now I will take a look at the second approach I tried to emulate what was done in the papers.
Since the issue with the first approach is how different the feature set is from the one used in the papers I studied, I started looking
for a dataset that provided the actual tweet text with it, so that I could generate a feature set as similar as possible to the one from the papers.
The one I settled on is available at https://www.kaggle.com/c/twitter-spam/overview. And what is good about this one is that each entry of the dataset has the following shape:
[tweet_text, num_of_following, num_of_followers, num_of_actions, is_retweet, type]
The main factor is the presence of the actual tweet text, from which I can extract the features used in the paper.
The code is basically the same of the one from the firts dataset, so there is no need to provide further explanations."""

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
dataset_location = './Datasets/dataset_2.csv'


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
    print(data['Type'].value_counts())
    print('\nDataset read:')
    print(data.head())
    return data.drop(['Id', 'location'], axis=1)

dataset = read_dataset(dataset_location)
print('\n')

"""Now in the following block of code I generate the feature set for each tweet. 
The one I am creating here is basically identical to the one used in the paper I studied, parsing the tweet we count the spamwords, the urls, the hashtags, 
and the digits so that the following feature array can be created:
[spam_words_count, num_of_hashtag_per_word, num_of_urls_per_word, num_of_words, num_of_digits, num_of_urls, num_of_hashtags, num_of_actions, is_retweeted, num_of_following, num_of_followers]"""


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

    return [spamword_count, hashtags / tweet_length, urls / tweet_length, tweet_length, digits, urls, hashtags,
            tweet[3] if not math.isnan(tweet[3]) else 0, tweet[4] if not math.isnan(tweet[4]) else 0,
            tweet[1] if not math.isnan(tweet[1]) else 0, tweet[2] if not math.isnan(tweet[2]) else 0]


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

"""Now that we have a feature set that is basicaly identical to the one used in the papers I will run the same algorithms I already tried with 
my first approach and see how this new feature set affects the results.
It's easy to see how this new, and improved feature set drastically improves the results. I think that the main improvement factor is given 
from the presence of the "num_of_spam_words" feature.
Let's analyze the results more in detail:
Once again the first algorithm I tried out is a Support Vector Machine classification.
It's immediatly clear how the results are now much closer to the ones of the papers, raising the accuracy from the 75% of the first approach up to 92% here.
Since I saw that in the previous approach using KNN granted some improvements I also tried it out here, and while there are improvements from the KNN results from the first approach, 
there aren't significant improvements with respect to the SVM results we just discussed. The accuracy is of around 92% also in this case.
Lastly since I observed a big improvement using a Random Forest classification with the first approach, I tried it also with this new feature set and the results are astounding.
This algorithm is able to guess with a 99% accuracy on the test set, making these results the best they could be."""


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