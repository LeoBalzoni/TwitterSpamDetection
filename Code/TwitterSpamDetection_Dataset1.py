"""This is my first try at replicating the experiments and results I saw in the papers I studied.
The topic of discussion is the detection of spam tweets. Unfortunately the datasets used from the authors of the papers are nowhere to be found, so I had to search for a different one.
The one I ended up using is available at http://nsclab.org/nsclab/resources/ and is part of the dataset used for the paper "6 Million Spam Tweets:
A Large Ground Truth for Timely Twitter Spam Detection" available at https://ieeexplore.ieee.org/document/7249453.
After some preprocessing the dataset is now made of tuples of with the following form:
[account_age, no_follower, no_following, no_userfavourites, no_lists, no_tweets, no_retweets, no_hashtag, no_usermention, no_urls, no_char, no_digits, label]
As one can see the features are quite different from the ones used in our reference paper, lets now see what results these features are able to provide."""

# Various imports
from sklearn import model_selection
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Setting the path of the dataset
dataset_location = './Datasets/dataset_1.csv'

# Reading and splitting the dataset in train and test sets
def read_dataset(filename):
    data = pd.read_csv(filename)
    print('____________________________________________\n_______________ Dataset Info _______________\n____________________________________________')
    print('Number of features: {}'.format(data.shape[1]))
    print('Number of examples: {}'.format(data.shape[0]))
    print(data['label'].value_counts())
    print('\nDataset read:')
    print(data.head())
    return model_selection.train_test_split(data.drop(['label'], axis=1), data["label"], train_size=0.8)

Xtrain, Xtest, Ytrain, Ytest = read_dataset(dataset_location)
print('\n')

"""What follows now are 3 blocks of code, in each of them I train and test a different machine learning algorithm to see which one works the best given the aforementioned feature set selected for the tweets.

The first algorithm I tried is Support Vector Machine, I chose this one first since it's the one used in most of the papers.
What I found out is that with this dataset and features the results are good but not optimal, it seems like the feature set I am using here is too different from the one used in the papers I saw 
to get comparable results. After some hyperparameters tuning the best result I were able to get is 75% accuracy.
While reading more about spam detection on twitter I found out that often KNN can provide improvements to the results, so my second approach was using the KNN algorithm provided from scikit-learn. 
The results in fact improved (even if marginally) raising the accuracy to 79%
Lastly what worked out best in the "6 Million Spam Tweets: A Large Ground Truth for Timely Twitter Spam Detection" paper was using a Random Forest classification,
an ensemble learning algorithm based on bagging that often provides great results. Also in my case it was the best, granting an accuracy of 87%.
"""

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

"""The results I were able to obtain from this dataset are definitely interesting but I think that it differs a bit too much from the ones referenced in the papers I studied.

What I decided to do is to find another dataset, made of actual tweets and not directly in the form of a feature array for each tweet, 
so that I can create a feature set by hand, making it as similar as possible to the one used in the referenced papers, to see how the results change when using something almost identical to the one they used."""