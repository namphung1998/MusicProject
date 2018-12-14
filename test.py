import numpy as np
from sklearn.model_selection import train_test_split

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_data():
    features = []
    labels = []

    for genre in genres:
        for i in range(10):
            with open("mfcc_data/{}{}.csv".format(genre, i), "r") as data_file:
                data = [float(x.strip()) for x in data_file.readlines()]
                features.append(data)
                labels.append(genre)

    return features, labels

features, labels = load_data()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=13)

from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
