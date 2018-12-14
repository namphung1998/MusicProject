from sklearn import svm
import numpy as np
from multiprocessing import Process

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

genres_dict = {i: genres[i] for i in range(len(genres))}


def load_data(test_size=0):
    train_data = []

    for genre in genres:
        for i in range(1):
            with open("mfcc_data/{}{}.csv".format(genre, i)) as data_file:
                data = [float(x.strip()) for x in data_file.readlines()[:4]]
                data.append(genre)

                train_data.append(data)

    train = np.array(train_data)
    np.random.shuffle(train)
    print(train)

    features = np.array(train[:, :-1], dtype=np.float64)
    labels = train[:, -1]

    if test_size > 0:
        return np.array(train[test_size:, :-1], dtype=np.float64), train[test_size:, -1], np.array(train[:test_size, :-1], dtype=np.float64), train[:test_size, -1]

    return np.array(train[:, :-1], dtype=np.float64), train[:, -1]


if __name__ == '__main__':
    features, labels, testFeatures, testLabels = load_data(200)

    # print(features[1, 1])

    #
    clf = svm.SVC()
    clf.fit(features, labels)

    predictions = clf.predict(testFeatures)
    incorrect = ((predictions == testLabels) == False).sum()

    print('incorrect percentage: ' + (len(predictions) - incorrect) / len(predictions))
