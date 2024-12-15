from sklearn.naive_bayes import MultinomialNB

def train_naive_bayes_model(train_vectors, train_labels):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(train_vectors, train_labels)
    return nb_classifier
