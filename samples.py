# samples.py
#
# Author: Wolfgang Mayer
#
# DO NOT MODIFY THIS FILE

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from game import Actions


#
# functions for loading data sets
#

def read_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


#
# feature extraction for Pacman
#
ALL_ACTIONS = sorted(Actions._directions.keys())


def extract_features(gameState, feature_function):
    """
    This function extracts features for gameState using feature_function.

    The returned dict includes separate columns for each feature for each action.
    """
    # extract features for legal actions
    features = {}
    legal_actions = gameState.getLegalPacmanActions()
    for action in legal_actions:
        features[action] = feature_function(gameState, action)

    # add dummy features for illegal actions
    any_feature_dict = list(features.values())[0]
    dummy_features = {k: type(v)() for k, v in any_feature_dict.items()}
    for action in ALL_ACTIONS:
        if action in legal_actions:
            continue
        features[action] = dummy_features

    # convert to flat dict for DictVectorizer
    all_features = {}
    for a, fs in features.items():
        for k, v in fs.items():
            all_features[a + ':' + k] = v

    return all_features


def prepare_dataset(name, feature_function):
    """
    Load data set 'name'_{train,test}.pkl, transform states to features using feature_function,
    and encode target action labels to numeric values.
    """
    vectorizer = DictVectorizer()
    label_encoder = LabelEncoder()
    label_encoder.fit(ALL_ACTIONS)
    train_data = read_dataset('data/%s_train.pkl' % name)
    features_train = [extract_features(state, feature_function) for state in train_data['states']]
    X_train = vectorizer.fit_transform(features_train)
    y_train = label_encoder.transform(train_data['actions'])
    legalActions = [label_encoder.transform(state.getLegalPacmanActions()) for state in train_data['states']]

    test_data = read_dataset('data/%s_test.pkl' % name)
    features_test = [extract_features(state, feature_function) for state in test_data['states']]
    X_test = vectorizer.transform(features_test)
    y_test = label_encoder.transform(test_data['actions'])
    legalActions_test = [label_encoder.transform(state.getLegalPacmanActions()) for state in test_data['states']]

    return X_train, y_train, legalActions, X_test, y_test, legalActions_test, vectorizer, label_encoder
