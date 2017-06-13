import samples
import util
from game import Agent, Directions, Actions
from sklearn import tree
import samples
import assignment2
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from game import Actions
import pickle

class ClassifierAgent(Agent):
    def __init__(self, training_set=None):
        super().__init__()
        training_data = samples.read_dataset(training_set)
        self.train(training_data)

    def train(self, training_data):
        """
        Train the agent using data set training_data.
        The data set comprises a dict with keys 'states', which maps to a list of game states, and 
        'actions', which maps to a list of actions corresponding to the list of game states.
        Store the resulting classifier in self.classifier
        """
        X, y, legal, Xt, yt, legalt, enc_vec, enc_lab = prepare_dataset(training_data, assignment2.extract_action_features)

        self.X = X
        self.classifier = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
        self.classifier.fit(X, y)

    def getAction(self, state):
        """
        getAction chooses the action predicted by the classifier that was learned from training_set.
        The classifier is accessible as self.classifier
        
        Just like in the practical project, getAction takes a GameState and returns
        some X in the set Directions.{North, South, West, East, Stop}.
        
        The returned action must be one of state.getLegalActions().
        """
        legal = state.getLegalActions()
        pred = self.classifier.predict(self.X)

        return Directions.STOP
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
    features_train = [extract_features(state, feature_function) for state in name['states']]
    X_train = vectorizer.fit_transform(features_train)
    y_train = label_encoder.transform(name['actions'])
    legalActions = [label_encoder.transform(state.getLegalPacmanActions()) for state in name['states']]

    features_test = [extract_features(state, feature_function) for state in name['states']]
    X_test = vectorizer.transform(features_test)
    y_test = label_encoder.transform(name['actions'])
    legalActions_test = [label_encoder.transform(state.getLegalPacmanActions()) for state in name['states']]

    return X_train, y_train, legalActions, X_test, y_test, legalActions_test, vectorizer, label_encoder