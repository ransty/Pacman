#
# COMP2019 ASSIGNMENT 2 2017
#
import util
from samples import prepare_dataset
from decisionTreeUtils import extract_decisiontree_rules

#
# QUESTION 1
#

def extract_action_features(gameState, action):
    """
    This function extracts features reflecting the effects of applying action to gameState.

    You should return a dict object of features where keys are the feature names and 
    the values the feature values. The feature names must be the same for all actions.

    All values must be of primitive type (boolean, int, float) that scikit-learn can handle.
    """
    features = dict()
    successorState = gameState.generateSuccessor(0, action)
    features['score'] = successorState.getScore()  # keep this?
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
    return features


def do_agent(agent):
    # load data sets and convert to feature matrices and target vectors for sklearn
    # X will denote the feature matrix and y the target vector for training; Xt,yt are for testing.
    # legal and legalt are lists of lists of legal moves in the respective states encoded by X and Xt.
    # env_vec and enc_lab are the encoders used to transform the feature dicts and list to sklearn vectors
    X, y, legal, Xt, yt, legalt, enc_vec, enc_lab = prepare_dataset(agent, extract_action_features)

    #
    # QUESTION 2
    #

    "*** YOUR CODE HERE ***"

    #
    # QUESTION 3
    #

    "*** YOUR CODE HERE ***"

    #
    # QUESTION 4
    #

    "*** YOUR CODE HERE ***"

    #
    # QUESTION 5
    #

    "*** YOUR CODE HERE ***"

    #
    # QUESTION 6: see file classifierAgents.py
    #


if __name__ == '__main__':
    for agent in ['food', 'leftturn', 'random', 'stop', 'suicide']: # pick agents you need to consider
        print("AGENT", agent, "\n" + "=" * 40, "\n")
        do_agent(agent)
        print("\n\n\n")
