#
# COMP2019 ASSIGNMENT 2 2017
#
import util
import numpy as np
from game import Actions
from samples import prepare_dataset
from sklearn.neural_network import MLPClassifier
from decisionTreeUtils import extract_decisiontree_rules
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals.six import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import tree

#
# QUESTION 1
#

def closestFood(pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


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
    #util.raiseNotDefined()

    # extract the grid of food and wall locations and get the ghost locations
    food = gameState.getFood()
    walls = gameState.getWalls()
    ghosts = gameState.getGhostPositions()

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    x, y = gameState.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
        features["eats-food"] = 1.0

    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
        # make the distance a number less than one otherwise the update
        # will diverge wildly
        features["closest-food"] = float(dist) / (walls.width * walls.height)
    features.divideAll(10.0)
    #print(features)
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
    #clf = tree.DecisionTreeClassifier()
    #clf.fit(X, y) # training data

    #
    # QUESTION 3
    #
    
    # 110171506

    # TRAINING data
    #y_pred = clf.predict(X)
    #print('Correctly predicted on TRAINING set: {}, errors: {}'.format(sum(y==y_pred), sum(y!=y_pred)))
    #print(classification_report(y, y_pred))
    #print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(y, y_pred)) , '\n')
    #print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    #print(confusion_matrix(y,y_pred,labels=clf.classes_))


    # TEST data
    #y_pred = clf.predict(Xt)
    #print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==y_pred), sum(yt!=y_pred)))
    #print(classification_report(yt, y_pred))
    #print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, y_pred)))
    #print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    #print(confusion_matrix(yt,y_pred,labels=clf.classes_), "\n")
        
    
    #
    # QUESTION 4
    #

    # Multi-Layer-Perceptron
    #mlp = MLPClassifier(max_iter=5000)
    #mlp.fit(X, y)
    
    # TRAINING data
    #y_pred = mlp.predict(X)
    #print('Correctly predicted on TRAINING set with MLP classifier: {}, errors: {}'.format(sum(y==y_pred), sum(y!=y_pred)))
    #print(classification_report(y, y_pred))
    #print('Accuracy on TRAINING set with MLP classifier: {:.2f}'.format(accuracy_score(y, y_pred)))
    #print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    #print(confusion_matrix(y,y_pred,labels=mlp.classes_))
    
    # TEST data
    #y_pred = mlp.predict(Xt)
    #print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==y_pred), sum(yt!=y_pred)))
    #print(classification_report(yt, y_pred))
    #print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, y_pred)))
    #print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    #print(confusion_matrix(yt,y_pred,labels=mlp.classes_), "\n")

    #
    # QUESTION 5
    #
    
    n1 = np.arange(1,11,1)
    n2 = np.arange(1,6,1)
    
    parameters = {
                'hidden_layer_sizes': [(1,1), (1,2), (1,3), (1,4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2,5),(3,1),(3,2),(3,3),(3,4),(3,5),(4,1),(4,2),(4,3),(4,4),(4,5),(5,1),(5,2),(5,3),(5,4),(5,5),(6,1),(6,2),(6,3),(6,4),(6,5),(7,1),(7,2),(7,3),(7,4),(7,5),(8,1),(8,2),(8,3),(8,4),(8,5),(9,1),(9,2),(9,3),(9,4),(9,5),(10,1),(10,2),(10,3),(10,4),(10,5)],
                'max_iter': [5000]
                }
    
    opt_mlp = GridSearchCV(MLPClassifier(), parameters)
    opt_mlp.fit(X, y)
    print("Best params for layers set")
    print()
    print(opt_mlp.best_params_)
    print()
    y_pred = opt_mlp.predict(X)
    print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(y,y_pred)))
    

    #
    # QUESTION 6: see file classifierAgents.py
    #


if __name__ == '__main__':
    for agent in ['leftturn', 'random', 'stop', 'suicide']: # pick agents you need to consider
        print("AGENT", agent, "\n" + "=" * 40, "\n")
        do_agent(agent)
        print("\n\n\n")
