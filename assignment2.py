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
from itertools import product
from sklearn import tree

#
# QUESTION 1
#

def nearestFood(pos, food, walls):
    spot = [(pos[0], pos[1], 0)]
    expanded = set()
    while spot:
        pos_x, pos_y, dist = spot.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if food[pos_x][pos_y]:
            return dist
        neighbours = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in neighbours:
            spot.append((nbr_x, nbr_y, dist+1))
    return None


def extract_action_features(gameState, action):
    """
    This function extracts features reflecting the effects of applying action to gameState.

    You should return a dict object of features where keys are the feature names and 
    the values the feature values. The feature names must be the same for all actions.

    All values must be of primitive type (boolean, int, float) that scikit-learn can handle.
    """
    features = util.Counter()
    successorState = gameState.generateSuccessor(0, action)
    features['score'] = successorState.getScore()  # keep this?

    food = gameState.getFood()
    walls = gameState.getWalls()
    ghosts = gameState.getGhostPositions()

    x, y = gameState.getPacmanPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    distance = nearestFood((next_x, next_y), food, walls)
    if distance is not None:
        features["nearest-food"] = float(distance) / (walls.width * walls.height)
    
    closest = 999999
    index = 0
    count = 0
    currentTemp = 0

    for ghost in ghosts:
        count += 1
        currentTemp = util.manhattanDistance( ghost, gameState.getPacmanPosition() )

    features['FoodLeft'] = successorState.getNumFood()
            
    temp = util.manhattanDistance(successorState.getGhostPosition(count), successorState.getPacmanPosition() )
    
    if temp > currentTemp:
        features['Moved away from Closest ghost'] = True
    else:
        features['Moved away from Closest ghost'] = False
        
    features['Distance-to-closest-ghost'] = temp
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
    clf = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)
    clf.fit(X, y) # training data

    #
    # QUESTION 3
    #
    
    # 110171506

    # TRAINING data
    print("*" * 40, "DECISION TREE CLASSIFIER", "*" * 40)
    pred = clf.predict(X)
    print('Correctly predicted on TRAINING set: {}, errors: {}'.format(sum(y==pred), sum(y!=pred)))
    print(classification_report(y, pred))
    print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(y, pred)) , '\n')
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(y,pred,labels=clf.classes_))


    # TEST data
    pred = clf.predict(Xt)
    print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==pred), sum(yt!=pred)))
    print(classification_report(yt, pred))
    print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(yt,pred,labels=clf.classes_), "\n")
        
    
    #
    # QUESTION 4
    #

    # Multi-Layer-Perceptron
    mlp = MLPClassifier(max_iter=5000)
    mlp.fit(X, y)
    
    # TRAINING data
    print("*" * 40, "Multi-Layer Perceptron CLASSIFIER", "*" * 40)
    pred = mlp.predict(X)
    print('Correctly predicted on TRAINING set with MLP classifier: {}, errors: {}'.format(sum(y==pred), sum(y!=pred)))
    print(classification_report(y, pred))
    print('Accuracy on TRAINING set with MLP classifier: {:.2f}'.format(accuracy_score(y, pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(y,pred,labels=mlp.classes_))
    
    # TEST data
    pred = mlp.predict(Xt)
    print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==pred), sum(yt!=pred)))
    print(classification_report(yt, pred))
    print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(yt,pred,labels=mlp.classes_), "\n")

    #
    # QUESTION 5
    #  
    parameters = {
                  'hidden_layer_sizes': list(product([1,2,3,4,5,7,8,9,10],[1,2,3,4,5])),
                  'max_iter': [5000]
          }
       
    opt_mlp = GridSearchCV(MLPClassifier(), parameters, n_jobs=8)
    opt_mlp.fit(X, y)
       
    print("Best params for layers set")
    print()
    print(opt_mlp.best_params_)
    print()
    y_pred = opt_mlp.predict(Xt)
    print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(yt,y_pred)))
     

    #
    # QUESTION 6: see file classifierAgents.py
    #


if __name__ == '__main__':
    for agent in ['leftturn', 'random', 'stop', 'suicide']: # pick agents you need to consider
        print("AGENT", agent, "\n" + "=" * 40, "\n")
        do_agent(agent)
        print("\n\n\n")
