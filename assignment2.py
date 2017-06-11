#
# COMP2019 ASSIGNMENT 2 2017
#
import util
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
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y) # training data

    #
    # QUESTION 3
    #
    
    # 110171506

    # TRAINING data
    y_pred = clf.predict(X)
    print('Correctly predicted on TRAINING set: {}, errors: {}'.format(sum(y==y_pred), sum(y!=y_pred)))
    print(classification_report(y, y_pred))
    print('Accuracy on TRAINING set: {:.2f}'.format(accuracy_score(y, y_pred)) , '\n')
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(y,y_pred,labels=clf.classes_))


    # TEST data
    #y_pred = clf.predict(Xt)
    print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==y_pred), sum(yt!=y_pred)))
    print(classification_report(yt, y_pred))
    print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, y_pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in clf.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(yt,y_pred,labels=clf.classes_), "\n")
        
    
    #
    # QUESTION 4
    #

    # Multi-Layer-Perceptron
    mlp = MLPClassifier(max_iter=5000)
    mlp.fit(X, y)
    
    # TRAINING data
    y_pred = mlp.predict(X)
    print('Correctly predicted on TRAINING set with MLP classifier: {}, errors: {}'.format(sum(y==y_pred), sum(y!=y_pred)))
    print(classification_report(y, y_pred))
    print('Accuracy on TRAINING set with MLP classifier: {:.2f}'.format(accuracy_score(y, y_pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(y,y_pred,labels=mlp.classes_))
    
    # TEST data
    y_pred = mlp.predict(Xt)
    print('Correctly predicted on TEST set: {}, errors: {}'.format(sum(yt==y_pred), sum(yt!=y_pred)))
    print(classification_report(yt, y_pred))
    print('Accuracy on TEST set: {:.2f}'.format(accuracy_score(yt, y_pred)))
    print("Confusion Matrix:\n "," ".join(["{:3d}".format(d) for d in mlp.classes_]),"<-- PREDICTED LABEL")
    print(confusion_matrix(yt,y_pred,labels=mlp.classes_), "\n")

    #
    # QUESTION 5
    #

    parameters = {'hidden_layer_sizes': np.arrange()}
    

    #
    # QUESTION 6: see file classifierAgents.py
    #


if __name__ == '__main__':
    for agent in ['leftturn', 'random', 'stop', 'suicide']: # pick agents you need to consider
        print("AGENT", agent, "\n" + "=" * 40, "\n")
        do_agent(agent)
        print("\n\n\n")
