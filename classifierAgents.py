import samples
import util
from game import Agent, Directions
from sklearn import tree



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
        print(training_data)
        classifier = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=5)


    def getAction(self, state):
        """
        getAction chooses the action predicted by the classifier that was learned from training_set.
        The classifier is accessible as self.classifier
        
        Just like in the practical project, getAction takes a GameState and returns
        some X in the set Directions.{North, South, West, East, Stop}.
        
        The returned action must be one of state.getLegalActions().
        """

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        return Directions.STOP
