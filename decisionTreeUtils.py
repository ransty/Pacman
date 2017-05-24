# decisionTreeUtils.py
#
# Author: Wolfgang Mayer
#
# DO NOT MODIFY THIS FILE

from sklearn.tree import _tree
import numpy as np

def extract_decisiontree_rules(classifier, feature_names, class_names):
    """
    Print a textual representation of the decision rules in a decision tree classifier.
    """
    tree_ = classifier.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            classno = classifier.classes_[np.argmax(tree_.value[node])]
            label = class_names[classno]
            value_text = str(np.around(tree_.value[node])).replace("  "," ")
            print("{}classify as {} ({})  (samples={}, counts={})".format(indent, label, classno, tree_.n_node_samples[node], value_text))

    recurse(0, 0)


def decisiontree_notebook(classifier, feature_names, action_labels):
    """
    Plot a graphical representation of a decision tree classifier in a Jupyter notebook.
    Requires the graphviz graph drawing tool.
    """
    from sklearn import tree
    import pydotplus
    from IPython.display import Image
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=feature_names,
                                    class_names=action_labels,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
