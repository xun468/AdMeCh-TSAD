from utils.utils import make_window, flatten, ROC
from sklearn.neighbors import NearestNeighbors
import numpy as np


def knn_experiment(train_data, test_data, test_labels, args):
    model_name = "knn"
    print("Evaluating " + model_name)
    
    test_data_w, test_labels_w = make_window(test_data, args['seq_len'], test_labels)
    
    knn = NearestNeighbors(n_neighbors = 10)
    knn.fit(train_data)
    
    distances, _ = knn.kneighbors(test_data)
    d = np.max(distances, 1)
    d_window = np.array(make_window(d, args['seq_len']))

    scores = flatten(d_window)
    labels = flatten(test_labels_w)

    thresh, auc = ROC(labels, scores)
    
    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}
    
    return metrics_dict

#Less efficient method 
# def knn_experiment(train_data, test_data, test_labels, args):
#     model_name = "knn"
#     knn = NearestNeighbors(n_neighbors = 10)
#     knn.fit(train_data)    

#     test_data_w, test_labels_w = make_window(test_data, args['seq_len'], test_labels)

#     distances = [] 
#     for window in test_data_w:
#         d, _ = model.kneighbors(window)
#         distances.append(d)   

#     scores = np.array(distances)
#     scores = np.max(scores,2)

#     scores = flatten(scores)
#     labels = flatten(test_labels_w)
    
#     thresh, auc = ROC(labels, scores)