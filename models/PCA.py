from utils.utils import make_window, flatten, ROC
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np

def pca_test(pca, inputs):
    scores = []
    for series in inputs:
        series = np.array(series)
        
        series_pca = pca.transform(series)
        recon = pca.inverse_transform(series_pca)
        error = ((series - recon)**2).mean(axis=1)        
        scores.append(error)   
        
    return scores

def pca_experiment(train_data, test_data, test_labels, args):
    model_name = 'pca'
    print("Evaluating " + model_name)
    test_data_w, test_labels_w = make_window(test_data, args['seq_len'], test_labels)
    
    pca = PCA(n_components="mle")
    _ = pca.fit_transform(train_data)      
    scores = pca_test(pca, test_data_w) 

    scores = flatten(scores)
    labels = flatten(test_labels_w)

    thresh, auc = ROC(labels, scores)
    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}
    
    return metrics_dict