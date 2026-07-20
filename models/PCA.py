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
    test_data_w, labels = make_window(test_data, args['seq_len'], test_labels)
    
    pca = PCA()
    _ = pca.fit_transform(train_data)      
    scores = pca_test(pca, test_data_w) 

    return flatten(labels), flatten(scores)