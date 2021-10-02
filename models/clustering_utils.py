import numpy as np
from tqdm import tqdm_notebook
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering, OPTICS
from sklearn_extra.cluster import KMedoids
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from scipy.spatial.distance import braycurtis, pdist, squareform
from IPython.core.debugger import set_trace
from IPython.display import clear_output
from utils import filter_paths, unpack_data, js, get_neigh_perc, entropy, clustering, plot_clustering
from numba import cuda, jit, njit, prange, vectorize
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from umap import UMAP
from sklearn.manifold import TSNE
 
from hdbscan import validity_index as DBCV
from hdbscan import HDBSCAN


class SpectralClustering_prec:
    def __init__(self, *args, **kwargs):
        self.method = SpectralClustering(*args, **kwargs, affinity='precomputed', random_state=42)
    def fit_predict(self, X):
        '''
        X - pairwise_distance matrix
        '''
        A = np.exp(-self.method.gamma * np.abs(X))
        return self.method.fit_predict(A)    


def clustering_by_methods(data, methods_dict, precomputed=False, d=None):
    results = []
    for method_name, [method_class, param_range] in methods_dict.items():
        
        cluster_metrics, cluster_results = clustering(data, 
                                                        method_class, 
                                                        param_range,
                                                        precomputed,
                                                        d=d)
        results.append([cluster_metrics, cluster_results])
    return results


def clustering(dataset, method_class, param_dict, precomputed=False, verbose=False, d=None):
    '''
    dataset: data
    method_class: class constructor for clustering
    param_range: params for class constructor, {'n_clusters':[2,3,4,5]}
    precomputed: datasets_dict will be treated as dict of distance matrices
    '''
    
    cluster_metrics = defaultdict(list) # clustering metrics
    cluster_results = defaultdict(list) # partition results
    
    for p in ParameterGrid(param_dict):
        
        metrics = {}
        results = {}
        
        method = method_class(**p)
        pred = method.fit_predict(dataset)
        
        if max(pred) > 0: # at least 2 clusters, -1:outliers, 0-first cluster, 1-second cluster ...
            
            # consider only non-noise clusters
            non_noise_mask = pred != -1
            
            # if too much noise - continue
            if sum(non_noise_mask)/len(non_noise_mask) < 0.6:
                if verbose:
                    print(f'Too much nose! Skipping for p={p}, {method_class.__name__}')
                continue
            
            # filter-out small clusters
            abundance_mask = np.zeros(len(pred), dtype=bool)
            for k in np.unique(pred[non_noise_mask]): 
                # more than 1% of the data
                if sum(pred==k)/sum(non_noise_mask) > CLUSTER_PERCENTAGE_THRESHOLD:
                    abundance_mask[pred==k] = True
                elif verbose:
                    print(f'Small cluster {k} with {sum(pred==k)} items removed')
                    
            mask = abundance_mask
            results['mask'] = abundance_mask
            # all data was separated into small clumps
            data_used = mask.sum()/len(pred)
            
            if data_used < 0.5:
                if verbose:
                    print(f'Too much data were removed!')
                continue
            
            # no outliers left
            assert (pred[mask] >= 0).all()
            # all clusters contain more that 1% of data
            assert Counter(pred[mask]).most_common()[-1][1]/sum(non_noise_mask) > CLUSTER_PERCENTAGE_THRESHOLD
            
            unique_clusters = np.unique(pred[mask])
            n = len(unique_clusters) # number of clusters
            
            # re-numerated unique_clusters labels
            labels = np.zeros((sum(mask)), dtype=int)
            for i,k in enumerate(unique_clusters):
                labels[pred[mask]==k] = i
            results['labels'] = labels
            
            if n > 1:
                if precomputed:
                    metrics['dbind'] = davies_bouldin_score_precomputed(dataset[mask][:,mask], labels)
                    metrics['silh'] = silhouette_score(dataset[mask][:,mask], labels, metric='precomputed')
                    metrics['dbcv'] = DBCV(dataset[mask][:,mask], labels, metric='precomputed', d=d)
#                     metrics['ch'] = calinski_harabasz_score_precomputed(dataset[mask][:,mask], labels)
                    metrics['ps'] = prediction_strength_CV_precomputed(dataset[mask][:,mask], method)

                else:
                    metrics['dbind'] = davies_bouldin_score(dataset[mask], labels)
                    metrics['silh'] = silhouette_score(dataset[mask], labels)
                    metrics['dbcv'] = DBCV(dataset[mask], labels)
#                     metrics['ch'] = calinski_harabasz_score(dataset[mask], labels)
                    metrics['ps'] = prediction_strength_CV(dataset[mask], method) 

                # data mass distribution
                cl_dist = np.ones(n)
                for i,cl_number in enumerate(np.unique(labels)):
                    cl_dist[i] = sum(labels == cl_number)/sum(mask)

                metrics['noise_ratio'] = sum(pred == -1)/len(pred)
                metrics['entropy'] = entropy(cl_dist)
                metrics['data_used'] = data_used
                metrics['dist'] = cl_dist

                # for each [n] there may be more than 1 partition!
                cluster_metrics[n].append(metrics) 
                cluster_results[n].append(results)
                
            else:
                if verbose:
                    print(f'No clusters found for p={p}, {method_class.__name__}')
                continue

        # no clusters found
        else:
            if verbose:
                print(f'No clusters found for p={p}, {method_class.__name__}')
            continue
                
    return cluster_metrics, cluster_results


def calinski_harabasz_score_precomputed(X, labels):
    raise NotImplementedError()

def davies_bouldin_score_precomputed(D, labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = D.shape
    n_labels = len(le.classes_)
    intra_dists = np.zeros(n_labels)
    centroids = []
    
    for k in range(n_labels):
        mask_k = labels==k
        cluster_k = D[mask_k][:,mask_k] #_safe_indexing(X, labels == k)
        centroid_index = np.argmin(cluster_k.mean(1))
        intra_dists[k] = cluster_k[centroid_index].mean()
        centroid_index = np.arange(len(mask_k))[mask_k][centroid_index]
        centroids.append(centroid_index)
        
    centroid_distances = D[centroids][:,centroids] # kxk matrix

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists # kxk matrix
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def prediction_strength(y_pred, y_test):
    '''
    For each pair of test observations 
    that are assigned to the same test cluster, 
    we determine whether they are also assigned 
    to the same cluster based on the train desicion boundary.
    '''
    
    test_clusters = np.unique(y_test)
    counts = []
    
    for k in test_clusters:
        mask = y_test == k
        n_k = mask.sum()

        c = Counter(y_pred[mask])
        
        count = c.most_common(1)[0][1]
        count = count * (count - 1) # number of pairs that fall in the same cluster given train decision function
        count /= (n_k * (n_k - 1)) # divided by the total number of pairs in cluster

        counts.append(count)
    return min(counts)


def prediction_strength_CV_precomputed(D, method, n_splits = 3, plot=False, axes=None):
    
    ps_s = []
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    for i,(train_index, test_index) in enumerate(kfold.split(D)):
        
        # getting clustering from train data
        D_train = D[train_index][:,train_index]
        y_train = method.fit_predict(D_train) 
        
        # getting clustering from test data
        D_test = D[test_index][:,test_index]
        y_test = method.fit_predict(D_test)
        
        D_ = D[test_index][:,train_index]
        y_pred = y_train[np.argsort(D_, axis=1)[:,:4]]
        y_pred = mode(y_pred, axis=1).mode.flatten()
        
        ps = prediction_strength(y_pred, y_test) # y_train, y_test 
        ps_s.append(ps)
        
        if plot:
            axes[i].scatter(X_test[:,0], X_test[:,1], c=y_test) # test data with train boundaries
            axes[i].set_title(f'PS: {ps}', fontsize=20)
            plot_decision_regions(X_test,clf, axes[i])
        
    return np.mean(ps_s)


def prediction_strength_CV(X, method, n_splits = 3, plot=False, axes=None, precomputed=False):
    
    ps_s = []
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    for i,(train_index, test_index) in enumerate(kfold.split(X)):
        
        # getting clustering from train data
        X_train = X[train_index]
        y_train = method.fit_predict(X_train) 
        
        # getting clustering from test data
        X_test = X[test_index]
        y_test = method.fit_predict(X_test)

        clf = KNeighborsClassifier(weights='distance', p=2) 
        clf.fit(X_train, y_train) # fit decision regions from train data
        y_pred = clf.predict(X_test) # predict test clustering
        
        ps = prediction_strength(y_pred, y_test) # y_train, y_test 
        ps_s.append(ps)
        
        if plot:
            axes[i].scatter(X_test[:,0], X_test[:,1], c=y_test) # test data with train boundaries
            axes[i].set_title(f'PS: {ps}', fontsize=20)
            plot_decision_regions(X_test,clf, axes[i])
        
    return np.mean(ps_s)