#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:17:41 2022

@author: edo
"""

import networkx as nx
from tqdm import tqdm
import pickle
import numpy as np
import scipy.linalg

#from sklearn.manifold import spectral_embedding
from sklearn.cluster import SpectralClustering, KMeans
#from sklearn.cluster import AffinityPropagation
#from sklearn_extra.cluster import KMedoids
from itertools import takewhile

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from networkx.algorithms.community import greedy_modularity_communities, girvan_newman
from networkx import edge_betweenness_centrality as betweenness

from partitioning import sensors_choice, betweenness_sensors, medoid_sensors, get_clusters, get_clusters_aff,get_clusters_hierarchical, get_clusters_markov, corr_partitioning_medoid #, get_clusters_modularity


###############################################################################
###############################################################################
###############################################################################

# PATH HAS INFO ABOUT THE TYPE OF NET AND THE NUMBER OF SOURCES

path = ''
kind = ''

###############################################################################
###############################################################################
###############################################################################


# =============================================================================
# #CHOOSE WHICH K
# =============================================================================

min_k = 20
max_k = 120
step_k = 2


with open (path + 'networks' + kind, 'rb') as f:
    networks_data = pickle.load(f)


#corr_list length is the number of networks
with open (path + 'correlations' + kind, 'rb') as f:
    net_corr_lists = pickle.load(f)



partitionings_data = []

max_iter = 300
atol = 1e-6

#Each corr_list is the list of all the corr for a single network
corr_num = []

num_of_net = 10
for net_num,net_data, corr_list in zip(range(0,num_of_net),networks_data[:num_of_net], net_corr_lists[:num_of_net]):
    
    # Unfold networks data
    G, Qdict, _, _ = net_data
    
    
    # Networks data needed
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    norm_L = nx.linalg.normalized_laplacian_matrix(G)
    norm_L = norm_L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    _, norm_eigvecs = scipy.linalg.eigh(norm_L) #automatically sorted in ascending order, and eigvecs are orthonormal, even in the degenerate case
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    _, eigvecs = scipy.linalg.eigh(L) #automatically sorted in ascending order, and eigvecs are orthonormal, even in the degenerate case

    # Demands
    s = np.asarray( list(nx.get_node_attributes(G,'current').values()) )
    #p = np.asarray( list(nx.get_node_attributes(G,'pressure').values()) )
    # NOdes number
    N = len(s)


    # To avoid sorting all the times for all the k's: QUI LA SORTI CON ABS!!!
    sorted_corr_list = []
    for corr in corr_list:
        np.fill_diagonal(corr,0)
        sorted_corr = np.dstack(np.unravel_index(np.argsort(np.abs(corr).ravel()), (N, N)))[0][::-1] #[N::][::2]
        np.fill_diagonal(corr,1)
        
        sorted_corr_list.append(sorted_corr)
    
    k_th_list = []
    for corr, sorted_corr in zip(corr_list, sorted_corr_list):
        k_th_dic = {}
        for th in np.linspace(1e-5,np.max(corr),10):
            labels,sensors = corr_partitioning_medoid(np.abs(corr),th,sorted_corr)
            k_th_dic[len(sensors)] = th
        k_th_list.append( k_th_dic )
    
    def find_th_hints(myArr, k):
        k_up = myArr[myArr >= k].min()  
        k_down = myArr[myArr <= k].max()
        return k_down, k_up   
    

    # =============================================================================
    #         GIRVAN-NEWMAN
    # =============================================================================
         
    def most_central_edge(G):
        centrality = betweenness(G, weight="weight")
        return max(centrality, key=centrality.get)
   
    comp = girvan_newman(G, most_valuable_edge=most_central_edge)
    
    limited = takewhile(lambda c: len(c) <= max_k, comp)
    
    # In limited there should be all numbers between min_k and max_k
    gn_partitionings = dict()
    for communities in limited:
        if len(communities) in list(range(min_k,max_k,step_k)):
          gn_partitionings[ len(communities) ] = communities


    # Initialize dic for this network
    partitionings_list = {k:[] for k in range(min_k,max_k,step_k)}

    for k in tqdm(partitionings_list.keys(), total=len(partitionings_list.keys()), position=0, leave=True):
        
        
        ################################################################################################################################
        ############################################ PARTITIONING OF CORRELATIONS ###################################################################
        ################################################################################################################################

        
        # =============================================================================
        #         OUR CORRELATIONS PARTITIONGS
        # =============================================================================
        
            
        # for corr in corr_list:
            
        #     labels, sensors = get_clusters('medoid1', np.abs(corr), k, 0, np.max(corr), max_iter, atol=atol)
        #     partitionings_list[k].append((labels,sensors))
    
    
        for corr, sorted_corr, k_th_dic in zip(corr_list, sorted_corr_list, k_th_list):
            
            myArr = np.asarray(list(k_th_dic.keys()))
            if k >= myArr[0]:
                k_down, k_up = find_th_hints(myArr, k)
            
                th_down = k_th_dic[k_down]
                th_up = k_th_dic[k_up]
                
                labels, sensors = get_clusters('medoid', np.abs(corr), k, th_down, th_up, max_iter, atol=atol, sorted_corr=sorted_corr)
                if type(sensors)!=float:
                    if len(sensors) != k:
                        print('Non k medoid')
                partitionings_list[k].append((labels,sensors))
                
            else:
                partitionings_list[k].append((np.nan,np.nan))
    
            
        for corr, sorted_corr in zip(corr_list, sorted_corr_list):
            
            labels, sensors = get_clusters('affcomm', np.abs(corr), k, 0, np.max(corr), max_iter, atol=atol, sorted_corr=sorted_corr)
            if type(sensors)!=float:
                if len(sensors) != k:
                    print('Non affcomm')
            partitionings_list[k].append((labels,sensors))
    

        for corr_num,corr in enumerate(corr_list):
            
            dissimilarity = np.copy( 1 - np.abs(corr) )
            dissimilarity = (dissimilarity + dissimilarity.T)/2
            dissimilarity[np.isclose(dissimilarity,0)]=0
            
            hierarchy = linkage(squareform(dissimilarity), method='single', optimal_ordering=False) #single, complete, weighted, centroid, ward
            labels = get_clusters_hierarchical(hierarchy, k, 0, np.max(dissimilarity), max_iter, atol=atol)
            if not np.isnan(labels[0]):
                labels = labels - 1
                sensors = betweenness_sensors(G,labels)
                if not all(np.isnan(sensors)):
                    if len(sensors) != k:
                        print('Non k hier')
                partitionings_list[k].append((labels,sensors))
            else:
                partitionings_list[k].append((np.nan,np.nan))
                
          
        for corr_num,corr in enumerate(corr_list):
            
            similarity = np.copy( np.abs(corr) )
            similarity = (similarity + similarity.T)/2
            spectral_clustering = SpectralClustering(n_clusters = k, n_components = k, affinity='precomputed', assign_labels='kmeans', n_init=30, random_state=10).fit(similarity)
            labels = spectral_clustering.labels_ 
            sensors = betweenness_sensors(G,labels)
            partitionings_list[k].append((labels,sensors))
        

        
        ################################################################################################################################
        ############################################ PARTITIONING OF OTHER METHODS ###################################################################
        ################################################################################################################################
        
        # =============================================================================
        #         MODULARITY MAXIMIZATION
        # =============================================================================
            
        communities = greedy_modularity_communities(G, weight='weight', resolution=1, cutoff=k, best_n=k)
        
        if len(communities) != k:
            print('Modularity imbroglia')
        
        i = -1
        labels = np.zeros((len(G.nodes)), dtype=int)
        for clus in communities:
            i += 1
            labels[ list(clus) ] = i
            
        sensors = betweenness_sensors(G,labels)
        partitionings_list[k].append((labels,sensors))
        
        
        # =============================================================================
        #         GIRVAN-NEWMAN
        # =============================================================================
        
        communities = gn_partitionings[k]
        
        labels = np.zeros((N,), dtype=int)
        i = -1
        for clus in communities:
            i += 1
            labels[list(clus)] = i

        sensors = betweenness_sensors(G,labels)
        partitionings_list[k].append((labels,sensors))
        

        # =============================================================================
        #         NORMALIZED SPECTRAL (Ncut) WITH KMEANS
        # =============================================================================

        embedding = norm_eigvecs[:,:k] #first k eigenvectors, also the kernel is necessary
        
        kmeans = KMeans(n_clusters=k).fit(embedding)
        labels = kmeans.labels_
        
        sensors = betweenness_sensors(G,labels)
        partitionings_list[k].append((labels,sensors))
        
        
        # # =============================================================================
        # #         UNNORMALIZED SPECTRAL (RATIOCUT) WITH KMEANS
        # # =============================================================================
        
        embedding = eigvecs[:,:k] #first k eigenvectors, also the kernel is necessary
        
        kmeans = KMeans(n_clusters=k).fit(embedding)
        labels = kmeans.labels_
        
        sensors = betweenness_sensors(G,labels)
        partitionings_list[k].append((labels,sensors))
        
        
        # # =============================================================================
        # #         NORMALIZED SPECTRAL (NCUT)
        # # =============================================================================
                
        # spectral_clustering = SpectralClustering(n_clusters = k, n_components = k, affinity='precomputed', assign_labels='kmeans', n_init=50, random_state=123).fit(A)
        # labels = spectral_clustering.labels_ 
        # sensors = betweenness_sensors(G,labels)
        # partitionings_list[k].append((labels,sensors))
        
    
    # =============================================================================
    # SAVE THE RESULTS OF THE CURRENT NETWORK BEFORE GOING TO THE NEXT ONE
    # ============================================================================
    
    # BEFORE GOING TO THE NEXT NET, SAVE THE DICT 
    #partitionings_data.append( partitionings_list )

    data = partitionings_list
    with open (path + 'partitionings' + kind + ' %i' %net_num,'wb') as f:
        pickle.dump(data,f)   
