
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:39:31 2021

@author: edoardo
"""

import networkx as nx
import numpy as np
import pickle
from tqdm import tqdm
import time

from failure import network_response, failure_localization, fluctuations_network_response, fluc_failure_localization

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
#     DATA STORED NEEDED
# =============================================================================

# NETWORKS DATA
with open (path + 'networks' + kind, 'rb') as f:
    networks_data = pickle.load(f)
    
# # PARTITIONINGS DATA
# with open (path + 'partitionings uniform', 'rb') as f:
#     partitionings_data = pickle.load(f)
    
net_results = []
num_of_net = 10
        
for net_num, network_list in tqdm(zip(range(0,num_of_net),networks_data[0:num_of_net]), total=len(networks_data), position=0, leave=True):    
    
    # PARTITIONINGS DATA
    with open (path + 'partitionings' + kind + ' %i' %net_num, 'rb') as f:
        partitionings_list = pickle.load(f)
        
    # Unfold networks_list
    G, Qdict, p_diff_list, fluc_list = network_list
  
    # =============================================================================
    #     DATA NEEDED
    # =============================================================================
    
    # Laplacian matrix
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    
    # Demands
    s = np.asarray( list(nx.get_node_attributes(G,'current').values()) )
    p = np.asarray( list(nx.get_node_attributes(G,'pressure').values()) )
    N = len(s)
    
    # =============================================================================
    #     BEGIN
    # =============================================================================
    
    #RESULTS FOR EACH NETWORK AND EACH K
    #netk_results = []
    netk_results = {k:[] for k in partitionings_list.keys()}
    
    # =============================================================================
    #     ITERATION OVER THE NUMBER OF PARTITIONS K
    # =============================================================================

    #for each clustering method (i.e. couple clusters, sensors), test failure localization for different deamnds patterns and save the results
    for k in tqdm(list(partitionings_list.keys()), total=len(partitionings_list.keys()), position=0, leave=True):
    #for k in range(93,94):
        
        #RESULTS FOR EACH NETWORK, EACH K, EACH CLUSTERING
        netkclus_results = []
        
        # =============================================================================
        #     ITERATION OVER ALL THE PARTITIONING METHODS
        # =============================================================================
    
        #for partitioning in partitionings_list[k]:
        for num,partitioning in enumerate(partitionings_list[k]):
            
            labels = partitioning[0]
            sensors = partitioning[1]
            
            if np.isnan(labels).all():
                
                netkclus_results.append( [np.nan] )
         
                # current_results_dic = {edge:[np.nan,np.nan] for edge in G.edges()}
                # netkclus_results.append( current_results_dic )
                
                continue 
        
            #clusters list creation
            clusters = []
            for i in range(0,max(labels)+1):
                arr = np.where(labels==i)[0]
                if arr.size > 0:
                    clusters.append(arr.tolist())
            
            # Check if all nodes are in a cluster
            # i = 0
            # for clus in clusters:
            #     i += len(clus)
                   
            #################################################################################################################################################
            ########################################## REDUCED NETWORK CREATION ##########################################################################
            #################################################################################################################################################
            
            #for each couple of different clusters, compute the cut size between them (i.e the sum of all the edges between them, with weights)
            #if intraclusters_weight (i.e cut size) is different from zero, connect the sensor nodes of the two clusters with an edge with that weight
            # self edges, i.e nx.cut_size(G,clusters[i],clusters[i], weight='weight') to denote the dimension of the edges in the cluster? i.e its volume
            
            #REDUCED NETWORK EDGES WEIGHTS COMPUTATION 
            R_edges = []
            G_edges_boundary = dict()
            for i in range(len(clusters)):
                for j in range(i+1,len(clusters)): #no repetition (useless for undirected, and no self edges)
                    #dipoles_number += nx.cut_size(G, clusters[i], clusters[j])
                    cutsize = nx.cut_size(G, clusters[i], clusters[j], weight='weight') #alternatives: normalized_cutsize, conductance
                    if cutsize!=0:
                        #to add edges with weight to the reduced network R, add_edges_from needs an ebunch, so an iterable container of edge.tuples
                        #since we want also to add the weights from cutsize, we need 3-tuples of the form (2, 3, {'weight': 3.1415}), i.e (node,node,edge attribute dictionary)
                        R_edges.append((sensors[i], sensors[j], {'weight': cutsize}))   
                        G_edges_boundary[i,j] = list( nx.edge_boundary(G, clusters[i], clusters[j]) ) 
                        G_edges_boundary[j,i] = G_edges_boundary[i,j]
        
                
            #Reduced network creation
            R = nx.Graph()
            R.add_nodes_from(sensors)
            R.add_edges_from(R_edges)
            
       
            #################################################################################################################################################
            ########################################## LAPLACIAN SPECTRUM REDUCED NETWORK #################################################################################
            #################################################################################################################################################
            
            # #DRAW PARAMETERS
            # #pos = nx.spring_layout(R, weight='weight', seed=123)
            # red_pos = {}
            # for sens in sensors:
            #     red_pos[sens] = G.nodes[sens]['pos']
            
            # red_L = nx.linalg.laplacianmatrix.laplacian_matrix(R)
            # red_L = red_L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
            
            # red_eigvals, red_eigvecs = scipy.linalg.eigh(red_L) #automatically sorted in ascending order, and eigvecs are orthonormal, even in the degenerate case
            # if not np.allclose(red_L,red_L.T):
            #     raise Exception('Warning: eigh with non symmetric matrix')
            
            # #SERVE?
            # red_eigvecs_inv = scipy.linalg.inv(red_eigvecs) #la prima componente, che dovrebbe essere nulla è 10^-14
            # red_pk = red_eigvecs_inv@p[sensors]
    
            # #SOURCES E SINKS AGGREGATI
            # #Original demands and stationary solution
            # red_s = [sum(s[clus]) for clus in clusters]
            # clustered_red_p = [sum(p[clus]) for clus in clusters]
            # red_p = scipy.linalg.lstsq(red_L, red_s, lapack_driver='gelsy', overwrite_a=True)[0]
            
            # #For the dynamcis on the reduced network
            # red_st = [sum(s[clus]) for clus in clusters]
            # red_p_st = scipy.linalg.lstsq(red_L, red_st, lapack_driver='gelsy')[0]


            # C = np.zeros((N,len(clusters)))
            # for i,clus in enumerate(clusters):
            #     C[clus,i] = 1
                
            # # np.allclose(C.T@L@C, red_L)
            
            # # D = np.zeros((len(sensors),N))
            # # for i,sens in enumerate(sensors):
            # #     D[i,sens] = 1
            
            # Cpin = scipy.linalg.pinv(C)
            # Laverage = Cpin@L@C #che NON È SIMMETRICA
            
            # eigvals, eigvecs = scipy.linalg.eigh(L) 
            # av_eigvals, av_eigvecs = scipy.linalg.eig(Laverage) 
        
            
            #################################################################################################################################################
            ########################################## FAILURES #################################################################################
            #################################################################################################################################################

            tested_edges = list(G.edges())
                    
            # =============================================================================
            #             # No monopole threshold
            # =============================================================================
    
            current_results, current_confidence = failure_localization(G, p_diff_list, s, p, Qdict, R, tested_edges, labels, sensors, G_edges_boundary)
            
            current_results_dic = {edge:[res,conf] for edge,res,conf in zip(tested_edges, current_results, current_confidence)}
            netkclus_results.append( current_results_dic )
            

        # Once finished all partitioning for net_resultsa k, before going to the next k save these results 
        # netk_results.append( netkclus_results )
        netk_results[k] = netkclus_results 
    
        
    # Once finished all partitioning for all k for a network, before going to the next one save these results 

    # =============================================================================
    # SAVE THE RESULTS OF THE CURRENT NETWORK BEFORE GOING TO THE NEXT ONE
    # =============================================================================

    data = (G, Qdict, netk_results) 
    with open (path + 'failures' + kind + ' %i' %net_num, 'wb') as f:
        pickle.dump(data,f)


