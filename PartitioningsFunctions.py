#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:35:51 2022

@author: edo
"""


################################################################################################################################################
######################################### FROM CORRELATION AND THRESHOLD TO CLUSTERS ##########################################################################
################################################################################################################################################


import numpy as np
import networkx as nx
  
from sklearn.cluster import AffinityPropagation
import markov_clustering as mc

from scipy.cluster.hierarchy import fcluster


################################################################################################################################################
######################################### MY CORR PARTITIONINGS ##########################################################################
################################################################################################################################################
    

def corr_partitioning_medoid1(corr, corr_threshold):
    
    #Number of nodes
    N = len(corr)
    
    corr = np.copy(corr)
    
    # If we want to consider only the abs value
    #corr = np.abs(corr)
    
    np.fill_diagonal(corr,0)
    sorted_corr = np.dstack(np.unravel_index(np.argsort(corr.ravel()), (N, N)))[0][::-1] #[N::][::2]
    #np.fill_diagonal(corr,1)
    
    labels = np.arange(0,N)
    clusters_dict = {n:[] for n in np.arange(0,N)}
    
    for node1,node2 in sorted_corr:
        
        # Since they are sorted, as soon as we reach corr_threshold we break the for loop
        if corr[node1,node2] < corr_threshold: 
            break
        
        # If the node node1 that wants to represent node2 was already taken by another node (not in clusters.dict)
        # Or if node2 is already a representative node labels[node2] = node2
        # Or if node2 is already taken labels[node2] = another node, continue
        elif node1 not in clusters_dict.keys() or len( clusters_dict[labels[node2]] ) > 0:
            continue
        
        # Otherwise we can put edge[1] 
        else:
            labels[node2] = node1
            clusters_dict[node1].append(node2)
            del clusters_dict[node2]
            
    # Sensors are the medoids
    sensors = np.unique(labels)
    
    # The labels are converted from the medoid/sensor number to a sequential integer number 
    i = -1
    for sens in sensors:
        i += 1
        labels[ sens ] = i
        labels[ clusters_dict[sens] ] = i
              
    #clusters list creation
    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
    
    i = 0
    for clus in clusters:
        i += len(clus)
    if not i == N:
        print('Non hai classificato tutti i nodi')

    return labels, sensors
      
        

def corr_partitioning_medoid(corr, corr_threshold, sorted_corr):
    
    #Number of nodes
    N = len(corr)
    
    corr = np.copy(corr)
    
    # If we want to consider only the abs value
    #corr = np.abs(corr)
    
    # np.fill_diagonal(corr,0)
    # sorted_corr = np.dstack(np.unravel_index(np.argsort(corr.ravel()), (N, N)))[0][::-1] #[N::][::2]
    # np.fill_diagonal(corr,1)
    
    labels = np.arange(0,N)
    clusters_dict = {n:[n] for n in np.arange(0,N)}
    
    for edge in sorted_corr:
        
        # Since they are sorted, as soon as we reach corr_threshold we break the for loop
        if corr[edge[0],edge[1]] < corr_threshold: 
            break
        
        # Otherwise we check if the nodes of that edge will be put togheter in the same cluster
        else:
            # check the clusters in which the nodes of edge are
            label1 = labels[edge[0]]
            label2 = labels[edge[1]]
            
            # They are already in the same cluster
            if label1 == label2:
                continue
            
            # Nodes in the two clusters 
            nodes_clus1 = clusters_dict[label1]
            nodes_clus2 = clusters_dict[label2]
            
            candidate_clus = nodes_clus1 + nodes_clus2
                
            candidates_medoid = {}
            for node in candidate_clus:
                
                node_corr = corr[node,candidate_clus]
                
                #current_threshold = 0.9 * corr[edge[0],edge[1]]
                if all(k >= corr_threshold for k in node_corr):
                    candidates_medoid[node] = sum(node_corr) 
                    #candidates_medoid[node] = np.average(node_corr)
                    
            if not candidates_medoid:
                #try the next edge, this one is refused, no medoid that can represent all candidate_clus nodes
                continue 
            
            # Otherwise the best representative, if more than one, will be the new sensor of the newly formed cluster
            else: 
                # If we are here, one or more nodes will be added to an already formed cluster (formed by a single or more nodes) 
                # Medoid as label
                medoid = max(candidates_medoid, key=candidates_medoid.get)
                labels[nodes_clus1] = medoid
                labels[nodes_clus2] = medoid
                clusters_dict[medoid] = candidate_clus
                
                if len(candidate_clus) == N:
                    break
    
    # Sensors are the medoids
    sensors = np.unique(labels)
    
    # The labels are converted from the medoid/sensor number to a sequential integer number 
    i = -1
    for sens in sensors:
        i += 1
        labels[ clusters_dict[sens] ] = i
        
    i = 0
    for sens in sensors:
        i += len( clusters_dict[sens] )
    if not i == N:
        print('Non hai classificato tutti i nodi')
    
    
    return labels, sensors

        

def corr_partitioning_affcomm(corr, corr_threshold):
    
    #Number of nodes
    N = len(corr)
    
    # To avoid modifying the original one
    corr = np.copy(corr)
    
    # Correlation preprocessing 
    corr[corr < corr_threshold] = 0 #or corr = (corr > threshold) * corr
    np.fill_diagonal(corr, 0)
    
    # There might be Nan, they are put to 0
    corr = np.nan_to_num(corr, copy=False)

    # Inizialization
    clusters_dict = {n:[n] for n in np.arange(0,N)}
    
    # Affinity matrix is initialized as the thresholded correlation matrix
    affinity = np.copy(corr)
    
    while (True):

        # Normalize affinity columns to 100% 
        col_sum = affinity.sum(axis=0, dtype=np.float64)
        factors_to_normalize = np.divide(100, col_sum, out=np.zeros_like(col_sum), where=col_sum!=0)
        
        # Multiply each column of affinity with its factor (anche senza None funziona, per rows serve [:,None])
        affinity = factors_to_normalize[None] * affinity 
        representativity = affinity.sum(axis=1) 
        
        # ###!!!! oppureee
        # scores = representativity[:,None] * affinity # Each row with its factor
        # representativity = scores.sum(axis=1)
        
        if np.allclose(representativity,0):
          break

        # CLUSTERING
        center = np.argmax(representativity)
        nodes = np.nonzero( affinity[center,:] )[0].tolist()
        clusters_dict[center].extend(nodes)
        
        affinity[:,center] = 0
        affinity[center,:] = 0
        
        affinity[:,nodes] = 0 
        affinity[nodes,:] = 0
        #print(center, nodes)
        
        for node in nodes:
            del clusters_dict[node]

    sensors = np.asarray( list(clusters_dict.keys()) )
        
    # The labels are converted from the medoid/sensor number to a sequential integer number 
    labels = np.arange(0,N)
    i = -1
    for sens in sensors:
        i += 1
        labels[ clusters_dict[sens] ] = i
        
    i = 0
    for sens in sensors:
        i += len( clusters_dict[sens] )
    if not i == N:
        print('Non hai classificato tutti i nodi')


    return labels,sensors




################################################################################################################################################
######################################### GET CLUSTERS FUNCTION ##########################################################################
################################################################################################################################################
    


def get_clusters(method, corr, number_of_clusters, min_th, max_th, max_iter, atol, sorted_corr):
    
    th_before = -1
    iteration_counter = 0
    
    while (True):
        
        # Iteration counter update, if max reached return nan
        iteration_counter += 1
        if  iteration_counter == max_iter:
            return np.nan, np.nan
        
        # Update of the threshold
        th = min_th + (max_th - min_th) / 2
        
        if np.isclose(th_before, th, atol=atol):
            #print('ff')
            return np.nan, np.nan
        
        th_before = th
        #print(th)
   
        # Con python 3.10 and above use match cases
        # Method choice
        if method == 'medoid1':
            labels, sensors = corr_partitioning_medoid1(corr, corr_threshold=th)

        elif method == 'medoid':
            labels, sensors = corr_partitioning_medoid(corr, corr_threshold=th, sorted_corr=sorted_corr)

        elif method == 'affcomm':
            labels, sensors = corr_partitioning_affcomm(corr, corr_threshold=th)
    
        else: 
            raise Exception ('Wrong method type')
        
        
        current_clus_num = len(sensors)
        #print(current_clus_num)
 
        # Check wether iterate again or we are ok 
        if current_clus_num == number_of_clusters:
            return labels, sensors
        
        # So that in the next iteration th is smaller than the current th
        # The bigger th the more the communites
        elif current_clus_num > number_of_clusters:
            max_th = th
            continue

        elif current_clus_num < number_of_clusters:
            min_th = th
            continue
   
        
        
def get_clusters_hierarchical(hierarchy, number_of_clusters, min_th, max_th, max_iter, atol):
    
    th_before = min_th
    iteration_counter = 0
    
    while (True):
        
        # Iteration counter update, if max reached return nan
        iteration_counter += 1
        if  iteration_counter == max_iter:
            return [np.nan]
        
        # Update of the threshold
        th = min_th + (max_th - min_th) / 2
        
        if np.isclose(th_before, th, atol=atol):
            return [np.nan]
        
        th_before = th
        
        labels = fcluster(hierarchy, th, criterion='distance') 
        
        current_clus_num = len(np.unique(labels))

        # Check wether iterate again or we are ok 
        if current_clus_num == number_of_clusters:
            return labels
        
        # So that in the next iteration th is bigger than the current th
        # The smaller th the more the communites
        elif current_clus_num > number_of_clusters:
            min_th = th
            continue

        elif current_clus_num < number_of_clusters:
            max_th = th
            continue
        
        
    
def get_clusters_aff(corr, number_of_clusters, min_th, max_th, max_iter):
    
    i = 0
    tol = 0
    while (True):
        
        i += 1
        if  i % max_iter == 0:
            tol += 1
        
        th = min_th + (max_th - min_th) / 2
        
        clustering = AffinityPropagation(damping=0.9, preference = th, affinity='precomputed', max_iter=6000).fit(corr)
        labels = clustering.labels_
        sensors = medoid_sensors(labels, corr)
        
        if number_of_clusters - tol <= len(sensors) <= number_of_clusters + tol:
            return labels,sensors

        elif len(sensors) > number_of_clusters:
            max_th = th
            continue

        elif len(sensors) < number_of_clusters:
            min_th = th
            continue        
        
        

def get_clusters_markov(corr, number_of_clusters, min_th, max_th, max_iter):
    
    i = 0
    tol = 0
    while (True):
        
        i += 1
        if  i % max_iter == 0:
            tol += 1
        
        th = min_th + (max_th - min_th) / 2
        
        result = mc.run_mcl(corr, inflation = th)
        clusters = mc.get_clusters(result)
        
        if number_of_clusters - tol <= len(clusters) <= number_of_clusters + tol:
            return clusters

        elif len(clusters) > number_of_clusters:
            max_th = th
            continue

        elif len(clusters) < number_of_clusters:
            min_th = th
            continue



################################################################################################################################################
######################################### SENSOR CHOICE FOR OTHER CLUSTERINGS ##########################################################################
################################################################################################################################################
    

def betweenness_sensors(G, labels):
    
    # Clusters list creation
    clusters = []
    for lab in np.unique(labels):
        arr = np.where(labels==lab)[0]
        clusters.append(arr.tolist())

    sensors = []
    for clus in clusters:
        nodes_dict = nx.betweenness_centrality(G.subgraph(clus),weight='weight')
        sensors.append( max(nodes_dict, key=nodes_dict.get) ) #returns the key of the element with the max value

    return sensors
        

# #current_flow_betweenness_centrality
# sensors = []
# for clus in clusters:
#     nodes_dict = nx.current_flow_betweenness_centrality(G.subgraph(clus),weight='weight')
#     sensors.append( max(nodes_dict, key=nodes_dict.get) ) #returns the key of the element with the max value
# partitionings_list.append((labels,sensors))     
       


# #eigenvector_centrality
# sensors = []
# for clus in thclusters:
#     nodes_dict = nx.eigenvector_centrality(G.subgraph(clus),max_iter=300,weight='weight')
#     sensors.append( max(nodes_dict, key=nodes_dict.get) ) #returns the key of the element with the max value
   

            
def sensors_choice(D, labels, choice):
    
    sensors = []
    
    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
    
    #SENSOR ON THE NODE WITH HIGHEST IN-DEGREE (WEIGHTED) FOR EACH CLUSTER
    if choice == 'in':
        for clus in clusters:
            #D is the directed version of the graph, according to edges fluxes
            #taking the clus as a subgraph (i.e only INTERNAL edges), select the node with highest in_degree, weighted not by the edge weight but by its flux
            in_degrees = dict(D.subgraph(clus).in_degree(weight='current'))
            sensors.append(max(in_degrees, key=in_degrees.get))
        sensors = np.asarray(sensors)
        return sensors
    
    #SENSOR ON THE NODE WITH HIGHEST OUT-DEGREE (WEIGHTED) FOR EACH CLUSTER
    elif choice == 'out':
        for clus in clusters:
            out_degrees = dict(D.subgraph(clus).out_degree(weight='current'))
            sensors.append(max(out_degrees, key=out_degrees.get))
        sensors = np.asarray(sensors)
        return sensors
        
    else: raise Exception ('wrong input')
    
    

def medoid_sensors(labels, corr):

    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
    
    sensors = []
    
    for clus in clusters:
        
        candidates_medoid = dict()
    
        for node in clus:

            # candidates_medoid[node] = sum( np.abs(corr[node,clus]) )
            corr_values = [ corr[node,clus_node] for clus_node in clus if corr[node,clus_node] > 0]
            candidates_medoid[node] = sum( corr_values )
        
        sensors.append( max(candidates_medoid, key=candidates_medoid.get) )
    
    sensors = np.asarray(sensors) 
    
    return sensors


