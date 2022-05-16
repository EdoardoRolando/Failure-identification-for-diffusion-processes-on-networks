#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:39:31 2021

@author: edoardo
"""

import networkx as nx
import numpy as np
import scipy.linalg
from scipy.special import comb
import pickle
from tqdm import tqdm

from creation import random_grid_networks_generator, ER_random_networks_generator, BA_random_networks_generator, random_regular_networks_generator
from failure import network_response, fluctuations_network_response


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
# NETWORKS CREATIONS
# =============================================================================

#NETWORK CREATION PARAMETERS
N = 225 #number of nodes
num_networks = 10 #random networks number 
sources_num = 5


# =============================================================================
# #E-R
# =============================================================================

# ER_expected_edges_num = comb(N,2) * wiring_prob
# ER_expected_av_degree = 2 * ER_expected_edges_num / N
# print(ER_expected_av_degree)
# print(wiring_prob, ( (1 + 0.00001) * np.log(N) ) / N) #if bigger than wiring_prob, almost surely connected

# desired_av_deg = 3.73
# needed_p = desired_av_deg * N / ( 2 * comb(N,2))
# wiring_prob = needed_p

# G_list = ER_random_networks_generator(N, num_networks, sources_num, wiring_prob, seed=123)


# =============================================================================
# # #RANDOM GRID NETWORKS
# =============================================================================

# N_side = 15
# N = N_side * N_side
# G_list = random_grid_networks_generator(N_side, num_networks, sources_num, seed=123456)


# =============================================================================
# RANDOM REGULAR 
# =============================================================================

G_list = random_regular_networks_generator(N, num_networks, sources_num, seed=123)


# =============================================================================
# B-A
# =============================================================================

# edges_num = 2 #for each new node, preferentially attached to previous nodes with high degree
# G_list = BA_random_networks_generator(N, num_networks, sources_num, edges_num, seed=54124)


# =============================================================================
# VENICE
# =============================================================================

# grafo = nx.read_adjlist('grafo_er-speed')
# grafo = nx.convert_node_labels_to_integers(grafo, first_label=0, ordering='default', label_attribute='old label')
# mapping = {node:int(grafo.nodes[node]["old label"]) for node in grafo.nodes()}
# grafo = nx.relabel_nodes(grafo, mapping)
# for node in grafo:
#    del grafo.nodes[node]['old label']


# =============================================================================
# Other to be implemented
# =============================================================================

# RANDOM N-W-S NETWORKS 
# random_partition_graph
# gaussian_random_partition_graph


# =============================================================================
# NETWORKS DATA COMPUTATION
# =============================================================================

# TO SAVE G, QDICT, P_DIFF_LIST_ FLUC_LIST, CORR
data_to_be_saved = []

for net_num, G in tqdm(enumerate(G_list), total=len(G_list), position=0, leave=True):
    

    #################################################################################################################################################
    ########################################## LAPLACIAN SPECTRUM #################################################################################
    #################################################################################################################################################
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G,nodelist=sorted(G.nodes), weight='weight')
    #A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G,nodelist=sorted(G.nodes), weight='weight')
    #L = nx.linalg.laplacianmatrix.laplacian_matrix(G, weight='weight')
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L) #automatically sorted in ascending order, and eigvecs are orthonormal, even in the degenerate case
    #eigvecs_inv = scipy.linalg.inv(eigvecs)
    if np.isclose(eigvals[1],0):
        print('Non connesso')
    
    #################################################################################################################################
    ############################################ LINEAR SYSTEM Lp=s SOLUTION ########################################################
    #################################################################################################################################
    
    # Demands
    s = np.asarray( list(nx.get_node_attributes(G,'current').values()) )
    
    # Stationary solution
    # With the pseudoinverse or lstsq, p has no components in the ker
    p = scipy.linalg.lstsq(L,s, lapack_driver='gelsy')[0]
    
    # Normalize
    p_sum = np.sum(p)
    if not np.isclose(p_sum, 0):
        p = p - p_sum / N

    #Set p as nodes attributes
    p_dict = dict(zip( G.nodes(), p ))
    nx.set_node_attributes(G, p_dict, name='pressure')


    # =============================================================================
    #     FLUXES 
    # =============================================================================
        
    # # Edges fluxes
    # # Directed copy of G, with edges direction following current direction
    # D = nx.create_empty_copy(G) #there are nodes and attribure from G, but no edges
    # D = nx.DiGraph(D)
    
    # for u,v in G.edges():
    #     weight = G[u][v]['weight']
    #     current = ( p[u] - p[v] ) * weight
    #     if current > 0:
    #         D.add_edge(u,v)
    #         D[u][v]['weight'] = weight
    #         D[u][v]['current'] = current 
    #     else:
    #         D.add_edge(v,u)
    #         D[v][u]['weight'] = weight
    #         D[v][u]['current'] = -current 
    
    # Both fluxes
    Qdict = dict()
    for u,v in G.edges():
        current = ( p[u] - p[v] ) * G[u][v]['weight']
        Qdict[(u,v)] = current
        Qdict[(v,u)] = -current

    
    # =============================================================================
    #     CORRELATION VS FLUXES
    # =============================================================================

    # x = []
    # y = []
    # for node1,node2 in G.edges():
    #     x.append( corr21[node1,node2] )
    #     #y.append( abs(Qdict_in[(node1,node2)]) ) 
    #     y.append( G[node1][node2]['weight'] )
      
    # plt.figure(figsize=(20,10))
    # plt.plot(x,y, linestyle='None', marker='o')
  
    # plt.title('corr21', fontsize=20)
    # plt.xlabel('Correlation', fontsize=20)
    # plt.ylabel('Weight', fontsize=20)
    # plt.show()
 
    
    # =============================================================================
    #     P_DIFF FOR EACH BROKEN EDGE
    # =============================================================================
        
    #PER OGNI NETWORK MA NON PER OGNI PARTITIONING
    p_diff_list = dict()
    for i,broken_edge in enumerate(G.edges()):
        p_diff_list[broken_edge] = network_response(L, s, p, broken_edge, timing='transient')
        
    # # Tests to see qual è il massimo mano a mano 
    # np.argmax(np.abs(ptdata - solution[:,None]),axis=0)
    # np.argmax((np.abs(ptdata - solution[:,None])[sensors]),axis=0)
    # np.argmax(np.round(np.abs(ptdata - solution[:,None])[sensors], 2),axis=0)
    # np.isclose(ptdata[:,1], solution,atol=10e-4)
    # np.isclose(ptdata[:,167], new_solution,atol=10e-4)


    # =============================================================================
    #     VAR AND COVAR OF FLUCTUATIONS FOR EACH FLUCTUATING EDGE 
    # =============================================================================

    # # SIMULATION PARAMETERS
    # dt = .001  # Time step
    # T = 7.  # Total time
    # n = int(T / dt)  # Number of time steps
    
    # # To avoid computing it at each iteration below
    # p_stat = scipy.linalg.lstsq(L, st[0], lapack_driver='gelsy')[0]
    # dt_demands = dt * st[0] 
    # dt_L = dt * L 
    
    # # Random variables
    # rng = np.random.default_rng(615)
    # random_var = np.zeros((n,))
    # for i in range(n-1):
    #     random_var[i] = rng.uniform(-np.sqrt(3), np.sqrt(3)) 
        
    # # EDGES FLUCTUATIONS
    fluc_list = dict()    
    # for i,fluc_edge in tqdm(enumerate(G.edges()), total=len(G.edges()), position=0, leave=True):
    #     fluc_list[fluc_edge] = fluctuations_network_response(A, dt_L, p_stat, dt_demands, fluc_edge, random_var, dt, T, n)


    # =============================================================================
    # NETWORKS DATA TO BE SAVED
    # =============================================================================
    
    data_to_be_saved.append((G, Qdict, p_diff_list, fluc_list))
        

# =============================================================================
# SAVE NETWORKS DATA
# =============================================================================


data = data_to_be_saved
with open (path + 'networks' + kind,'wb') as f:
    pickle.dump(data, f)



