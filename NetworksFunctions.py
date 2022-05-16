#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:37:17 2022

@author: edo
"""

import numpy as np
import networkx as nx


#################################################################################################################################################
############################################ NETWORK CREATION #################################################################################
#################################################################################################################################################


def random_grid_networks_generator(N_side, num_networks, sources_num, seed):    
    
    G_list = []
    
    for num in range (0, num_networks):
        G = nx.grid_graph([N_side,N_side])
        
        #grid_graph names nodes by default with (i,j); we create a dictionary mapping nodes' names (i.e.(row,column)) as positions (with rotation to have the (0,0) node on the top left)
        pos = {(x,y):(y,-x) for x,y in G.nodes()} 
        #set the positions (old labels) as nodes attributes, BEFORE changing the label with convert_node_labels_to_integers
        nx.set_node_attributes(G, pos,'pos') 
        
        #convert nodes labels from the default (i,j) (now used only as a node attribute to give the position, with rotation) to incremental natural numbers
        G = nx.convert_node_labels_to_integers (G, first_label=0) 

        #edges weights and nodes demands definition
        edges_and_demands(G, sources_num, seed + num)
        
        G_list.append(G)
    return G_list


def ER_random_networks_generator(N, num_networks, sources_num, wiring_prob, seed, directed=False):    
    G_list = []
    
    #global tent_num
    tent_num = 0
    
    for num in range (0, num_networks):
        G = nx.fast_gnp_random_graph(N, wiring_prob, seed + num, directed)
        
        while not nx.is_connected(G):
            tent_num += 1
            G = nx.fast_gnp_random_graph(N, wiring_prob, seed + num + 30 + tent_num, directed)
            #print(seed + num + 154679 + tent_num)
            
        #edges weights and nodes demands definition
        edges_and_demands(G, sources_num, seed + num)
        
        G_list.append(G)
    return G_list


def BA_random_networks_generator(N, num_networks, sources_num, edges_num, seed):    
    G_list = []
    
    for num in range (0, num_networks):
        G = nx.barabasi_albert_graph(N, edges_num, seed + num) 
        
        #edges weights and nodes demands definition
        edges_and_demands(G, sources_num, seed + num)
        
        G_list.append(G)
    return G_list


def random_regular_networks_generator(N, num_networks, sources_num, seed):    
    
    G_list = []
    
    #global tent_num
    tent_num = 0
    
    for num in range (0, num_networks):
        G = nx.random_regular_graph(4, N, seed=seed)
        
        while not nx.is_connected(G):
            tent_num += 1
            G = nx.random_regular_graph(4, N, seed=seed+num+tent_num+130)
        
        #edges weights and nodes demands definition
        edges_and_demands(G, sources_num, seed + num)
        
        G_list.append(G)
    return G_list


# =============================================================================
# RANDOM WEIGHTS AND DEMANDS
# =============================================================================

def edges_and_demands(G, sources_num, seed):
    
    N = len(G.nodes())
    
    #Random edges weigth 
    rng = np.random.default_rng(seed = seed)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = rng.integers(1,10,endpoint=True)
        #G.edges[u,v]['weight'] = 1
        #G.edges[u,v]['weight'] = rng.exponential(15)
        
    #demands and sources
    s = rng.uniform(low=-10, high=-1, size=(N, ))  #half-open interval [low, high) or s = np.full((N, ), -1.)
    #s = - rng.exponential(15, size=(N,))
    sources_indices = rng.choice(G.nodes(), sources_num, replace=False)
    
    #set the s on sources to zero in order to adjust them to satisfy the solvability condition for the initial network
    s[sources_indices] = 0 
    total_demand = s.sum()
    s[sources_indices] = - total_demand/len(sources_indices)
    
    #solvability condition: s.sum() should be zero
    if not np.isclose(s.sum(), 0):
        raise Exception('Warning: s does not sum to zero')
    
    s_dict = dict(zip( G.nodes(), s ))
    nx.set_node_attributes(G, s_dict, name='current')
    
    
