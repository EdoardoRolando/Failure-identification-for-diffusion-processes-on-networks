#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 21:17:41 2022

@author: edo
"""

import pickle
import networkx as nx
import numpy as np
from tqdm import tqdm

from correlation import spectral_correlation, complete_spectralflux_correlation, spectralflux_correlation, single_spectralflux_correlation, multifluc_spectralflux_correlation, multiflucbi_spectralflux_correlation 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


###############################################################################
###############################################################################
###############################################################################

# PATH HAS INFO ABOUT THE TYPE OF NET AND THE NUMBER OF SOURCES

path = ''
kind = ''

###############################################################################
###############################################################################
###############################################################################


with open (path + 'networks' + kind, 'rb') as f:
    networks_data = pickle.load(f)
    
  
net_corr_lists = []

   
num_of_net = 10
for G, _, _, _ in tqdm(networks_data[:num_of_net], total=len(networks_data[:num_of_net]), position=0, leave=True):
    
    
    # Adj matrix of G
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    # Take the forcing from G attributes
    s = np.asarray( list(nx.get_node_attributes(G,'current').values()) )
    #p = np.asarray( list(nx.get_node_attributes(G,'pressure').values()) )
    
    N = len(s)
    
    
    # =============================================================================
    #   CORRELATIONS
    # =============================================================================
    
    corr_old  = spectral_correlation(G)
    corr_complete = complete_spectralflux_correlation(G, s)
    
    corr10 = spectralflux_correlation(G, s, k_neig=1, k_local=0)
    corr21 = spectralflux_correlation(G, s, k_neig=2, k_local=1)
    
    corr_single1 = single_spectralflux_correlation(G, s, k_neig=1)
    corr_single2 = single_spectralflux_correlation(G, s, k_neig=2)
    
    corr_multifluc1 = multifluc_spectralflux_correlation(G, s, k_neig=1)
    corr_multifluc2 = multifluc_spectralflux_correlation(G, s, k_neig=2)
    
    corr_multiflucbi1 = multiflucbi_spectralflux_correlation(G, s, k_neig=1)
    corr_multiflucbi2 = multiflucbi_spectralflux_correlation(G, s, k_neig=2)

    net_corr_lists.append((corr_old, corr_complete, corr10, corr21, corr_single1, corr_single2, corr_multifluc1, corr_multifluc2, corr_multiflucbi1, corr_multiflucbi2))
    #net_corr_lists.append((corr_old, corr10, corr_single1, corr_multiflucbi1, corr_multiflucbi2, corr_complete))
    
    
# =============================================================================
# SAVE CORRELATIONS
# =============================================================================

data = net_corr_lists
with open (path + 'correlations' + kind,'wb') as f:
    pickle.dump(data,f)



