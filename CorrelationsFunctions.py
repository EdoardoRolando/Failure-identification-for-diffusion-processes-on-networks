#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:35:51 2022

@author: edo
"""

import numpy as np
import scipy.linalg
import networkx as nx


def spectral_correlation(G):

    N = len(G.nodes)
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)
    
    cov = np.zeros((N,N))
    for k in range(1,N):
        tensor = np.outer(eigvecs[:,k], eigvecs[:,k]) 
        tensor = tensor * (1/(2*eigvals[k]) )
        cov += tensor
    
    corr = np.zeros((N,N))

    for i in range(0,N):
        for j in range(0,N):
            corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])

    return corr



def complete_spectralflux_correlation(G, s):

# =============================================================================
#     Structural part
# =============================================================================
    
    N = len(G.nodes)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)

# =============================================================================
#     Dynamical part
# =============================================================================
    
    p = scipy.linalg.lstsq(L, s, lapack_driver='gelsy')[0]
    
# =============================================================================
#     Correlation computation
# =============================================================================    
    
    #Correlation array inizialization
    corr = np.zeros((N,N))
    partial_cov = np.zeros((N,N))
    amp = np.zeros((N,N))
    
    #Coefficients = 1/l+m
    #Since symmetric, computed only for the upper triangular part, then symmetrized
    coeff = np.zeros((N-1,N-1)) #N-1 since no ker
    for l in range(N-1):
        coeff[l,l:] = [1/(eigvals[l+1] + mu) for mu in eigvals[l+1:]]
    coeff = coeff + coeff.T
    coeff[np.diag_indices_from(coeff)] /= 2
        
# =============================================================================
#         METRIC MATRIX C_kh
# =============================================================================

    metric = np.zeros((N,N))
    idx = np.nonzero(A) #indices where A is nonzero, to speed up (ci mette 15 secondi in meno così)
   
    metric[idx] = - ( A[idx] * (p[idx[0]] - p[idx[1]]) ) ** 2
    #metric[idx] = - (np.sign(A[idx]) * (p[idx[0]] - p[idx[1]]) )**2 
    
    metric_diag = - np.sum(metric, axis=1) #axis=0 same, is symmetric
    np.fill_diagonal(metric, metric_diag) #np.allclose(np.sum(metric,axis=0),0) o axis=1 should be 0 
   
    # #With metric = identity we recover corr_old
    # metric = np.eye(N)
    
# =============================================================================
#         SCALAR PRODUCT MATRIX
# =============================================================================

    scalar_product = np.tensordot(metric,eigvecs[:,1:],axes=1)
    scalar_product = np.tensordot(eigvecs[:,1:].T,scalar_product,axes=1)

# =============================================================================
#         CORRELATION
# =============================================================================
    
    #OUTER PRODUCT BETWEEN COEFF 1/l+m E LA SCALAR PRODUCT MATRIX DI SOPRA
    cov = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            tensor = np.outer(eigvecs[i,1:], eigvecs[j,1:]) #no ker
            tensor = tensor*coeff #prodotto entry per entry
            cov[i,j] = np.einsum('lm,lm', tensor, scalar_product) 
            
    #CORRELATION COEFFICIENT: DIVIDE BY VARIANCE
    corr = np.zeros((N,N))

    for i in range(0,N):
        for j in range(0,N):
            corr[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])


    return corr


def spectralflux_correlation(G, s, k_neig, k_local):

# =============================================================================
#     Structural part
# =============================================================================
    
    N = len(G.nodes)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)

# =============================================================================
#     Dynamical part
# =============================================================================
    
    p = scipy.linalg.lstsq(L, s, lapack_driver='gelsy')[0]
    
# =============================================================================
#     Correlation computation
# =============================================================================    
    
    #Correlation array inizialization
    corr = np.zeros((N,N))

    
    #Coefficients = 1/l+m
    #Since symmetric, computed only for the upper triangular part, then symmetrized
    coeff = np.zeros((N-1,N-1)) #N-1 since no ker
    for l in range(N-1):
        coeff[l,l:] = [1/(eigvals[l+1] + mu) for mu in eigvals[l+1:]]
    coeff = coeff + coeff.T
    coeff[np.diag_indices_from(coeff)] /= 2
    
# =============================================================================
#         MODIFIED ADJ MATRIX
# =============================================================================
           
    # We evaluate the correlation of each node with all the nodes in a beighborhood of it decided by k_neig
    # According to the node we use as center, the metric is different (local list), thus we need to compute each one separatley 
    # guardiamo le correlazioni di center con i primi vicini, facendo fluttuare l'ambiente
    # l'ambiente è definito come i link che non sono nelle immediate vicinanze del nostro possibile cluster
    # quindi prendiamo un ambiente che NON comprende nè i primi vicini nè i secondi/i terzi vicini etc (k_local)
    for center in G.nodes():
        
        #NODES TO COMPUTE CORRELATIONS WITH NODE CENTER: NO FLUCTUATIONS, YES CORRELATION
        neig_list = list(nx.single_source_shortest_path_length(G, center, cutoff=k_neig).keys()) #neig_list = list(G.neighbors(center)) #non mette center nella list
        
        #NODES IN THE LOCAL BUBBLE: NO FLUCTUATIONS, NO CORRELATION
        local_list = list(nx.single_source_shortest_path_length(G, center, cutoff=k_local).keys())
        
# =============================================================================
#         MODIFIED ADJ MATRIX
# =============================================================================

        #The fluctuation of each link are proportional to its weight
        #Thus to make the links in local list not to fluctuate, we set their weights to zero
        A_tilde = np.copy(A)
        A_tilde[:,local_list] = 0
        A_tilde[local_list,:] = 0
            
# =============================================================================
#         METRIC MATRIX C_kh
# =============================================================================

        metric = np.zeros((N,N))
        idx = np.nonzero(A_tilde) #indices where A is nonzero, to speed up (ci mette 15 secondi in meno così)
        #print(len(idx[0]))
        metric[idx] = - ( A_tilde[idx] * (p[idx[0]] - p[idx[1]]) ) ** 2
        #metric[idx] = - (np.sign(A_tilde[idx]) * (p[idx[0]] - p[idx[1]]) )**2 
        
        metric_diag = - np.sum(metric, axis=1) #axis=0 same, is symmetric
        np.fill_diagonal(metric, metric_diag) #np.allclose(np.sum(metric,axis=0),0) o axis=1 should be 0 
        
# =============================================================================
#         SCALAR PRODUCT MATRIX
# =============================================================================

        scalar_product = np.tensordot(metric,eigvecs[:,1:],axes=1)
        scalar_product = np.tensordot(eigvecs[:,1:].T,scalar_product,axes=1)
        # #sotto ci mette 0.214 s, sopra ci mette 0.000332, 640 volte in meno
        # scalar_product = np.zeros((N-1,N-1)) #no ker
        # for l in range(N-1): 
        #     for m in range(N-1):
        #         scalar_product[l,m] = eigvecs[:,l+1]@metric@eigvecs[:,m+1]
        
# =============================================================================
#         CORRELATION
# =============================================================================

        #nodes couples to be evaluated (all wrt center, non between them)
        var_couples = [(ele,ele) for ele in neig_list if ele != center]
        covar_couples = [(center,ele) for ele in neig_list]  
        
        #MIGLIORABILE!!! ###!!!
        #OUTER PRODUCT BETWEEN COEFF 1/l+m E LA SCALAR PRODUCT MATRIX DI SOPRA
        cov_dict = dict()
        for coup in var_couples + covar_couples: 
            tensor = np.outer(eigvecs[coup[0],1:], eigvecs[coup[1],1:]) 
            tensor = tensor*coeff #prodotto entry per entry
            cov_dict[coup] = np.einsum('lm,lm', tensor, scalar_product) #equiv a thcov[i,j] = np.sum(tensor*scalar_product)
        
        #Center node is the first element: thus row index in corr is the center node
        for coup in covar_couples:
            corr[coup[0],coup[1]] = cov_dict[coup] / (cov_dict[(coup[0],coup[0])] * cov_dict[(coup[1],coup[1])]) ** 0.5

# =============================================================================
#     LOOP ENDS: CORRELATION MATRIX IS READY
# =============================================================================

        # #TAKE THE AVERAGE OF THE 2 COMPUTATIONS OF EACH CORRELATION 
        # thcorr = (thcorr + thcorr.T)/2
        # np.fill_diagonal(thcorr,1)
           
    return corr



def k_shortest_paths(G, source, target, k):
    
    paths = nx.all_simple_edge_paths(G, source, target, cutoff=k)
    return paths
    
         
def single_spectralflux_correlation(G, s, k_neig):

# =============================================================================
#     Structural part
# =============================================================================
    
    N = len(G.nodes)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)

# =============================================================================
#     Dynamical part
# =============================================================================
    
    p = scipy.linalg.lstsq(L, s, lapack_driver='gelsy')[0]
    
# =============================================================================
#     Correlation computation
# =============================================================================    
    
    #Correlation array inizialization
    corr = np.zeros((N,N))
    
    #Coefficients = 1/l+m
    #Since symmetric, computed only for the upper triangular part, then symmetrized
    coeff = np.zeros((N-1,N-1)) #N-1 since no ker
    for l in range(N-1):
        coeff[l,l:] = [1/(eigvals[l+1] + mu) for mu in eigvals[l+1:]]
    coeff = coeff + coeff.T
    coeff[np.diag_indices_from(coeff)] /= 2
    
    # =============================================================================
    #         MODIFIED ADJ MATRIX
    # =============================================================================
    
    for node1 in G.nodes():
        
        neig_list = list(nx.single_source_shortest_path_length(G, node1, cutoff=k_neig).keys())
        neig_list.remove(node1)    
        
        for node2 in neig_list:
            
            #The fluctuation of each link are proportional to its weight
            #Thus to make the links in local list not to fluctuate, we set their weights to zero
            A_tilde = np.copy(A)
        
            for path in k_shortest_paths(G, node1, node2, k_neig):
                for edge in path:
                    A_tilde[edge[0],edge[1]] = 0
                    A_tilde[edge[1],edge[0]] = 0

    # =============================================================================
    #         METRIC MATRIX C_kh
    # =============================================================================
    
            metric = np.zeros((N,N))
            idx = np.nonzero(A_tilde) #indices where A is nonzero, to speed up (ci mette 15 secondi in meno così)
            
            metric[idx] = - ( A_tilde[idx] * (p[idx[0]] - p[idx[1]]) ) ** 2
            #metric[idx] = - (np.sign(A_tilde[idx]) * (p[idx[0]] - p[idx[1]]) )**2 
            
            metric_diag = - np.sum(metric, axis=1) #axis=0 same, is symmetric
            np.fill_diagonal(metric, metric_diag) #np.allclose(np.sum(metric,axis=0),0) o axis=1 should be 0 
            
    # =============================================================================
    #         SCALAR PRODUCT MATRIX
    # =============================================================================
    
            scalar_product = np.tensordot(metric,eigvecs[:,1:],axes=1)
            scalar_product = np.tensordot(eigvecs[:,1:].T,scalar_product,axes=1)
    
    # =============================================================================
    #         CORRELATION
    # =============================================================================
    
            couples = [(node1,node1), (node2,node2), (node1,node2)]
    
            #OUTER PRODUCT BETWEEN COEFF 1/l+m E LA SCALAR PRODUCT MATRIX DI SOPRA
            cov_dict = dict()
            for coup in couples: 
                tensor = np.outer(eigvecs[coup[0],1:], eigvecs[coup[1],1:]) 
                tensor = tensor*coeff #prodotto entry per entry
                cov_dict[coup] = np.einsum('lm,lm', tensor, scalar_product) 
            
            corr[node1,node2] = cov_dict[(node1,node2)] / (cov_dict[(node1,node1)] * cov_dict[(node2,node2)]) ** 0.5

    np.fill_diagonal(corr, 1)
    
    return corr



# Versione solo nodi vicini di spectral 10 e 21 sopra
def multifluc_spectralflux_correlation(G, s, k_neig):

# =============================================================================
#     Structural part
# =============================================================================
    
    N = len(G.nodes)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)

# =============================================================================
#     Dynamical part
# =============================================================================
    
    p = scipy.linalg.lstsq(L, s, lapack_driver='gelsy')[0]
    
# =============================================================================
#     Correlation computation
# =============================================================================    
    
    #Correlation array inizialization
    corr = np.zeros((N,N))

    
    #Coefficients = 1/l+m
    #Since symmetric, computed only for the upper triangular part, then symmetrized
    coeff = np.zeros((N-1,N-1)) #N-1 since no ker
    for l in range(N-1):
        coeff[l,l:] = [1/(eigvals[l+1] + mu) for mu in eigvals[l+1:]]
    coeff = coeff + coeff.T
    coeff[np.diag_indices_from(coeff)] /= 2

    # =============================================================================
    #         MODIFIED ADJ MATRIX
    # =============================================================================
        
    for center in G.nodes():
    
        neig_list = list(nx.single_source_shortest_path_length(G, center, cutoff=k_neig).keys())
        #neig_list = list(G.neighbors(center)) # no center in list
        
        for eval_node in neig_list:
            
                center_eval_distance = nx.shortest_path_length(G, center, eval_node)
                
                # Per questo center, loro non possono fluc
                no_fluc_nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=center_eval_distance-1).keys())
                
                # Tutti i vicini di eval possono fluttuare, tranne quelli che si trovano entro oppure sulla sfera k_neig-1 dal centro
                A_tilde = np.zeros((N,N), dtype=int)
                
                A_tilde[eval_node,:] = A[eval_node,:]
                A_tilde[:,eval_node] = A[:,eval_node]
                
                A_tilde[no_fluc_nodes,:] = 0
                A_tilde[:,no_fluc_nodes] = 0


                if np.allclose(A_tilde,0):
                    corr[center,eval_node] = 0
                    continue
                
        # =============================================================================
        #         METRIC MATRIX C_kh
        # =============================================================================
        
                metric = np.zeros((N,N))
                idx = np.nonzero(A_tilde) #indices where A is nonzero, to speed up (ci mette 15 secondi in meno così)
                metric[idx] = - ( A_tilde[idx] * (p[idx[0]] - p[idx[1]]) ) ** 2
                metric_diag = - np.sum(metric, axis=1) #axis=0 same, is symmetric
                np.fill_diagonal(metric, metric_diag) #np.allclose(np.sum(metric,axis=0),0) o axis=1 should be 0     
        
        # =============================================================================
        #         SCALAR PRODUCT MATRIX
        # =============================================================================
        
                scalar_product = np.tensordot(metric,eigvecs[:,1:],axes=1)
                scalar_product = np.tensordot(eigvecs[:,1:].T,scalar_product,axes=1)

        # =============================================================================
        #         CORRELATION
        # =============================================================================
        
                couples = [(center,center), (eval_node,eval_node), (center,eval_node)]
                
                #OUTER PRODUCT BETWEEN COEFF 1/l+m E LA SCALAR PRODUCT MATRIX DI SOPRA
                cov_dict = dict()
                for coup in couples: 
                    tensor = np.outer(eigvecs[coup[0],1:], eigvecs[coup[1],1:]) 
                    tensor = tensor*coeff #prodotto entry per entry
                    cov_dict[coup] = np.einsum('lm,lm', tensor, scalar_product) #equiv a thcov[i,j] = np.sum(tensor*scalar_product)
                
                corr[center,eval_node] = cov_dict[(center,eval_node)] / (cov_dict[(center,center)] * cov_dict[(eval_node,eval_node)]) ** 0.5
    
    np.fill_diagonal(corr, 1)
    
    return corr



# versione solo vicini di sing1 e 2
def multiflucbi_spectralflux_correlation(G, s, k_neig):

# =============================================================================
#     Structural part
# =============================================================================
    
    N = len(G.nodes)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G, weight='weight')
    A = A.toarray()

    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray()
    
    if not np.allclose(L,L.T):
        raise Exception('Warning: eigh with non symmetric matrix')
    eigvals, eigvecs = scipy.linalg.eigh(L)

# =============================================================================
#     Dynamical part
# =============================================================================
    
    p = scipy.linalg.lstsq(L, s, lapack_driver='gelsy')[0]
    
# =============================================================================
#     Correlation computation
# =============================================================================    
    
    #Correlation array inizialization
    corr = np.zeros((N,N))
    
    #Coefficients = 1/l+m
    #Since symmetric, computed only for the upper triangular part, then symmetrized
    coeff = np.zeros((N-1,N-1)) #N-1 since no ker
    for l in range(N-1):
        coeff[l,l:] = [1/(eigvals[l+1] + mu) for mu in eigvals[l+1:]]
    coeff = coeff + coeff.T
    coeff[np.diag_indices_from(coeff)] /= 2

    # =============================================================================
    #         MODIFIED ADJ MATRIX
    # =============================================================================
        
    for center in G.nodes():
        
        neig_list = list(nx.single_source_shortest_path_length(G, center, cutoff=k_neig).keys())
        #neig_list = list(G.neighbors(center)) # no center in list
        
        for eval_node in neig_list: #G.nodes()
            
                center_eval_distance = nx.shortest_path_length(G, center, eval_node)
                #local_nodes = list(nx.single_source_shortest_path_length(G, center, cutoff=distance).keys())
                
                center_fluc_nodes = list(G.neighbors(center)) 
                eval_fluc_nodes = list(G.neighbors(eval_node))
                
                for node in center_fluc_nodes:
                    distance = nx.shortest_path_length(G, node, eval_node)
                    if distance < center_eval_distance:
                        center_fluc_nodes.remove(node)
                
                for node in eval_fluc_nodes:
                    distance = nx.shortest_path_length(G, node, center)
                    if distance < center_eval_distance:
                        eval_fluc_nodes.remove(node)
    
                A_tilde = np.zeros((N,N), dtype=int)
                
                for node in center_fluc_nodes:
                    A_tilde[center,node] = A[center,node]
                    A_tilde[node,center] = A[node,center]
                    
                for node in eval_fluc_nodes:
                    A_tilde[eval_node,node] = A[eval_node,node]
                    A_tilde[node,eval_node] = A[node,eval_node]
                    
                    
                if np.allclose(A_tilde,0):
                    corr[center,eval_node] = 0
                    continue
                
        # =============================================================================
        #         METRIC MATRIX C_kh
        # =============================================================================
        
                metric = np.zeros((N,N))
                idx = np.nonzero(A_tilde) #indices where A is nonzero, to speed up (ci mette 15 secondi in meno così)
                metric[idx] = - ( A_tilde[idx] * (p[idx[0]] - p[idx[1]]) ) ** 2
                metric_diag = - np.sum(metric, axis=1) #axis=0 same, is symmetric
                np.fill_diagonal(metric, metric_diag) #np.allclose(np.sum(metric,axis=0),0) o axis=1 should be 0     
        
        # =============================================================================
        #         SCALAR PRODUCT MATRIX
        # =============================================================================
        
                scalar_product = np.tensordot(metric,eigvecs[:,1:],axes=1)
                scalar_product = np.tensordot(eigvecs[:,1:].T,scalar_product,axes=1)

        # =============================================================================
        #         CORRELATION
        # =============================================================================

                couples = [(center,center), (eval_node,eval_node), (center,eval_node)]
        
                #OUTER PRODUCT BETWEEN COEFF 1/l+m E LA SCALAR PRODUCT MATRIX DI SOPRA
                cov_dict = dict()
                for coup in couples: 
                    tensor = np.outer(eigvecs[coup[0],1:], eigvecs[coup[1],1:]) 
                    tensor = tensor*coeff #prodotto entry per entry
                    cov_dict[coup] = np.einsum('lm,lm', tensor, scalar_product) 
                
                corr[center,eval_node] = cov_dict[(center,eval_node)] / (cov_dict[(center,center)] * cov_dict[(eval_node,eval_node)]) ** 0.5
                
        np.fill_diagonal(corr, 1)
    
    return corr









