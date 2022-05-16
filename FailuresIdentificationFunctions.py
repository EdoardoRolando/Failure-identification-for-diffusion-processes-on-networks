#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:35:56 2022

@author: edo
"""

import numpy as np
import scipy.linalg
import networkx as nx
from scipy import special


#################################################################################################################################################
########################################## FAILURE: NETWORK MODIFICATION FUNCTION #################################################################################
#################################################################################################################################################



def network_response(L, demands, solution, broken_edge, timing): 
    
    N = len(demands)
    
    #EDGE BREAKING
    #broken_edge_weight = G.edges[broken_edge[0], broken_edge[1]]['weight'] 
    broken_edge_weight = -L[broken_edge[0], broken_edge[1]]
    #print(broken_edge_weight)
    failure_factor = 0.9 #rng.uniform(7,10) / 10 #leak percentage
    broken_edge_reduction = round(broken_edge_weight * failure_factor, 2)
    
    #LAPLACIAN MODIFICATION
    newL = L.astype(dtype=float, copy=True)
    
    newL[broken_edge[0], broken_edge[1]] += broken_edge_reduction 
    newL[broken_edge[1], broken_edge[0]] += broken_edge_reduction 
    newL[broken_edge[0], broken_edge[0]] -= broken_edge_reduction 
    newL[broken_edge[1], broken_edge[1]] -= broken_edge_reduction 
    
    #solution = scipy.linalg.lstsq(L, demands, lapack_driver='gelsy')[0] 
    new_solution = scipy.linalg.lstsq(newL, demands, lapack_driver='gelsy')[0]
    
    # Since lstsq sometimes does the monello
    newp_sum = np.sum(new_solution)
    if not np.isclose(newp_sum, 0):
        new_solution = new_solution - newp_sum / N

    if timing == 'stationary':
        
        # Stationary solutions difference
        p_diff = new_solution - solution

        # Return p_diff, the only thing that is observable after the network changes
        return p_diff
        
    if timing == 'transient':
        
        #Spectrum of the new Laplacian, needed for the time evolution
        neweigvals, neweigvecs = scipy.linalg.eigh(newL, driver='evd') #default is "evr", seems to be slower 
        
        #Coefficients for the time evolution
        c = scipy.linalg.solve( neweigvecs, solution - new_solution )
        if not np.isclose(c[0], 0):
            print('c[0] non è zero', broken_edge)

        #tau[0] is the characterictic decay time for the eigenvector N-1 (highest), and so on: the last one would be infinite, we do not put it 
        tau = 1/neweigvals[1::][::-1]
        
        transient_fraction = [0.01, 0.1, 0.5] #we observe the solution of the system at 1/10 of the total time needed to reach the new stationary state
        obs_time = [tau[N-2] * i for i in transient_fraction]  #obs_t = np.linspace(0, tau[N-2], 10)
        #number_decayed_eig = np.count_nonzero(tau < obs_time) #number of decayed eigs before obs_t

        #TEMPORAL EVOLUTION
        #each column are all the pressure at a time: since the last tau would be infinite, the last column is the new stationary solution np.allclose(ptdata[:,N-1],new_solution) would be 0 
        ptdata = np.zeros((N,len(obs_time)+1)) # +1 to append the new stationary at the end
        for i,t in enumerate(obs_time):
            pt = new_solution + np.matmul( neweigvecs, c * np.exp( - neweigvals * t )) 
            ptdata[:,i] = pt - solution
        ptdata[:,len(obs_time)] = new_solution - solution
        
        return ptdata
        
    else: 
        raise Exception('Wrong input')
    


#################################################################################################################################################
########################################## FLUCTUATION: NETWORK MODIFICATION FUNCTION #################################################################################
#################################################################################################################################################



def fluctuations_network_response(A, dt_L, p_stat, dt_demands, fluc_edge, random_var, dt, T, n):
    
    N = len(dt_demands)

    #SIMULATION VECTORS: that will contain all succesive values of our process during the simulation
    x = np.zeros((N,n)) #k-th column are all the p at the k-th time step
    
    # Initial condition
    x[:,0] = p_stat 
    
    # Fluctuating edge weight
    edge_weight = A[fluc_edge[0],fluc_edge[1]]
  
    for i in range(n-1):
        
        #if i%3 == 0: # per il moltiplicativo 
        edge_fluctuation = random_var[i] * edge_weight
    
        #x[:,i+1] = x[:,i] - dt*L@x[:,i] + dt*s - sqrtdt*delta_L@p_stat
        #rumore moltiplicativo facendo flutt il network completo, la soluzione esplode (se tieni deltaL fisso per, per esempio, 10 time steps)
        # x[:,i+1] = x[:,i] - dt * L @ x[:,i] + dt * demands - dt * delta_L @ x[:,i] 
        
        x[:,i+1] = x[:,i] - dt_L @ x[:,i] + dt_demands
        term = dt * edge_fluctuation * ( - x[fluc_edge[0],i] + x[fluc_edge[1],i] )
        x[fluc_edge[0],i+1] -= term
        x[fluc_edge[1],i+1] += term
   
    ini = 0
    comp_cov = np.cov( x[:,ini:] )

    return comp_cov


#################################################################################################################################################
########################################## FAILURE LOCALIZATION FUNCTION #################################################################################
#################################################################################################################################################


def order_of_magnitude(n):
    return int( np.floor( np.log10(abs(n)) ))


def likelihood_function(x):
    
    if 0 <= x <= 0.5:
        pos_like = 0
        neg_like = 1-2*x
        dip_like = 2*x
        
    elif 0.5 < x <= 1:
        pos_like = 2*x-1
        neg_like = 0
        dip_like = 2-2*x
        
    else:
        raise Exception('Ratio is not between 0 and 1')
    
    return pos_like, neg_like, dip_like


def adaptive_likelihood_function(x, m):
    
    if m != 0:
        
        arg = -np.exp(-1/m)/m
        lambert = special.lambertw(arg, k=0, tol=1e-8)
        
        if 0 <= x <= 0.5:
            pos_like = 0
            neg_like = np.exp( (1/m + lambert)*(-2*x) )
            dip_like = 1 - np.exp( (1/m + lambert)*(-2*x) )
            
        elif 0.5 < x <= 1:
            pos_like = np.exp( (1/m + lambert)*(-2*(-x+1)) )
            neg_like = 0
            dip_like = 1 - np.exp( (1/m + lambert)*(-2*(-x+1)) )
            
        else:
            raise Exception('Ratio is not between 0 and 1')
        
        return pos_like, neg_like, dip_like
    
    else: 
        
        if 0 <= x <= 0.5:
            pos_like = 0
            neg_like = 0
            dip_like = 1 
            
        elif 0.5 < x <= 1:
            pos_like = 0
            neg_like = 0
            dip_like = 1 
            
        else:
            raise Exception('Ratio is not between 0 and 1')
        
        return pos_like, neg_like, dip_like
        


def failure_localization(G, p_diff_list, demands, solution, Qdict, R, tested_edges, labels, sensors, G_edges_boundary):
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    
    N = len(L)
    
    # Results lists
    results = []
    confidence = []
    
    # Clusters list creation
    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
        else: print('Label problem')
            
    # # Clusters list creation
    # clusters = []
    # for lab in np.unique(labels):
    #     arr = np.where(labels==lab)[0]
    #     clusters.append(arr.tolist())
    
    
    # =============================================================================
    #     LIKELIHOOD FUNCTION BASED ON NUMBER OF INTERNAL EDGES
    # =============================================================================
    
    internal_edges_counter = 0
    internal_edges = []
    for clus in clusters:
        edges = nx.subgraph(G, clus).edges()
        internal_edges_counter += len(edges)
        internal_edges.append(list(edges))
        
    #dipoles_number = len(G.edges()) - monopoles_number
    pos_monopoles_ratio = 0.5*internal_edges_counter/len(G.edges)
    #print(pos_monopoles_ratio)
    

    # =============================================================================
    #     FAILURE LOCALIZATION LOOP
    # =============================================================================

    for broken_edge_num,broken_edge in enumerate(tested_edges): 
        
        # To check if we sense the correct cluster
        clus1 = labels[broken_edge[0]]
        clus2 = labels[broken_edge[1]]
        sens1 = sensors[clus1]
        sens2 = sensors[clus2]
        
        
        # =============================================================================
        #         DIFFERENCE OF P
        # =============================================================================
            
        # Signal difference on the complete network (the only observable we have)
        #p_diff = network_response(L, demands, solution, broken_edge, timing, obs_time)
        p_diff = p_diff_list[broken_edge][:,3] # ultime che è la nuova stat
        
        
        # =============================================================================
        #         ROUNDING TO DESIRED SENSITIVITY
        # =============================================================================
                
        # Example: order of magnitude is -2, relative sensitivity is 1, sensitivity is 3,
        # i.e. we need to measure 10-3 to be sensitive to 1 part out of 10 
        relative_sensitivity = 10
        sensitivity = relative_sensitivity - order_of_magnitude(np.average(abs(solution)))  
        p_diff = np.round(p_diff, sensitivity)
        
        
        # =============================================================================
        #         P_DIFF DICTIONARY
        # =============================================================================
        
        # Dictionary with sensor:p_diff
        p_diff_sensors = dict(zip(sensors, p_diff[sensors]))
        # Sort signal on sensors according to the absolute value (descending order)
        p_diff_sensors = dict(sorted(p_diff_sensors.items(), key=lambda item: abs(item[1]), reverse=True))
        
        # If we do not have enough sensibility to measure any p_diff on the sensors
        # This happens if we have low sensitivity and weak leaks
        if np.allclose(list(p_diff_sensors.values()),0):
            results.append( 0 )
            # Go to the next broken_edge
            continue
        
        # Sensors with a positive and negative difference
        pos_sensors = [key for (key, value) in p_diff_sensors.items() if value > 0 ]
        neg_sensors = [key for (key, value) in p_diff_sensors.items() if value < 0 ]
              
        # =============================================================================
        #         CHECK IF WE ARE IN THE 'DEGENERATE CASE'
        # =============================================================================
        
        # 2) OR DUE TO OUR CHOICE OF SUM(P)=0 (WITH OTHER CHOICES THE SAME BUT TRANSLATED, i.e. inf our choice is node x is ground
        # we would have all nodes to 0 a part from 1, but still it gives problems
        
        # Degenerate case: in p_diff we have the same value for all ndoes except one of the two nodes of broken_edge
        # Thusif we detect it, it means that we have a sensor on this node and thus we know directly the possible edges
        # If we don't, we have only sensors with the same identical p_diff --> no info
        
        p_diff_sensors_values = list(p_diff_sensors.values())
        
        # If they are all the same
        if np.allclose(p_diff_sensors_values, p_diff_sensors_values[0]): # since we have sorted it 
            results.append( 0 )
            confidence.append( 0 )
            continue
        
        # If they are equal to the second one (and the first one is different otherwise we would have exit at the if before)
        if np.allclose(p_diff_sensors_values[1:], p_diff_sensors_values[1]):

            # We are sure it is this one
            broken_node, broken_node_signal = list(p_diff_sensors.items())[0]
            possible_edges = [edge for edge in list( G.edges(broken_node) ) if Qdict[edge] * broken_node_signal > 0]
            not_possible_edges = [edge for edge in list( G.edges(broken_node) ) if Qdict[edge] * broken_node_signal <= 0]
            
            if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                results.append( len(possible_edges) ) 
                # Like below in confidence there is the 'probability that the group of nodes choosen is correct
                # and we are sure about this group of nodes, regardless they are part of a cluster or between two clusters
                confidence.append( 1 ) 
            
            # Actually it does not work always, sometimes (rare), the broken_node (sensor with different p_diff) is not a broken_one, but adjacent
            else:
                results.append( 0 )
                confidence.append( 0 )
            
            # Go to the next broken_edge
            continue
        
        # =============================================================================
        #         P_DIFF DICTIONARY WITH REPR_FACTOR
        # =============================================================================
        
        # To convert the solution from the complete network to the reduced network
        # Since we multiply p_diff with repr_factor, the less sensitive we are in measuring p_diff (more rounding) 
        # the more this error will propagate to p_diff_sensors, generating problems 
        
        repr_factor = [len(clus) for clus in clusters]
        p_diff_sensors = dict(zip(sensors, p_diff[sensors] * repr_factor))
 
        p_diff_sensors = dict(sorted(p_diff_sensors.items(), key=lambda item: abs(item[1]), reverse=True))
        
        # =============================================================================
        # # =============================================================================
        # #         DIPOLE
        # # =============================================================================      
        # =============================================================================

        #All edges (tuple) between positive and negative sensors (positive first): not known order, decided by edge_boundary()
        if len(pos_sensors) > 0 and  len(neg_sensors) > 0:
            if R.has_edge(pos_sensors[0], neg_sensors[0]):
                dipoles_edges = [(pos_sensors[0], neg_sensors[0])]
            else:
                dipoles_edges = list(nx.edge_boundary(R, pos_sensors, neg_sensors))
        else:
            dipoles_edges = []
        
        #dipoles_edges = list(nx.edge_boundary(R, pos_sensors, neg_sensors))
        
        if dipoles_edges:
            
            # Sort the dipole_edges according to the greatest difference/gap 
            dipoles_gap = {edge: p_diff_sensors[edge[0]] - p_diff_sensors[edge[1]] for edge in dipoles_edges}
            dipoles_gap = dict(sorted(dipoles_gap.items(), key=lambda item: abs(item[1]), reverse=True))
            
            # Take the unique values (since they might be degenerate due to p_diff sensibility) in decreasing order
            gaps_values = sorted(set(dipoles_gap.values()), key=abs, reverse=True) 
     
            # Take the biggest value and all the dipoles corresponding to that value
            max_gap = gaps_values[0] 
            max_dipoles = [k for k,v in dipoles_gap.items() if v == max_gap]
            
            # Each edge between max_dipoles that is compatible with the observed p_diff is a possible candidate
            # Thus for each dipole, take the corresponding clusters, the edges between them, check if the fluxes are compatible
            # like for the monopole, if they are compatible, they are candidate link
            #max_dipoles_edges = [list( nx.edge_boundary(G, clusters[labels[dipole[0]]], clusters[labels[dipole[1]]]) ) for dipole in max_dipoles]
            max_dipoles_edges = [G_edges_boundary[ labels[dipole[0]], labels[dipole[1]] ] for dipole in max_dipoles]
            
            max_dipoles_edges = [edge for sublist in max_dipoles_edges for edge in sublist] # to flatten the list
            
            dipole_possible_edges = max_dipoles_edges
            
        else:
            
            results.append( 0 )
            confidence.append( 0 )
            continue
    
        
        # =============================================================================
        # # =============================================================================
        # #         MONOPOLE
        # # =============================================================================       
        # =============================================================================

        # if len(max_dipoles) > 1:
        #     #for max_dipole in max_dipoles:
        #         #print(broken_edge_num,broken_edge,max_dipole,dipoles_gap[max_dipole])
        #     print(broken_edge,'Più dipoli degeneri!')

        # From now on continue as if there was just one dipole
        pos_monopole = max_dipoles[0][0]
        neg_monopole = max_dipoles[0][1]
        
        pos_signal = p_diff_sensors[pos_monopole]
        neg_signal = p_diff_sensors[neg_monopole]
        
        # Check if there are internal edges in the cluster represented by max_sensor that can cause this polarization (i.e. this sign of p_diff on max_sensor)
        # If there are, those are the links we identify as possible failures 
        # If there are not, according to out criterio we identified it as a monopole, but acutally it cannot be: thus it's probably a weak dipole --> no_monopole_edges = True
        # NOTE: we can infer for sure whether an internal link failure causes an increase or decrease of the monopole sensor signal only if that link has the sensor as one of the two nodes
        # In all the other cases, we should do more complex checks on the local topology 
            
        # =============================================================================
        #         POS MONOPOLE
        # =============================================================================
               
        # Nodes belonging to the cluster of pos_monopole
        nodes = clusters[labels[pos_monopole]]
        
        # Take the edges that have max_sens as one of the nodes (no G.neig(max_sens) because there might be also nodes outside the cluster)
        # As written here is ugly, but to evaluate the sign of the fluxes we need max_sens as first node in the tuples
        # Nodes belonging to the cluster of max_sens and that are neighbors of max_sens
        sensor_neighbors = list( set(G.neighbors(pos_monopole)) & set(nodes) )
        
        # Considering fluxes like Qdict[(max_sens, other node)] --> from max_sens to other node:
        # If the flux is positive, it is outgoing from max_sensor: in this case, if this link breaks, we expect a positive p_diff
        # If the flux is negative, it is incoming to max_sensor: in this case, if this link breaks, we expect a negative p_diff
        # Thus, the considered link is a possible failure if the flux sign is the same as the sign of max_signal
        edges_not_possible = [(pos_monopole,sens_neig) for sens_neig in sensor_neighbors if Qdict[(pos_monopole,sens_neig)] * pos_signal < 0]

        # Edges internal to the cluster of max_sens
        pos_possible_edges = internal_edges[labels[pos_monopole]] #list(nx.subgraph(G, nodes).edges())
        pos_possible_edges = [edge for edge in pos_possible_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]

        # Just to check if the above part of monopole possible edges is correct
        if sens1 == sens2 == pos_monopole and broken_edge not in pos_possible_edges and (broken_edge[1],broken_edge[0]) not in pos_possible_edges:
            print('\n Non dovresti finire qui, hai tolto edges che erano possibili per il monopole pos', broken_edge, edges_not_possible)
        
        # =============================================================================
        #         NEG MONOPOLE
        # =============================================================================
               

        nodes = clusters[labels[neg_monopole]]
        sensor_neighbors = list( set(G.neighbors(neg_monopole)) & set(nodes) )
        edges_not_possible = [(neg_monopole,sens_neig) for sens_neig in sensor_neighbors if Qdict[(neg_monopole,sens_neig)] * neg_signal < 0]
        
        neg_possible_edges = internal_edges[labels[neg_monopole]] #list(nx.subgraph(G, nodes).edges())
        neg_possible_edges = [edge for edge in neg_possible_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]

        # Just to check if the above part of monopole possible edges is correct
        if sens1 == sens2 == neg_monopole and broken_edge not in neg_possible_edges and (broken_edge[1],broken_edge[0]) not in neg_possible_edges:
            print('\n Non dovresti finire qui, hai tolto edges che erano possibili per il monopole neg', broken_edge, edges_not_possible)

     
        # =============================================================================
        # # =============================================================================
        # #         CHECK
        # # =============================================================================      
        # =============================================================================
        
        pos_ratio = pos_signal / (pos_signal + abs(neg_signal))
        
        #pos_like, neg_like, dip_like = likelihood_function(pos_ratio)
        pos_like, neg_like, dip_like = adaptive_likelihood_function(pos_ratio,pos_monopoles_ratio)
        
        if not 0 <= pos_like <= 1:
            print('prob pos', pos_like, pos_ratio)
            
        if not 0 <= neg_like <= 1:
            print('prob pos')
            
        if not 0 <= dip_like <= 1:
            print('prob pos')
        
        # Only if there are possible edges, if there are not results to 0
        tot = pos_like * len(pos_possible_edges) + neg_like * len(neg_possible_edges) + dip_like * len(dipole_possible_edges)
        
        if not np.isclose(tot,0):
            
            # Bayes update
            pos_like = (pos_like * len(pos_possible_edges)) / tot
            neg_like = (neg_like * len(neg_possible_edges)) / tot
            dip_like = (dip_like * len(dipole_possible_edges)) / tot
            
        else:
            results.append( 0 )
            confidence.append( 0 )
            continue

        # ============================================================================================
        # It choses the one with greater confidence
        # ============================================================================================
                
        mychoice = np.argmax((pos_like, neg_like, dip_like))
                
        if mychoice == 0:
            possible_edges = pos_possible_edges
            mychoice_confidence = pos_like
            
        if mychoice == 1:
            possible_edges = neg_possible_edges
            mychoice_confidence = neg_like
            
        if mychoice == 2:
            possible_edges = dipole_possible_edges
            mychoice_confidence = dip_like
      
        # Now check if our choice is correct
        if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
            results.append( len(possible_edges) ) 
            confidence.append( mychoice_confidence )
            #confidence.append( mychoice_confidence / len(possible_edges) )
            
        else:
            results.append( 0 )
            confidence.append( 0 )
    
        # =============================================================================
        # Correct identification if either in pos, neg or dip
        # =============================================================================
            
        # if broken_edge in pos_possible_edges or (broken_edge[1],broken_edge[0]) in pos_possible_edges:
        #     results.append( len(pos_possible_edges) )
        #     confidence.append( pos_like )
        #     #confidence.append( pos_like / len(pos_possible_edges) )
        
        # if broken_edge in neg_possible_edges or (broken_edge[1],broken_edge[0]) in neg_possible_edges:
        #     results.append( len(neg_possible_edges) )
        #     confidence.append( neg_like )
        #     #confidence.append( neg_like / len(neg_possible_edges) )
            
        # if broken_edge in dipole_possible_edges or (broken_edge[1],broken_edge[0]) in dipole_possible_edges:
        #     results.append( len(dipole_possible_edges) )
        #     confidence.append( dip_like )
        #     #confidence.append( dip_like / len(dipole_possible_edges) )
        # else:
        #     results.append( 0 )
        #     confidence.append( 0 )
    
            
    return results, confidence



def failure_localization_original(G, p_diff_list, monopole_threshold, demands, solution, Qdict, R, tested_edges, labels, sensors, timing, obs_time):
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    
    # Results lists
    cases_results = []
    results = []
    
    # Clusters list creation
    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
    
# =============================================================================
#     FAILURE LOCALIZATION LOOP
# =============================================================================

    for broken_edge_num,broken_edge in enumerate(tested_edges): 
        
        #print(broken_edge)
        
        # To check if we sense the correct cluster
        clus1 = labels[broken_edge[0]]
        clus2 = labels[broken_edge[1]]
        sens1 = sensors[clus1]
        sens2 = sensors[clus2]
        
    # =============================================================================
    #         EDGE BREAKING  
    # =============================================================================
        
        # Signal difference on the complete network (the only observable we have)
        #p_diff = network_response(L, demands, solution, broken_edge, timing, obs_time)
        p_diff = p_diff_list[broken_edge]
        
        ###!!!! ROUNDINGSSSS
        p_diff = np.round(p_diff,5)
        
        # To convert the solution from the complete network to the reduced network
        # Since we multiply p_diff with repr_factor, the less sensitive we are in measuring p_diff (more rounding) 
        # the more this error will propagate to p_diff_sensors, generating problems 
        repr_factor = [len(clus) for clus in clusters]
        p_diff_sensors = dict(zip(sensors, repr_factor * p_diff[sensors]))
        #p_diff_sensors = dict(zip(sensors, p_diff[sensors]))
        
        # If we do not have enough sensibility to measure any p_diff on the sensors
        # This happens if we have low sensitivity and weak leaks
        if all(val == 0 for val in p_diff_sensors.values()):
            results.append( 0 )
            cases_results.append('no diff')
            continue
        
        # Sort signal on sensors according to the absolute value (descending order)
        p_diff_sensors = dict(sorted(p_diff_sensors.items(), key=lambda item: abs(item[1]), reverse=True))
        
    
    # =============================================================================
    # # =============================================================================
    # #         MONOPOLE
    # # =============================================================================       
    # =============================================================================

        # First we determine if twe have a net monopole or not, according to the highest sensors signal
        # Take the unique values (since they might be degenerate due to p_diff sensibility) in decreasing order
        signal_values = sorted(set(p_diff_sensors.values()), key=abs, reverse=True) 
        
        max_signal = signal_values[0] 
        max_sensors = [k for k,v in p_diff_sensors.items() if v == max_signal] # It should be difficult that there are more than one 
        
        # 1) The second one is choosen as the highest among all with a different sign (or eventually zero becouse of rounding) than max_signal
        sec_max_signal = next(( signal for signal in signal_values if signal * max_signal <= 0), 0)
        sec_max_sensors = [k for k,v in p_diff_sensors.items() if v == sec_max_signal]

        # 2) The second one is choosen as the highest among max_sensors neighbors with a different sign (or eventually zero becouse of rounding) than max_signal
        # max_sensors_neig = []
        # for max_sens in max_sensors:
        #     max_sensors_neig.extend( list(R.neighbors(max_sens)) )
        # max_sensors_neig_dict = {neig:p_diff_sensors[neig] for neig in max_sensors_neig if p_diff_sensors[neig] * max_signal < 0}
        # if max_sensors_neig_dict:
        #     sec_max_signal = max(max_sensors_neig_dict.values(), key=abs)
        # else: 
        #     sec_max_signal = 10000000
        
        # 3) Below if we want to take the second highest signal regardless of the sign
        # sec_max_signal = signal_values[1] 
        # sec_max_sensors = [k for k,v in p_diff_sensors.items() if v == sec_max_signal]
        
     
        # =============================================================================
        #         Net monopole
        # =============================================================================
        
        # Boolean variable: we go to the dipole case either if the first one is not bigger than monopole_threshold times the second one 
        # (with a different sign, eventually 0), or if it is satisfied but the candidate monopole has no possible edges
        no_monopole_edges = False
               
        if abs(max_signal) > monopole_threshold * abs(sec_max_signal):
            
            ###!!! Se cambiano le demands nel tempo possono cambiare anche la direzione del flusso!!!   QUINDI ATTENDO SE LE PRENDI DA QDICT O DA EDGES ATTRIBUTES
            # Cosa succede se durante un transiente cambia la domanda in modo tale che il flusso si inverte?
            # Il dipolo che si stava creando cambia segno????
            #PER ORA LA DEMAND NON CAMBIA DURANTE UN TRANSIENTE, O ALMENO NON PRIMA DEL (TENTATIVO DI) IDENTIFICAZIONE 
            
            # Check if there are internal edges in the cluster represented by max_sensor that can cause this polarization (i.e. this sign of p_diff on max_sensor)
            # If there are, those are the links we identify as possible failures 
            # If there are not, according to out criterio we identified it as a monopole, but acutally it cannot be: thus it's probably a weak dipole --> no_monopole_edges = True
            # NOTE: we can infer for sure whether an internal link failure causes an increase or decrease of the monopole sensor signal only if that link has the sensor as one of the two nodes
            # In all the other cases, we should do more complex checks on the local topology 
                
            possible_edges = []
            for max_sens in max_sensors:
                
                # Nodes belonging to the cluster of max_sens
                nodes = clusters[labels[max_sens]]
                
                # Take the edges that have max_sens as one of the nodes (no G.neig(max_sens) because there might be also nodes outside the cluster)
                # As written here is ugly, but to evaluate the sign of the fluxes we need max_sens as first node in the tuples
                # Nodes belonging to the cluster of max_sens and that are neighbors of max_sens
                sensor_neighbors = list( set(G.neighbors(max_sens)) & set(nodes) )
                
                # Considering fluxes like Qdict[(max_sens, other node)] --> from max_sens to other node:
                # If the flux is positive, it is outgoing from max_sensor: in this case, if this link breaks, we expect a positive p_diff
                # If the flux is negative, it is incoming to max_sensor: in this case, if this link breaks, we expect a negative p_diff
                # Thus, the considered link is a possible failure if the flux sign is the same as the sign of max_signal
                edges_not_possible = [(max_sens,sens_neig) for sens_neig in sensor_neighbors if Qdict[(max_sens,sens_neig)] * max_signal < 0]

                # Edges internal to the cluster of max_sens
                internal_edges = list(nx.subgraph(G, nodes).edges())
                internal_edges = [edge for edge in internal_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]
                possible_edges.extend( internal_edges )
                
    
            # If not empty, otherwise it makes no sense to focus on this
            if possible_edges:
                
                # Just to check if the above part of possible edges is correct
                if sens1 == sens2 and sens1 in max_sensors and broken_edge not in possible_edges and (broken_edge[1],broken_edge[0]) not in possible_edges:
                    print('\n Non dovresti finire qui, hai tolto edges che erano possibili', broken_edge, edges_not_possible)
                
                # Correct identification
                if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                    results.append( len(possible_edges) ) # we consider them to be equally probable
                    cases_results.append('monopolo giusto')
                    
                # Wrong identification
                else: 
                    results.append( 0 )
                    
                    # Just to save which case are we in 
                    # Actually it was the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                    if sens2 != sens1 and sens1 in max_sensors or sens1 != sens2 and sens2 in max_sensors:
                        cases_results.append('monopolo a metà')
                    
                    # Else is completely wrong
                    else: cases_results.append('monopolo sbagliato')
        
            else:
                # If possible edges is empty, there are no internal edges in monopole cluster that can cause the observed max_signal
                # Thus it might be: 
                #    1) the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                #    2) monopole on weakest monopoles than max_sensor (thus with the same sign)
                # Heuristically (e anche a pensarci), it seems that the first one is by far the most probable; 
                # Thus, go to dipole identification (that will take automatically take the first strongest dipole, that has max_sensor as one of the poles)

                no_monopole_edges = True


# =============================================================================
# # =============================================================================
# #         DIPOLE
# # =============================================================================      
# =============================================================================

        # If we don't sense a net monopole (or the sensed monopole was not possible), we look for a dipole
    
        if abs(max_signal) <= monopole_threshold * abs(sec_max_signal) or no_monopole_edges == True:
            
            # Sensors with a positive and negative difference
            pos_sensors = [key for (key, value) in p_diff_sensors.items() if value > 0 ]
            neg_sensors = [key for (key, value) in p_diff_sensors.items() if value < 0 ]
            
            # All edges (tuple) between positive and negative sensors (positive first): not known order, decided by edge_boundary()
            dipoles_edges = list(nx.edge_boundary(R, pos_sensors, neg_sensors))
            
            # NOTE: if we are here because no_monopole_edges = True due to p_diff sensitivity approximation, 
            # it might be that there are no sensors with p_diff with a different sign than max_sensor (or even different from zero)
            # In this case we take as possible edges all the edges between max_sensor and its neighbors, even if p_diff=0
            # This should happen only if our sensibility is low
             
            if not dipoles_edges:
                    
                max_dipoles_edges = []
                for max_sens in max_sensors:
                    for max_sens_neig in R.neighbors(max_sens):
                        max_dipoles_edges.extend( list( nx.edge_boundary(G, clusters[labels[max_sens]], clusters[labels[max_sens_neig]]) ) )
                possible_edges = [edge for edge in max_dipoles_edges if Qdict[edge] * max_signal > 0]
                
                ###!!!! TOGLI I LINK TRA I PIU MAX_SENSORS SE CI SONO!!! NON POSSONO ESSERE MONOIPOLI
                
                # Here possible edges is always not empty of course
                if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                    results.append( len(possible_edges) ) # we consider them to be equally probable
                    cases_results.append('dipolo giusto 0')
                else: 
                    results.append( 0 )
                    cases_results.append('dipolo sbagliato 0')
        
            
            
            # If there are dipole edges
            else:
                # Sort the dipole_edges according to the greatest difference/gap 
                dipoles_gap = {edge: p_diff_sensors[edge[0]] - p_diff_sensors[edge[1]] for edge in dipoles_edges}
                dipoles_gap = dict(sorted(dipoles_gap.items(), key=lambda item: abs(item[1]), reverse=True ))
                
                # Take the unique values (since they might be degenerate due to p_diff sensibility) in decreasing order
                gaps_values = sorted(set(dipoles_gap.values()), key=abs, reverse=True) 
                
                # Take the biggest value and all the dipoles corresponding to that value
                max_gap = gaps_values[0] 
                max_dipoles = [k for k,v in dipoles_gap.items() if v == max_gap]
                
                # sec_max_dipoles = 0
                # if len(gaps_values) > 1:
                #     sec_max_gap = gaps_values[1] 
                #     sec_max_dipoles = [k for k,v in dipoles_gap.items() if v == sec_max_gap]
                
                ############################
                
                # Each edge between max_dipoles that is compatible with the observed p_diff is a possible candidate
                # Thus for each dipole, take the corresponding clusters, the edges between them, check if the fluxes are compatible
                # like for the monopole, if they are compatible, they are candidate link
                max_dipoles_edges = [list( nx.edge_boundary(G, clusters[labels[dipole[0]]], clusters[labels[dipole[1]]]) ) for dipole in max_dipoles]
                max_dipoles_edges = [edge for sublist in max_dipoles_edges for edge in sublist] # to flatten the list
                
                # The first sensor in max_dipoles_edges (like in dipoles_edges) in the positive one (by construction)
                possible_edges = [edge for edge in max_dipoles_edges if Qdict[edge] > 0]
        
        
                # If there are not possible edges, either we need to check the second highest dipole or we need to check
                # Separately the two clusters that form the highest dipole (with no possible edges)
                # Similarly to above, if there were not possible edges in monopole, we could have checked the second highest monopole
                # but it is more probable that actually it is a weak dipole
                # So at the same way now we check for the two monopoles, not the dipoles
                # Actually it makes more sense to check directly inside the highest monopole

                if not possible_edges:
          
                    # monop = set()
                    # for node1,node2 in max_dipoles:
                    #     monopole = max(p_diff_sensors[node1], p_diff_sensors[node2])
                    #     monop.add(monopole)
                    # monop = [max(max_dip) for max_dip in max_dipoles]
                    
                    ###!!! EXACTLY AS ABOVE, SO MAYBE FUNCTION IS BETTER
                    
                    #max_sensors = [node for couple in max_dipoles for node in couple]
                    
                    possible_edges = []
                    for max_sens in max_sensors:
                        
                        # Nodes belonging to the cluster of max_sens
                        nodes = clusters[labels[max_sens]]
                        sensor_neighbors = list( set(G.neighbors(max_sens)) & set(nodes) ) #intersection
                        edges_not_possible = [(max_sens,sens_neig) for sens_neig in sensor_neighbors if Qdict[(max_sens,sens_neig)] * max_signal < 0]
        
                        # Edges internal to the cluster of max_sens
                        internal_edges = list(nx.subgraph(G, nodes).edges())
                        internal_edges = [edge for edge in internal_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]
                        possible_edges.extend( internal_edges )
                    
                    # If not empty, otherwise it makes no sense to focus on this
                    if possible_edges:
   
                        if sens1 == sens2 and sens1 in max_sensors and broken_edge not in possible_edges and (broken_edge[1],broken_edge[0]) not in possible_edges:
                            print('\n Non dovresti finire qui, hai tolto edges che erano possibili (caso 2)', broken_edge, edges_not_possible)
                         
                        # Correct identification
                        if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                            results.append( len(possible_edges) ) # we consider them to be equally probable
                            cases_results.append('monopolo giusto (caso 2)')
                            
                        # Wrong identification
                        else: 
                            results.append( 0 )
                            
                            # Just to save which case are we in 
                            # Actually it was the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                            if sens2 != sens1 and sens1 in max_sensors or sens1 != sens2 and sens2 in max_sensors:
                                cases_results.append('monopolo a metà (caso 2)')
                            
                            # Else is completely wrong
                            else: cases_results.append('monopolo sbagliato (caso 2)')
                                  
                    else:
                        
                        #print('\n na')
                        # results.append( 0 )
                        # cases_results.append('dipolo sbagliato caso 2')
                                                
                        max_sensors = [node for dipole in max_dipoles for node in dipole if node not in max_sensors]
                        max_sensors_dict = {node:p_diff_sensors[node] for node in max_sensors}
                        max_sens = max(max_sensors_dict, key=abs)
                        max_signal = p_diff_sensors[max_sens]
                        
                        max_sensors = [max_sens]
                        
                        possible_edges = []
                        for max_sens in max_sensors:

                            # Nodes belonging to the cluster of max_sens
                            nodes = clusters[labels[max_sens]]
                            sensor_neighbors = list( set(G.neighbors(max_sens)) & set(nodes) )
                            edges_not_possible = [(max_sens,sens_neig) for sens_neig in sensor_neighbors if Qdict[(max_sens,sens_neig)] * max_signal < 0]

                            # Edges internal to the cluster of max_sens
                            internal_edges = list(nx.subgraph(G, nodes).edges())
                            internal_edges = [edge for edge in internal_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]
                            possible_edges.extend( internal_edges )
                            
                
                        # If not empty, otherwise it makes no sense to focus on this
                        if possible_edges:
                            
                            # Just to check if the above part of possible edges is correct
                            if sens1 == sens2 and sens1 in max_sensors and broken_edge not in possible_edges and (broken_edge[1],broken_edge[0]) not in possible_edges:
                                print('\n Non dovresti finire qui, hai tolto edges che erano possibili', broken_edge, edges_not_possible)
                            
                            # Correct identification
                            if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                                results.append( len(possible_edges) ) # we consider them to be equally probable
                                cases_results.append('monopolo giusto (caso 3)')
                                
                            # Wrong identification
                            else: 
                                results.append( 0 )
                                
                                # Just to save which case are we in 
                                # Actually it was the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                                if sens2 != sens1 and sens1 in max_sensors or sens1 != sens2 and sens2 in max_sensors:
                                    cases_results.append('monopolo a metà (caso 3)')
                                
                                # Else is completely wrong
                                else: cases_results.append('monopolo sbagliato (caso 3)')# + str(sens1) + str(sens2) + str(max_dipoles) + str(max_sensors) )
                        
                        else: 
                            results.append( 0 )
                            cases_results.append('No possible edges')
                                      

                else:
                    if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                        results.append( len(possible_edges) ) 
                        cases_results.append('dipolo giusto')
                    else:
                        results.append( 0 )
                        cases_results.append('dipolo sbagliato')# + str(max_dipoles) )
    
                            
    return results, cases_results


#################################################################################################################################################
########################################## FLUCTUATIONS FAILURE LOCALIZATION FUNCTION #################################################################################
#################################################################################################################################################


def fluc_failure_localization(G, fluc_list, monopole_threshold, demands, solution, Qdict, R, tested_edges, labels, sensors, timing, obs_time):
    
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G)
    L = L.toarray() #from sparse matrix to dense ndarray matrix, in order to use scipy linalg instead of scipy.sparse.linalg (same)
    
    #Results lists
    cases_results = []
    results = []
    
    #clusters list creation
    clusters = []
    for i in range(0,max(labels)+1):
        arr = np.where(labels==i)[0]
        if arr.size > 0:
            clusters.append(arr.tolist())
    
    #FAILURE LOCALIZATION
    for broken_edge_num,fluc_edge in enumerate(tested_edges): 
        
        # To check if we sense the correct cluster
        clus1 = labels[fluc_edge[0]]
        clus2 = labels[fluc_edge[1]]
        sens1 = sensors[clus1]
        sens2 = sensors[clus2]

# =============================================================================
#         EDGE FLUCTUATIONS
# =============================================================================
        
        import itertools
 
        # Covariance matrix of fluc_edge
        comp_cov = fluc_list[fluc_edge]
        
        # Write the info in comp_cov in a dict, more comfortable to use
        cov_dict = dict()
        for couple in itertools.combinations_with_replacement(sensors, 2):
            cov_dict[couple] = comp_cov[couple[0], couple[1]]
        
        # Sort it in decreasing values
        cov_dict = dict(sorted(cov_dict.items(), key=lambda item: abs(item[1]), reverse=True))

# =============================================================================
# # =============================================================================
# #         MONOPOLE 
# # =============================================================================       
# =============================================================================

        # First we determine if twe have a net monopole or not, according to the highest sensors signal
        # Take the unique values (since they might be degenerate due to p_diff sensibility) in decreasing order
        signal_values = sorted(set(cov_dict.values()), key=abs, reverse=True) 
        
        max_signal = signal_values[0] 
        max_sensors = [k for k,v in cov_dict.items() if v == max_signal] # It should be difficult that there are more than one 
        
        # The second one is choosen as the highest with a different sign (or eventually zero) than max_signal
        sec_max_signal = next(( signal for signal in signal_values if signal * max_signal <= 0), 0)
        sec_max_sensors = [k for k,v in cov_dict.items() if v == sec_max_signal]
        
        # =============================================================================
        #         Net monopole
        # =============================================================================
        
        # Boolean variable: we go to the dipole case either if the first one is not bigger than monopole_threshold times the second one 
        # (with a different sign, eventually 0), or if it is satisfied but the candidate monopole has no possible edges
        no_monopole_edges = False
               
        if abs(max_signal) > monopole_threshold * abs(sec_max_signal):
            
            ###!!!!
            # se threshold basso, andrò più volte a cercare direttamete un monopolo (finisco qua più volte)
            # e questo penalizza i miei clusterings rispetto a corr_old
            # la ragione puo essere che i miei sono più sensibili a individuare dipoli ????
            # o perchè ho più rotture tra due che interne e quindi migliorano o peggiorano tutti???
            
            #get clusters1 al posto di get_clusters
            
            ###!!! Se cambiano le demands nel tempo possono cambiare anche la direzione del flusso!!!   QUINDI ATTENDO SE LE PRENDI DA QDICT O DA EDGES ATTRIBUTES
            # Cosa succede se durante un transiente cambia la domanda in modo tale che il flusso si inverte?
            # Il dipolo che si stava creando cambia segno????
            #PER ORA LA DEMAND NON CAMBIA DURANTE UN TRANSIENTE, O ALMENO NON PRIMA DEL (TENTATIVO DI) IDENTIFICAZIONE 
            
            # Check if there are internal edges in the cluster represented by max_sensor that can cause this polarization (i.e. this sign of p_diff on max_sensor)
            # If there are, those are the links we identify as possible failures 
            # If there are not, according to out criterio we identified it as a monopole, but acutally it cannot be: thus it's probably a weak dipole --> no_monopole_edges = True
            # NOTE: we can infer for sure whether an internal link failure causes an increase or decrease of the monopole sensor signal only if that link has the sensor as one of the two nodes
            # In all the other cases, we should do more complex checks on the local topology 
            
            
            ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            max_sensors = [max_sensors[0][0]]

            
            possible_edges = []
            for max_sens in max_sensors:
                
                # Nodes belonging to the cluster of max_sens
                nodes = clusters[labels[max_sens]]
    
                # Edges internal to the cluster of max_sens
                internal_edges = list(nx.subgraph(G, nodes).edges())
                possible_edges.extend( internal_edges )
                
    
            # If not empty, otherwise it makes no sense to focus on this
            if possible_edges:
                
                # Just a check: Se ci abbiamo azzeccato ma abbiamo tolto troppi links
                if sens1 == sens2 and sens1 in max_sensors and fluc_edge not in possible_edges and (fluc_edge[1],fluc_edge[0]) not in possible_edges:
                    print('\n Non dovresti finire qui, hai tolto edges che erano possibili', fluc_edge, possible_edges)
                
                # Correct identification
                if fluc_edge in possible_edges or (fluc_edge[1],fluc_edge[0]) in possible_edges:
                    results.append( len(possible_edges) ) # we consider them to be equally probable
                    cases_results.append('monopolo giusto')
                    
                # Wrong identification
                else: 
                    results.append( 0 )
                    
                    # Just to check which case are we in:
                        
                    # Actually it was the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                    if sens2 != sens1 and sens1 in max_sensors or sens1 != sens2 and sens2 in max_sensors:
                        cases_results.append('monopolo a metà')
                    
                    else: cases_results.append('monopolo sbagliato')
        
            else:
                # If possible edges is empty, there are no internal edges in monopole cluster that can cause the observed max_signal
                # Thus it might be: 
                #    1) the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                #    2) monopole on weakest monopoles than max_sensor (thus with the same sign)
                # Heuristically (e anche a pensarci), it seems that the first one is by far the most probable; 
                # Thus, go to dipole identification (that will take automatically take the first strongest dipole, that has max_sensor as one of the poles)

                no_monopole_edges = True


# =============================================================================
# # =============================================================================
# #         DIPOLE
# # =============================================================================      
# =============================================================================


        # If we don't sense a net monopole (or the sensed monopole was not possible), we look for a dipole
    
        if abs(max_signal) <= monopole_threshold * abs(sec_max_signal) or no_monopole_edges == True:
            
            max_dipoles = sec_max_sensors
            
            # Each edge between max_dipoles that is compatible with the observed p_diff is a possible candidate
            # Thus for each dipole, take the corresponding clusters, the edges between them, check if the fluxes are compatible
            # like for the monopole, if they are compatible, they are candidate link
            max_dipoles_edges = [list( nx.edge_boundary(G, clusters[labels[dipole[0]]], clusters[labels[dipole[1]]]) ) for dipole in max_dipoles]
            max_dipoles_edges = [edge for sublist in max_dipoles_edges for edge in sublist] # to flatten the list
            
            # The first sensor in max_dipoles_edges (like in dipoles_edges) in the positive one (by construction)
            #possible_edges = [edge for edge in max_dipoles_edges if Qdict[edge] > 0]
            
            #########!!!!!!! NON HO PIU COME PRIMA UN AUMENTO O DIMENUZIONE CHE MI DICE FLUSSO!!!
            possible_edges = max_dipoles_edges
            
            if not possible_edges:
                
                results.append( 0 ) # we consider them to be equally probable
                cases_results.append('No possible edges')
                #print('NESSUN EDGE POSSIBILE', (sens1,sens2), broken_edge, possible_edges, max_sensors, sec_max_sensors)
                #print('NESSUN EDGE POSSIBILE', (sens1,sens2), possible_edges, list(p_diff_sensors.items())[:6] )
                #print('i')
                
                # # monop = set()
                # # for node1,node2 in max_dipoles:
                # #     monopole = max(p_diff_sensors[node1], p_diff_sensors[node2])
                # #     monop.add(monopole)
                # # monop = [max(max_dip) for max_dip in max_dipoles]
                
                # ###!!! TORNIAMO A MAX_SENSORS E FORZIAMO LA RICERCA DI UN MONOPOLO DENTRO
                
                # possible_edges = []
                # for max_sens in max_sensors:
                    
                #     # Nodes belonging to the cluster of max_sens
                #     nodes = clusters[labels[max_sens]]
                #     sensor_neighbors = list( set(G.neighbors(max_sens)) & set(nodes) ) #intersection
                #     edges_not_possible = [(max_sens,sens_neig) for sens_neig in sensor_neighbors if Qdict[(max_sens,sens_neig)] * max_signal < 0]
    
                #     # Edges internal to the cluster of max_sens
                #     internal_edges = list(nx.subgraph(G, nodes).edges())
                #     internal_edges = [edge for edge in internal_edges if edge not in edges_not_possible and (edge[1],edge[0]) not in edges_not_possible]
                #     possible_edges.extend( internal_edges )
                    
                # # If not empty, otherwise it makes no sense to focus on this
                # if possible_edges:
   
                #     #Just a check ###!!!! Se ci abbiamo azzeccato ma abbiamo tolto troppi links
                #     if sens1 == sens2 and sens1 in max_sensors and broken_edge not in possible_edges and (broken_edge[1],broken_edge[0]) not in possible_edges:
                #         print('\n Non dovresti finire qui, hai tolto edges che erano possibili 2', broken_edge, possible_edges)
                #         # p_diff1 = network_response(broken_edge, timing, obs_time)
                #         # p_diff = p_diff_list[broken_edge_num]
                #         # if not np.allclose(p_diff1,p_diff):
                #         #     print(broken_edge,broken_edge_num,'Nope')
                    
                    
                #     # Correct
                #     if broken_edge in possible_edges or (broken_edge[1],broken_edge[0]) in possible_edges:
                #         results.append( len(possible_edges) ) # we consider them to be equally probable
                #         cases_results.append('monopolo giusto')
                        
                #     # Wrong
                #     else: 
                #         results.append( 0 )
                        
                #         # Just to check which case are we in:
                #         # Actually it was the pole of a (weak, since it evaded the first check with monopole_threshold) dipole
                #         if sens2 != sens1 and sens1 in max_sensors or sens1 != sens2 and sens2 in max_sensors:
                #             cases_results.append('monopolo a metà')
                        
                #         else: cases_results.append('monopolo sbagliato')
                              
                #     #print(broken_edge, (sens1,sens2),' e ',max_dipoles, sec_max_dipoles, max_sensors, ' res: ', results[len(results)-1] )
                
                # else:
                #     #print('No edgessssss') ###!!! ce ne sono!!!
                #     results.append( 0 )
                #     cases_results.append('dipolo sbagliato caso 2' )
                    
     
            else:
                if fluc_edge in possible_edges or (fluc_edge[1],fluc_edge[0]) in possible_edges:
                    results.append( len(possible_edges) ) 
                    cases_results.append('dipolo giusto')
                else:
                    results.append( 0 )
                    cases_results.append('dipolo sbagliato' + str(max_dipoles) )
        

    return results, cases_results

