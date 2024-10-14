from numpy.random import standard_t as tdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#def std_weighted(values, average, weights):
    # implementing unbiased weighted std formula 
    # http://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean
#    average = np.sum(weights*values)
#    variance = np.sum(weights * (values-average)**2) / (1. - np.sum(weights**2))
#    return np.sqrt(variance)

def std_weighted(values, average, weights, state):
    try:
        # Calculate weighted average
        #average = np.average(values, weights=weights)
        
        # Calculate variance
        numerator = np.sum(weights * (values - average)**2)
        denominator = 1. - np.sum(weights**2)
        
        # Check for division by zero
        if denominator == 0:
            #print(f"Debug info for {state if state else 'unknown state'}:")
            #print(f"Values: {values}")
            #print(f"Weights: {weights}")
            #print(f"Sum of weights: {np.sum(weights)}")
            #print(f"Sum of squared weights: {np.sum(weights**2)}")
            #raise ZeroDivisionError("Sum of squared weights is 1, leading to division by zero")
            return 0.001
        #print(numerator)
        variance = numerator / denominator
        return np.sqrt(variance)
    
    except ZeroDivisionError as e:
        print(f"Error in std_weighted: {e}")
        # You can choose to return a specific value or re-raise the error
        return np.nan  # or return 0, or whatever makes sense for your use case
    
    except Exception as e:
        print(f"Unexpected error in std_weighted: {e}")
        raise  # Re-raise unexpected errors for debugging

def simulate_elections(ppd, states, electors, bias = False, correlations = False, 
                       date_range = None, poll_type = 'pct', min_weight = 0.01, nsims = 1000, 
                       poll_permute = False, results_2020=None, harris_filter=False, **kwargs):
    """
    Simulate election outcomes
    
    Parameters:
    -----------
        ppd:          dictionary with the 538 poll data
        states:       a list of strings of state names
        electors:     a list of integers = the number of electors of each state
        bias:         boolean, flag to apply bias correction
        correlations: boolean, flag to use covariance matrix to impose correlations 
                      on simulated values in different states
        poll_permute: boolean, flag indicating whether to use bootstrap of poll results
                        for Monte Carlo model realizations
        kwargs:       dictionary of additional keyword arguments that are used to pass data such as 
                      covariance matrices
        
    """
    
    nstates = len(states)
    
    # initialize arrays 
    ave_biden, std_biden, n_biden = np.zeros(nstates), np.zeros(nstates), np.zeros(nstates, dtype=int); 
    ave_trump, std_trump, n_trump = np.zeros(nstates), np.zeros(nstates), np.zeros(nstates, dtype=int); 
    
    biden_vote_dist = np.zeros((nstates,nsims))
    trump_vote_dist = np.zeros((nstates,nsims))
    
    biden_electoral_votes = np.zeros(nsims)
    trump_electoral_votes = np.zeros(nsims)    
    
    ndraws = 0 
    
    if bias: # extract bias vectors
        dem_bias = kwargs['dem_bias']
        rep_bias = kwargs['rep_bias']
    
    if correlations: # extract covariance matrices
        cov_dem = kwargs["cov_dem"]
        cov_rep = kwargs["cov_rep"]
    
    # main loop over states
    for i, state in enumerate(states): 

        # array of the polls for which element is for a given state
        stind = (np.array(ppd['state']) == state)

        #the following lines convert list for a key to np array and select elements for each wyind contains True
        name      = np.array(ppd['candidate_name'])[stind] 
        polls     = np.array(ppd[poll_type])[stind]
        wpoll     = np.array(ppd['weight'])[stind]
        startdate = np.array(ppd['startdate'])[stind]
        enddate   = np.array(ppd['enddate'])[stind]
        party     = np.array(ppd['party'])[stind]


        startdates = pd.Series(startdate)
        enddates   = pd.Series(enddate)

        # need to change to Harris. and get new data. 
        if harris_filter:
            indb = ((name == 'Kamala Harris') & (wpoll >= min_weight) &  
                                (startdates > pd.to_datetime(date_range[0])) & (enddates <= pd.to_datetime(date_range[1])))
            indt = ((party == 'Donald Trump') & (wpoll >= min_weight) &  
                                (startdates > pd.to_datetime(date_range[0])) & (enddates <= pd.to_datetime(date_range[1])))
        else:
            indb = ((party == 'DEM') & (wpoll >= min_weight) &  
                                   (startdates > pd.to_datetime(date_range[0])) & (enddates <= pd.to_datetime(date_range[1])))
            indt = ((party == 'REP') & (wpoll >= min_weight) &  
                                   (startdates > pd.to_datetime(date_range[0])) & (enddates <= pd.to_datetime(date_range[1])))
            
        
        
        
        # Current data have no DC polls
        if len(indb) == 0 or len(indt) == 0:
            if state == 'District of Columbia':            
            # Option 2: Assign default values
                ave_biden[i] = results_2020[state]['Joseph R. Biden Jr.'] /100.0
                ave_trump[i] = results_2020[state]['Donald Trump'] /100.0
                std_biden[i] = 0.0  # Default standard deviation of 1%
                std_trump[i] = 0.0
                #biden_votes = np.ones(nsims)  # All simulations give Biden 100%
                #trump_votes = np.zeros(nsims)  # All simulations give Trump 0%
                #biden_electoral_votes += electors[i]  # Biden always wins DC
                continue  # Skip to the next state
            else:
                print(f'No data for {state}')
            continue

    
        
        poll_wb = wpoll[indb] 
        poll_wb /= np.sum(poll_wb)
        poll_wt = wpoll[indt] 
        poll_wt /= np.sum(poll_wt)
        biden_polls   = polls[indb]
        trump_polls   = polls[indt]
        n_biden[i] = biden_polls.size 
        n_trump[i] = trump_polls.size 
    
        if np.sum(poll_wb) > 0 and np.sum(poll_wt) > 0:
            # compute weighted averages and st devs for polls in the selected date range
            ave_biden[i] = np.average(biden_polls, weights=poll_wb)
            ave_trump[i] = np.average(trump_polls, weights=poll_wt)


            if len(biden_polls) == 1:
                print(f'{state} - 1 poll only: Dem = {biden_polls}, Rep = {trump_polls}')
            std_biden[i] = std_weighted(biden_polls, ave_biden[i], poll_wb, state)
            std_trump[i] = std_weighted(trump_polls, ave_trump[i], poll_wt, state)
        else:

            # Option 2: Assign default values
            ave_biden[i] = results_2020[state]['Joseph R. Biden Jr.'] 
            ave_trump[i] = results_2020[state]['Donald Trump'] 
            std_biden[i] = 0.0  # Default standard deviation of 1%
            std_trump[i] = 0.0
            print(f'{state} - 2020 results: Dem = {ave_biden[i]}, Rep ={ave_trump[i]}')            
        if bias: # if this option, apply bias to the averages
            ave_biden[i] += dem_bias[i]
            ave_trump[i] += rep_bias[i]
                
        if poll_permute: # use bootstrap for model realization
            nx = biden_polls.size
            ibs = np.random.choice(nx, size = nsims)
            biden_votes = biden_polls[ibs] 
            trump_votes = trump_polls[ibs] 
        elif correlations: # if correlations are to be imposed only produce random numbers
            if n_biden[i] <= 1: 
                biden_votes = tdist(1, size = nsims)
            else:
                biden_votes = tdist(n_biden[i]-1, size = nsims)
            if n_trump[i] <= 1:
                trump_votes = tdist(1, size = nsims)        
            else:
                trump_votes = tdist(n_trump[i]-1, size = nsims)
        else: # realization for the fiducial model
            biden_votes = ave_biden[i] + std_biden[i] * tdist(max(1,n_biden[i]-1), size = nsims)
            trump_votes = ave_trump[i] + std_trump[i] * tdist(max(1,n_trump[i]-1), size = nsims)
    
        # now we can process simulations. using numpy filtering ops
        # allows us to avoid any loops over simulations here
        if correlations: 
            biden_vote_dist[i,:] = biden_votes
            trump_vote_dist[i,:] = trump_votes
        else:                
            draws = (trump_votes == biden_votes)
            biden_wins = (biden_votes[:] > trump_votes[:])  
            trump_wins = (trump_votes[:] > biden_votes[:])

            biden_electoral_votes[biden_wins] += electors[i]
            # assign draws with 50% probability but here they are all assigned to Trump
            # to see how this boosts his changes, the chance of a draw here is negligible however
            trump_electoral_votes[draws]      += electors[i] # assume all draws go to Trump
            
            trump_electoral_votes[trump_wins] += electors[i]
            ndraws += len(trump_electoral_votes[draws])
            biden_vote_dist[i,:] = biden_votes
            trump_vote_dist[i,:] = trump_votes
        
    # now handle case where correlations are imposed 
    if correlations: 
        # Cholesky decomposition of covariance matrices
        l_dem = np.linalg.cholesky(cov_dem)
        l_rep = np.linalg.cholesky(cov_rep)
        for j in range(nsims):
            # generate correlated random numbers 
            biden_vote_dist[:,j] = ave_biden + np.dot(l_dem, biden_vote_dist[:,j]) 
            trump_vote_dist[:,j] = ave_trump + np.dot(l_rep, trump_vote_dist[:,j]) 
        # analyze simulations for different states
        for j in range(nstates):
            biden_votes, trump_votes = biden_vote_dist[j,:], trump_vote_dist[j,:]
            draws = (trump_votes == biden_votes)
        
            biden_electoral_votes[(biden_votes[:]>trump_votes[:])] += electors[j]
            trump_electoral_votes[draws] += electors[j] # assume all draws go to Trump
            trump_electoral_votes[(trump_votes[:]>biden_votes[:])] += electors[j]        

    return biden_electoral_votes, trump_electoral_votes, biden_vote_dist, trump_vote_dist, ave_biden, ave_trump
    
