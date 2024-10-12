import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def election_stats(model_name, model_results):
    """
    Helper function to output basic statistics about results of a model
    
    Parameters: 
    -----------
    model_name: string containing name of the model
    
    model_results: a list of arrays output by simulate_elections function
    """
    biden_electoral_votes, trump_electoral_votes = model_results[0], model_results[1]
    biden_vote_dist, trump_vote_dist = model_results[2], model_results[3]
    nsims = np.shape(biden_electoral_votes)[0]
    
    print('='*15+'Results of election simulations in the ', model_name, '===')

    trump_wins   = (trump_electoral_votes > biden_electoral_votes) 
    biden_wins = (trump_electoral_votes < biden_electoral_votes) 
    draws = (trump_electoral_votes == biden_electoral_votes) 
    fdraws = len(biden_electoral_votes[draws]) / nsims
    fbiden_wins = len(biden_electoral_votes[biden_wins]) / nsims
    ftrump_wins   = len(trump_electoral_votes[trump_wins]) / nsims

    print("Harris wins in %.4f per cent of elections"%(fbiden_wins*100.))
    print("        average and median Nelectoral = %.2f and %.2f; in 95 percent range = [%.2f  %.2f]"%(
        np.mean(biden_electoral_votes), np.median(biden_electoral_votes), 
        np.percentile(biden_electoral_votes,2.5), np.percentile(biden_electoral_votes,97.5)))
    print("Trump   wins in %.4f per cent of elections"%(ftrump_wins*100.))
    print("        average and median Nelectoral = %.2f and %.2f; in 95 percent range = [%.2f  %.2f]"%(
        np.mean(trump_electoral_votes), np.median(trump_electoral_votes), 
        np.percentile(trump_electoral_votes,2.5), np.percentile(trump_electoral_votes,97.5)))

    print(" %.4f per cent of elections end up in electoral college draw"%(fdraws*100.))



def state_results(model_name, model_results, states, states_to_print=None):
    """
    Helper function to output basic statistics about results in specific states
    
    Parameters: 
    -----------
    model_name: string containing name of the model
    
    model_results: a list of arrays output by simulate_elections function
    
    states: array of strings of state names
    
    states_to_print: a list of state names that will be printed. If None, all states will be printed
    """
    
    biden_electoral_votes, trump_electoral_votes = model_results[0], model_results[1]
    biden_vote_dist, trump_vote_dist = model_results[2], model_results[3]
    nsims = np.shape(biden_electoral_votes)[0]
    
    if states_to_print is None: 
        states_to_print = states
        
    nc = np.zeros(51); colors = np.zeros((51,4))
    print('='*10+' State-by-state vote predictions for {:s} '.format(model_name)+'='*10)
    print('               State   Haris win prob.  Trump win prob. ')
    for i, ncd in enumerate(nc): 
        if states[i] in states_to_print: 
            biden_wins = (biden_vote_dist[i,:] > trump_vote_dist[i,:]) 
            trump_wins   = (trump_vote_dist[i,:] > biden_vote_dist[i,:])
            vote_ratio_trump = len(trump_vote_dist[i,trump_wins]) / nsims
            vote_ratio_biden = len(biden_vote_dist[i,biden_wins]) / nsims
            print("%20s        %.3f           %.3f"%(states[i], vote_ratio_biden, vote_ratio_trump))


def plot_electoral_vote_distribution(model_name, model_results, 
                                     nbins=40, savefig=None):
    """
    plot histogram of the electoral vote distribution based on model elections
    Parameters:  
        biden_electoral_votes, 
        trump_electoral_votes = total counts of electoral votes for Biden and Trump in model elections
        nbins = number of bins to use 
    """
    
    biden_electoral_votes, trump_electoral_votes, = model_results[0], model_results[1], 
    nsims = np.shape(biden_electoral_votes)[0]

    _, ax = plt.subplots(1, 1, figsize=(8,4))
    ax.set_title(model_name)
    ax.set_xlabel(r'$N_{\rm electoral\ votes}$')
    ax.set_ylabel(r'fraction of model elections')
    ax.hist(biden_electoral_votes, color='b', histtype='stepfilled', alpha=0.7, bins=nbins, 
            label=r'Harris', density = 'True')
    ax.hist(trump_electoral_votes,color='r', histtype='stepfilled', alpha=0.7, bins=nbins, 
            label=r'Trump', density = 'True')
    ax.legend(loc='best', frameon=False, fontsize=15)


    if savefig != None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def plot_nelectoral_vs_popular_vote(model_name, model_results, states, electors, state_pop, savefig=None):
    """
    plots of the expected Electoral College votes vs. the fraction of popular vote 
    received for all of the elections
    """
    
    biden_electoral_votes, trump_electoral_votes = model_results[0], model_results[1]
    biden_vote_dist, trump_vote_dist = model_results[2], model_results[3]   
    nsims = np.shape(biden_electoral_votes)[0]
    biden_pop_vote, trump_pop_vote, = np.zeros(nsims), np.zeros(nsims)
    
    # compute the popular vote fractions
    total_pop   = np.sum(state_pop)
    nstates     = states.size
    fstate_pop = state_pop / total_pop
    
    for i, state in enumerate(states):
        biden_pop_vote[:] += biden_vote_dist[i,:] * state_pop[i] * 0.5 / 100
        trump_pop_vote[:] += trump_vote_dist[i,:] * state_pop[i] * 0.5 / 100
    
    _, ax = plt.subplots(1, 1, figsize=(6,6))
    plt.rc('font', size=15)
    ax.set_xlim(50., 100.)
    ax.set_xlabel('N_popular (millions)', fontsize=12)
    ax.set_ylabel('N_electoral', fontsize=12)
    ax.scatter(biden_pop_vote*1.e-6, biden_electoral_votes, marker='.', s=5.0, edgecolor='none', alpha=0.5, c='b')
    ax.scatter(trump_pop_vote*1.e-6, trump_electoral_votes, marker='.', s=5.0, edgecolor='none', alpha=0.5, c='r')
    x = np.arange(0., 100, 0.1); y = 270. * np.ones_like(x)
    ax.plot(x, y, '--', c='m')
    plt.title(model_name)




def std_weighted(values, average, weights):
    # implementing unbiased weighted std formula 
    # http://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean
    average = np.sum(weights*values)
    variance = np.sum(weights * (values-average)**2) / (1. - np.sum(weights**2))
    return np.sqrt(variance)
