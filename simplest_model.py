import numpy as np
from matplotlib import pylab as plt
import pandas as pd 
from datetime import datetime as dt
from datetime import date

from poll_weight import add_poll_weights
from utils import combine_districts 
from utils import election_stats, state_results, \
                  plot_electoral_vote_distribution, \
                    plot_nelectoral_vs_popular_vote
from utils import std_weighted

from simulation import simulate_elections



if __name__ == '__main__':

    # Read in polling data 
    url = './data/president_polls.csv'
    pad = pd.read_csv(url)

    # Cleanup data 
    pad = cleanup_data(pad)
    ppl = pad.to_dict(orient = 'list') 

    # states electoral votes
    states_electoral = pd.read_csv('./data/states_electoral.csv')
    states_abrv = np.array(states_electoral['State'])
    states      = np.array(states_electoral['Name'])
    electors    = np.array(states_electoral['Electors'])
    nstates     = states.size
    state_pop = np.array(states_electoral['Population'])
    

    date_range = ['2024-04-01', '2024-10-06']
    nsims = 10000
    poll_type = 'pct'

    simplest_model_results = simulate_elections(ppl, states=states, electors=electors, date_range=date_range, 
                                                poll_type=poll_type, min_weight = 0.00, nsims = nsims)
    
    model_name = 'model 1 (simplest model)'
    election_stats(model_name, simplest_model_results)

    #plot_nelectoral_vs_popular_vote(model_name, simplest_model_results, states, electors, state_pop)

    biden_electoral_votes, trump_electoral_votes, biden_vote_dist, trump_vote_dist, ave_biden, ave_trump = simplest_model_results
    
    print(biden_vote_dist)
    #plt.show()