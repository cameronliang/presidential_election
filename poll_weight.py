import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.arrays.datetimes")

def poll_weight(
    numeric_grade,
    is_projection,
    start_date,
    end_date,
    pollscore,
    sample_size,
    reference_today_date,
    time_penalty=9,
    harris_and_before_dropout=False,
    pollster_name: str = None,
    exclude_pollsters: list[str] | None = None,  # To answer a reddit question, not actually used
):
    """Given factors about a poll return some weighting. This is somewhat
    arbitrary and could be adjusted.

    Args:
        numeric_grade: The 538 rating of the pollster. Rated out of 3.0.
        is_projection: If the poll is a projection (not currently used)
        start_date: The start date of the poll
        end_date: The end date of the poll
        pollscore: The 538 pollster score. Lower values are better. Like the
            numberic grade, but just measures empirical track record, not
            factors like transparency.
        sample_size: The number of people polled
        reference_today_date: The date to use when weighting old polls
        time_penalty: Approximately the days half life
        harris_and_before_dropout: If the poll is for Harris and before the
            dropout day
        pollster_name: The name of the pollster
        exclude_pollsters: A list of pollsters to exclude (matches any substring)
            Not actually used in the function, but is a parameter to make it
            easier to answer a reddit question.
    """
    # If NaN then numeric_grade 1.5
    if pd.isna(numeric_grade):
        grade_clipped = 1.5
    else:
        grade_clipped = max(0.0, min(2.7, numeric_grade))
    score = grade_clipped**1.5 / 2.7**1.5
    if is_projection:
        score = (score**2) * 0.5
    # Pollscore which is some value considering only performance (ignoring transparency)
    # Lower values are better
    if pd.isna(pollscore):
        pollscore = 0.0
    score *= np.interp(pollscore, [-1.1, 0.5], [1.0, 0.6])
    # Some sample size consideration
    if pd.isna(sample_size):
        if numeric_grade > 2.0:
            sample_size = 500
        else:
            sample_size = 300
    score *= np.interp(sample_size, [200, 900], [0.7, 1.0])
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    # Time decay
    end_days = (reference_today_date - end_date).days
    if end_days < 0:
        return 0  # in the future
    start_days = (reference_today_date - start_date).days
    # Find a middle date (putting more weight on end date since I think some
    # pollsters will start out smaller and scale up(??))
    days = (end_days * 2 + start_days * 1) / 3
    days -= 1.5  # Prep time
    days = max(0, days)
    # Subtract a prep day
    time_decay = 0.5 ** (days / time_penalty)
    if harris_and_before_dropout:
        score *= 0.25
    score *= time_decay
    # Especially punish low quality
    #score = score ** 1 + days.days / time_decay

    if exclude_pollsters and pollster_name: # for reddit question
        for pollster in exclude_pollsters:
            if pollster_name.lower() in pollster.lower():
                return 0.0

    if score < 0:
        return 0.0
    return score

def add_poll_weights(df, reference_date, harris_and_before_dropout=False):
    # Convert date columns to datetime
    df.loc[:,'start_date'] = pd.to_datetime(df['start_date'], format='%m/%d/%y',errors='coerce')
    df.loc[:,'end_date'] = pd.to_datetime(df['end_date'], format='%m/%d/%y',errors='coerce')
    df = df.dropna(subset=['state','start_date','end_date'])

    # Set reference_today_date to the latest end_date in the dataset
    reference_today_date = pd.Timestamp(reference_date)#pd.Timestamp(date.today())
    
    # Apply the poll_weight function to each row
    df.loc[:,'weight'] = df.apply(lambda row: poll_weight(
        row['numeric_grade'],
        True,  # Assuming is_projection is always False for this dataset
        row['start_date'],
        row['end_date'],
        row['pollscore'],
        row['sample_size'],
        reference_today_date,
        harris_and_before_dropout = harris_and_before_dropout,
        pollster_name=row['pollster']
    ), axis=1)
    
    return df

def combine_districts(state):
    # Dictionary to map districts to states
    district_to_state = {
        'Maine CD-1': 'Maine',
        'Maine CD-2': 'Maine',
        'Nebraska CD-1': 'Nebraska',
        'Nebraska CD-2': 'Nebraska',
        'Nebraska CD-3': 'Nebraska'
    }
    
    # Return the mapped state if it's a district, otherwise return the original state
    return district_to_state.get(state, state)


def cleanup_data(pad, harris_filter=False):

    # Clean up polling data 
    try:
        pad['startdate'] = pd.to_datetime(pad['start_date'], format='%m/%d/%y',errors='coerce')
    except ValueError:
        # If parsing fails, mark those rows as problematic
        pad['startdate'] = pd.NaT
        
    try:
        pad['enddate'] = pd.to_datetime(pad['end_date'], format='%m/%d/%y',errors='coerce')
    except ValueError:
        # If parsing fails, mark those rows as problematic
        pad['enddate'] = pd.NaT
        
    # convert data frame to Python dictionary with keys containing names of columns in the data frame
    # and each key containing a list of strings or float values
    #pad = pad.dropna(subset=['state','start_date','end_date'])
    pad = add_poll_weights(pad,'10-12-2024',harris_filter)

    #pad = pad.dropna(subset=['state','start_date','end_date'])
    
    pad = pad[pad['weight'] >0]
    pad['state'] = pad['state'].apply(combine_districts)

    pad = pad[pad['party'].isin(['DEM','REP'])]

    """
    if harris_filter:
        # Filter for polls that include both Trump and Harris
        trump_polls = pad[pad['candidate_name'].str.contains('Trump', case=False, na=False)]['poll_id']
        harris_polls = pad[pad['candidate_name'].str.contains('Harris', case=False, na=False)]['poll_id']
        valid_polls = set(trump_polls) & set(harris_polls)  # Intersection of Trump and Harris polls
        
        # Keep only the rows from valid polls (those including both Trump and Harris)
        pad = pad[pad['poll_id'].isin(valid_polls)]
        
        # Final filter to keep only Trump and Harris rows
        pad = pad[pad['candidate_name'].str.contains('Trump|Harris', case=False, na=False)]
    """

    return pad