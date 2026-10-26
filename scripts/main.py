from life_of_eurpean import load_time_use, fit_models
from create_data import simulate_activity_log, validate_log
import pandas as pd
import numpy as np
from d3blocks import D3Blocks

# %% Set parameters
# ages=['Y10-14', 'Y15-24', 'Y15-29', 'Y15-64', 'Y25-34',  'Y30-44', 'Y30-64', 'Y35-44', 'Y45-54', 'Y45-64', 'Y55-64','Y65-74', 'Y_GE60', 'Y_GE65', 'Y_GE75'])

# countries=["Netherlands"],
countries = 'all'
# countries=["Netherlands", "Germany", "Norway"],

# Number of samples per country
k = 25

# %% Model age groups
# df = load_time_use("estat_tus_20age.csv", ages=['TOTAL'])
df1 = load_time_use("estat_tus_20age.csv", ages=['Y10-14'])
df2 = load_time_use("estat_tus_20age.csv", ages=['Y25-34', 'Y30-64'])
df3 = load_time_use("estat_tus_20age.csv", ages=['Y65-74','Y_GE75'])
model1 = fit_models(df1)
model2 = fit_models(df2)
model3 = fit_models(df3)
groups = zip([model1, model2, model3], [df1, df2, df3])

# %%
Xtot = []

for model, df in groups:
    X = simulate_activity_log(
        model,
        countries=countries,
        k=k,
        seed=42,
        bedtime_mean_min=22 * 60,
        bedtime_sd_min=10,
        late_fraction=0.10,    # share of late sleepers
        late_shift_min=30,     # how much later they go to bed
        # node sizing
        base_size=3, # node size
        eating_size=5, # node size after eating
        size_decay_min=120,    # time to shrink back
        size_step_min=60,      # smaller = smoother, more rows
        # Morning travel
        morning_travel_mean_min=7 * 60 + 30,   # typical departure, 07:30
        morning_travel_sd_min=30,
        morning_travel_max_min=8 * 60,         # latest departure, 08:00
        # traveling evening
        evening_travel_mean_min=17 * 60,   # 17:00
        evening_travel_sd_min=30,
        home_state="Home",                # what people are doing after getting home
        state_aliases={"Household": "Home", "Media": "Home", "Hobbies": "Home", "Socialising": "Leisure"}
    )

    X['state'] = X['state'].replace('Other','Sports/Hobbies/Unspecified')
    X['state'] = X['state'].replace('Home','Home/ Leisure')
    X['state'] = X['state'].replace('Study','School/ Study')
    X['state'] = X['state'].replace('Employment','Work')
    
    age_str = '_'.join(df['age'].unique())
    X['age'] = '_'.join(df['age'].unique())
    if '14' in age_str:
        X['size'] = np.random.randint(1, 5, X.shape[0])
    else:
        X['size'] = np.random.randint(4, 6, X.shape[0])

    Xtot.append(X)

X = pd.concat(Xtot, ignore_index=True)
X = X.sort_values('datetime', ascending=False)
X.reset_index(drop=True, inplace=True)

# %% Create flowchart

d3 = D3Blocks()
d3.flowgraph('''
flowchart TD
    C[Sleeping]
    C --> D[Personal care]
    C --> E[Eating]
    D[Personal care] --> E[Eating]
    D[Personal care] --> F[Travel]
    E[Eating] <--> F[Travel]
    F[Travel] <--> G[Work]
    F[Travel] <--> H[Study/ School]
    F[Travel] <--> I[Volunteer work]
    F[Travel] <--> J[Sports/ Hobbies/ Unspecified]
    G[Work] <--> H[Study/ School]
    G[Work] <--> I[Volunteer work]
    H[Study/ School] <--> I[Volunteer work]
    F[Travel] <--> K[Home/ Leisure]
    K[Home/ Leisure] <--> I[Volunteer work]
    K[Home/ Leisure] <--> H[Study/ School]
    K[Home/ Leisure] <--> J[Sports/ Hobbies/ Unspecified]
    K[Home/ Leisure] <--> G[Work]
    ''', layout='elk')

# %%
# Initialize
d3 = D3Blocks()
X['opacity'] = np.random.random(size=X.shape[0])

# Create the chart
d3.movingbubbles(X,
                 sample_id='uid',
                 size=X['size'],
                 color=X['age'],
                 opacity=X['opacity'],
                 timelines='country',
                 shape='age',
                 tooltip='age',
                 damper=0.2,
                 color_method='auto',
                 # stroke='#ffffff',
                 center="Travel", # Set the state "Travel" central
                 figsize = [1200, 1000],
                 title='Daily Life of a European',
                 note='Daily Life of a European',
                 cmap='Set2',
                 filepath=r'daily_life_of_european.html',
                 )
