print('\n\n ---------------- START ---------------- \n')



import time
start=time.time()

import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')






df_5_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_5_v2.txt'
df_10_saved_name = '2015_2016_2017_2018_2019_2020_2021_2022_2023_prem_df_for_ml_10_v2.txt'

save_df_10_fig = False
save_df_5_fig = False

colourbar = 'winter'


plot_results = [1] 




with open(f'prem_clean_fixtures_and_dataframes/{df_5_saved_name}', 'rb') as myFile:
    df_ml_5 = pickle.load(myFile)

with open(f'prem_clean_fixtures_and_dataframes/{df_10_saved_name}', 'rb') as myFile:
    df_ml_10 = pickle.load(myFile)






for i in range(0, len(df_ml_10)):
    if df_ml_10['Team Result Indicator'].loc[i] in plot_results:
        continue
    else:
        df_ml_10 = df_ml_10.drop([i], axis=0)

df_ml_10 = df_ml_10.reset_index(drop=True)


for i in range(0, len(df_ml_5)):
    if df_ml_5['Team Result Indicator'].loc[i] in plot_results:
        continue
    else:
        df_ml_5 = df_ml_5.drop([i], axis=0)

df_ml_5 = df_ml_5.reset_index(drop=True)





fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 10 Games', y=0.99, fontsize=16, fontweight='bold')

transparency = 0.6
markersize = 25


scat1 = ax1.scatter(df_ml_10['Team Av Shots Diff'], 
                   df_ml_10['Opponent Av Shots Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat2 = ax2.scatter(df_ml_10['Team Av Shots Inside Box Diff'], 
                   df_ml_10['Opponent Av Shots Inside Box Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat3 = ax3.scatter(df_ml_10['Team Av Fouls Diff'], 
                   df_ml_10['Opponent Av Fouls Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat4 = ax4.scatter(df_ml_10['Team Av Corners Diff'], 
                   df_ml_10['Opponent Av Corners Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)


scat5 = ax5.scatter(df_ml_10['Team Av Pass Accuracy Diff'], 
                   df_ml_10['Opponent Av Pass Accuracy Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat6 = ax6.scatter(df_ml_10['Team Av Goal Diff'], 
                   df_ml_10['Opponent Av Goal Diff'], 
                   c=df_ml_10['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)


fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots Difference',
        ylabel='Opponent Average Shots')

ax2.set(xlabel='Team Average Shots Inside Box Difference',
        ylabel='Opponent Average Shots Inside Box Difference')

ax3.set(xlabel='Team Average Fouls Difference',
        ylabel='Opponent Average Fouls Difference')

ax4.set(xlabel='Team Average Corners Difference',
        ylabel='Opponent Average Corners Difference')

ax5.set(xlabel='Team Average Pass Accuracy % Difference',
        ylabel='Opponent Average Pass Accuracy % Difference')

ax6.set(xlabel='Team Average Goals Difference',
        ylabel='Opponent Average Goals Difference')

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Result', loc='upper right', fontsize='small')
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    a_min = lims[0]
    a_max = lims[1]
    mult = lims[1] - lims[0]
    ax.plot([a_min, a_max], [a_min, a_max], '--', color = '
    
    


if save_df_10_fig:
    fig.savefig('figures/average_10_games_team_target_result.png')





fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(ncols=3,
                                                      nrows=2,
                                                      figsize=(18,12))

fig.suptitle('Data Averaged Over 5 Games', y=0.99, fontsize=16, fontweight='bold')


scat1 = ax1.scatter(df_ml_5['Team Av Shots Diff'], 
                   df_ml_5['Opponent Av Shots Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat2 = ax2.scatter(df_ml_5['Team Av Shots Inside Box Diff'], 
                   df_ml_5['Opponent Av Shots Inside Box Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat3 = ax3.scatter(df_ml_5['Team Av Fouls Diff'], 
                   df_ml_5['Opponent Av Fouls Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat4 = ax4.scatter(df_ml_5['Team Av Corners Diff'], 
                   df_ml_5['Opponent Av Corners Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)


scat5 = ax5.scatter(df_ml_5['Team Av Pass Accuracy Diff'], 
                   df_ml_5['Opponent Av Pass Accuracy Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)

scat6 = ax6.scatter(df_ml_5['Team Av Goal Diff'], 
                   df_ml_5['Opponent Av Goal Diff'], 
                   c=df_ml_5['Team Result Indicator'],
                   cmap = colourbar,
                   alpha=transparency,
                   s=markersize)


fig.tight_layout(pad=6)

ax1.set(xlabel='Team Average Shots Difference',
        ylabel='Opponent Average Shots')

ax2.set(xlabel='Team Average Shots Inside Box Difference',
        ylabel='Opponent Average Shots Inside Box Difference')

ax3.set(xlabel='Team Average Fouls Difference',
        ylabel='Opponent Average Fouls Difference')

ax4.set(xlabel='Team Average Corners Difference',
        ylabel='Opponent Average Corners Difference')

ax5.set(xlabel='Team Average Pass Accuracy % Difference',
        ylabel='Opponent Average Pass Accuracy % Difference')

ax6.set(xlabel='Team Average Goals Difference',
        ylabel='Opponent Average Goals Difference')

ax_iter = [ax1, ax2, ax3, ax4, ax5, ax6]

for ax in ax_iter:
    ax.legend(*scat2.legend_elements(), title='Target \n Team \n Result', loc='upper right', fontsize='small')
    ax.set_axisbelow(True)
    ax.grid(color='xkcd:light grey')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    a_min = lims[0]
    a_max = lims[1]
    mult = lims[1] - lims[0]
    ax.plot([a_min, a_max], [a_min, a_max], '--', color = '
    
    


if save_df_5_fig:
    fig.savefig('figures/average_5_games_team_target_result.png')




print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
