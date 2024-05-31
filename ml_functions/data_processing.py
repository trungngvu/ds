


import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
  




def scale_df(df, scale, unscale):
    
    scaled_np = preprocessing.scale(df)
    
    col_list = []
    for col in df.columns:
        col_list.append(col)
    
    scaled_df = pd.DataFrame(scaled_np)
    scaled_df.columns = col_list
    
    df1 = scaled_df.iloc[:, scale]
    df2 = df.iloc[:, unscale]
    
    final_df = pd.concat([df1, df2], axis=1, sort=False)
    
    return final_df

        

def scree_plot(pca_percentages, y_max=40):
    n_components = len(pca_percentages)
    
    
    fig, ax = plt.subplots()
    
    
    ax.bar(list(range(1, n_components+1, 1)), pca_percentages, color='paleturquoise', edgecolor='darkturquoise', zorder=0)
    
    
    for p in ax.patches:
        ax.annotate(f'{round(p.get_height(), 1)}%', (p.get_x() + 0.5, p.get_height() + 0.5))
    
    
    ax.plot(list(range(1, n_components+1, 1)), pca_percentages, c='firebrick', zorder=1)
    ax.scatter(list(range(1, n_components+1, 1)), pca_percentages, c='firebrick', zorder=2)
    
    
    fig.suptitle('PCA Scree Plot', y=0.96, fontsize=16, fontweight='bold');
    ax.set(xlabel='Principle Components',
           ylabel='Percentage Variation');
    ax.set_ylim([0,y_max])
    
    return fig


