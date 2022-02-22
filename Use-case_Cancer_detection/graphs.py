import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

def plot_correlation_matrix(df, title):
    fig, ax = plt.subplots(figsize = (15,8))

    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax)
    ax.set_title(title, fontsize=14)

   