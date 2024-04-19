"""
    Functions to analyze the results of the rPPG extraction from different landmarks
    Author: Emilie Rolland-Piegue
    Date: 09.04.2024
"""


import numpy as np
import pandas as pd
import constants
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import plotly.express as px

from scipy.stats import ttest_ind, zscore

PALETTE = 'Spectral'

##################### DATA FORMATTING ########################

def get_dataset_settings(datasets):
    """
        Get the settings of the videos in the dataset (ex: subject has beard, glasses, ...)
    """

    SETTINGS = {}
    for dataset in datasets:
        setting = constants.get_video_settings(dataset)
        SETTINGS[dataset] = setting 
        print(f'Video settings in {dataset}: ', SETTINGS[dataset].keys())
    for dataset in datasets:
        for key in SETTINGS[dataset].keys():
            for other_dataset in datasets:
                if other_dataset == dataset: continue
                if key not in SETTINGS[other_dataset].keys(): SETTINGS[other_dataset][key] = []
                
    return SETTINGS

def format_data(df):
    """
        Calculate overall score by taking the average of z-scores of MAE, rPPG_PCC, DTW
        The score has to be minimized, so lower score (negative z score, below average) is good
    """

    # Remove eliminated videos 
    df = df[~df['videoFilename'].isin(constants.eliminated_subjects)].reset_index(drop=True).copy()

    df['PCC'] = df['PCC'].abs()

    ### If multiple methods are used, take the mean of all methods for each landmark
    metrics = ['RMSE','MAE', 'PCC', 'SNR', 'rPPG_PCC', 'DTW',]
    df[metrics] = df.groupby(['videoFilename','landmarks'])[metrics].transform('mean') # average over methods
    df = df.drop_duplicates(['videoFilename','landmarks']).reset_index(drop=True).drop(columns=['method'])  # drop duplicates over methods

    ### Calculate overall score
    # Take min max normalization of each metric
    for metric in metrics:
        df[f'{metric}_z'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        if 'PCC' in metric: df[f'{metric}_z'] = 1-df[f'{metric}_z']
    # Calculate overall score
    df['OS'] = (1/2)*(df['MAE_z'] + df['DTW_z']).astype('float64')
    df = df.drop(columns=[metric+'_z' for metric in metrics])

    return df

def format_data_symmetric_landmark(df, symmetric=True):
    """
        Format the landmarks column
        Symmetric: take symmetric landmarks as one (True) or separate (False)
    """
    ### Get data
    if type(df['landmarks'].iloc[0]) != str: 
        if symmetric:
            df['landmarks'] = df['landmarks'].apply(lambda x: x[0] if 'left' not in x[0] else '_'.join(x[0].split('_')[1:])) # take symmetric landmarks as one
        else: 
            df['landmarks'] = df['landmarks'].apply(lambda x: x[0]) # take symmetric landmarks as separate

    return df

def get_df_rank(x, bins=[0, 0.1, 0.9, 1]):
    """
    Assigns ranks and bins to landmarks based on MAE

    Parameters:
        x (DataFrame): Input DataFrame containing data related to landmarks.
        bins (list): List of quantile values for binning (default is [0, 0.1, 0.9, 1]).

    Returns:
        Tuple: Tuple containing the updated DataFrame with ranks and bins assigned, 
               the DataFrame with ranks and bins information, 
               and a dictionary mapping landmarks to their ranks.
    """

    # Get mean metric value for each landmark and rank them
    df_rank = x[['landmarks_id', 'landmarks', 'region', 'MAE', 'OS', 'DTW']].groupby(['landmarks_id', 'landmarks' , 'region']).agg(['median', 'mean','std' ])
    df_rank['rank_MAE'] = df_rank[('MAE', 'mean')].rank(ascending=True).astype(int)
    df_rank['rank_DTW'] = df_rank[('DTW', 'mean')].rank(ascending=True).astype(int)
    df_rank['rank_OS'] = df_rank[('OS', 'mean')].rank(ascending=True).astype(int)

    # sort by mean OS score
    df_rank = df_rank.sort_values(by=('OS', 'mean')).reset_index()
    df_rank.columns = df_rank.columns.map('_'.join).str.strip('_') # Rename columns

    # individual landmarks: If landmarks are not a tuple, replace underscores with spaces
    if type(df_rank['landmarks'].iloc[0]) == str: 
        df_rank['landmarks'] = df_rank['landmarks'].apply(lambda x: x.replace('_', ' '))

    # Cut based on MAE quantiles
    quantiles = df_rank['MAE_mean'].quantile(bins)
    df_rank['bin'] = pd.cut(df_rank['MAE_mean'], bins=quantiles, labels=[f'G1: < {str(bins[1])}%',
                                                                         f'G2: {str(bins[1])}-{str(bins[2])}%', 
                                                                         f'G3: > {str(bins[2])}%'], 
                            include_lowest=True).astype('str')
    x = x.merge(df_rank[['landmarks_id', 'bin', 'MAE_mean', 'rank_MAE', 'rank_OS']], on='landmarks_id', how='left')
    
    # landmarks score dictionary
    ldmk_score_dict = dict(zip(df_rank['landmarks'], df_rank['rank_OS']))

    return x, df_rank, ldmk_score_dict


##################### PLOT ########################

def get_palette(df, hue='landmarks_id'):
    # Defining the color palette
    # palette = sns.color_palette(PALETTE, n_colors=len(df[hue].unique()))
    # palette = dict(zip(df[hue].unique(), palette))

    palette = sns.color_palette(PALETTE, n_colors=len(df['region'].unique()))
    palette = dict(zip(df['region'].unique(), palette))
    df['color'] = df['region'].apply(lambda x: palette[x])
    palette = dict(zip(df[hue].unique(), df.drop_duplicates(hue)['color']))

    return palette


def plot_boxplot_each_landmark(df, metric, groupby_col='landmarks', title_name=None, ax=None):
    """
    Plot the boxplot of the metric for each landmark
    Args:
        groupby_col: Change the visuals of the plot depending on groupby_col
        - 'landmarks': show landmarks names
        - 'landmarks_id': show landmarks id
    """

    # Determin mean metric order
    grouped = df[[groupby_col, metric]].groupby(groupby_col).mean().sort_values(by=metric) # sort by median
    if 'PCC' in metric:
        grouped = grouped.sort_values(by=metric, ascending=False) 
    palette = get_palette(df.sort_values(by=['region',groupby_col]), hue=groupby_col) # sort to get colors by region

    # Boxplot the distribution of metric value for each landmark
    if ax is None:
        ax = plt.gca()
    if groupby_col == 'landmarks':
        box = sns.boxplot(x=groupby_col, y=metric, data=df, order=grouped.index, palette=palette, hue=groupby_col, legend='brief', ax=ax, showfliers=False, showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"})
        # box = sns.boxplot(x=groupby_col, y=metric, data=df, order=grouped.index, palette=palette, hue=groupby_col, legend='brief', ax=ax, showfliers=True,) # with outliers

        # Change the labels to be landmarks names
        box.legend_.remove()
        labels = [item.get_text().replace('_', ' ')  for item in box.get_xticklabels()]
        box.set_xticks(range(len(labels)))
        box.set_xticklabels(labels,rotation=90, fontsize=10)
        if metric == 'MAE': metric = r'$\Delta$' + 'BPM'
        box.set_xlabel('Landmark')
        box.set_ylabel(metric)
        
    if groupby_col == 'landmarks_id':
        box = sns.boxplot(y=groupby_col, x=metric, data=df, order=grouped.index, palette=palette, hue=groupby_col, legend='brief', ax=ax, showfliers=False, showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})

        # Change the labels to be landmarks_id. If mean MAE < 5 for the landmark, mark it in bold
        box.legend_.remove()
        top_ldmk_ids = df[[groupby_col, 'MAE']].groupby(groupby_col).mean().sort_values(by='MAE').query('MAE < 5').index.values
        for i, label in enumerate(ax.get_yticklabels()):
            if label.get_text() in top_ldmk_ids:
                label.set_fontweight('bold')
        if metric == 'MAE': metric = r'$\Delta$' + 'BPM'
        box.set_ylabel('Landmark')
        box.set_xlabel(metric)

    # Change title
    title = f"{metric} Individual landmarks ordered by mean ({title_name})"
    box.set_title(title)

    return grouped, box


def plot_tukey_results(tukey_results, ldmk_score_dict, comparison_name='glabella', thresh=False, ax=None):
    """
    Plot Tukey HSD test results for comparing landmarks' performance in HR estimation.

    Parameters:
    - tukey_results (object): Object containing Tukey HSD test results.
    - ldmk_score_dict (dict): Dictionary mapping landmarks to their performance scores.
    - comparison_name (str, optional): Name of the landmark for comparison. Defaults to 'glabella'.
    - thresh (bool): threshold at 5 BPM
    - ax (object, optional): Matplotlib axis to plot on. If not provided, a new axis will be created.
    """

    # Extract group mean 
    means = tukey_results._multicomp.groupstats.groupmean

    # Identify significant, non-significant, and potentially significant landmarks
    sigidx = [] # rejected null hypothesis
    nsigidx = []
    maybeidx = [] # confidence intervals above 5 MAE
    minrange = [means[i] - tukey_results.halfwidths[i] for i in range(len(means))]
    maxrange = [means[i] + tukey_results.halfwidths[i] for i in range(len(means))]
    if comparison_name not in tukey_results.groupsunique:
        raise ValueError('comparison_name not found in group names.')
    midx = np.where(tukey_results.groupsunique==comparison_name)[0][0]
    for i in range(len(means)):
        if tukey_results.groupsunique[i] == comparison_name:
            continue
        if (min(maxrange[i], maxrange[midx]) -
                                    max(minrange[i], minrange[midx]) < 0):
            sigidx.append(i)
        elif minrange[i] > 5:
            if thresh:
                maybeidx.append(i)
            else:
                nsigidx.append(i)
        else:
            nsigidx.append(i)
    
    # Create DataFrame
    df_tukey = pd.DataFrame({
        'landmarks': tukey_results.groupsunique,
        'mean': tukey_results._multicomp.groupstats.groupmean,
        'halfwidths':  tukey_results.halfwidths,
    })
    df_tukey['sigidx'] = df_tukey.index.isin(sigidx)
    df_tukey['nsigidx'] = df_tukey.index.isin(nsigidx)
    df_tukey['maybeidx'] = df_tukey.index.isin(maybeidx)
    df_tukey['rank_OS'] = df_tukey['landmarks'].apply(lambda x: ldmk_score_dict[x])
    df_tukey = df_tukey.sort_values(by='rank_OS', ascending=False).reset_index(drop=True)
    midx = np.argmax(df_tukey['landmarks'] == comparison_name)
    # Assign colors based on significance
    df_tukey['hue'] = np.where(df_tukey['sigidx'] == True, 'red', np.where(df_tukey['nsigidx']==True, 'grey', np.where(df_tukey['maybeidx'], 'orange', 'blue')))

    #Plot the main comparison
    ax.tick_params(axis='both', which='major', labelsize=15)
    for i, row in df_tukey.iterrows():
        ax.errorbar(x=row['mean'], y=row['landmarks'], xerr=row['halfwidths'], color=row['hue'], linestyle='None', marker='o', ecolor=row['hue'], mfc=row['hue'], mec=row['hue'])

    # Plot threshold lines
    if thresh:
        ax.axvline(5, color='orange', linestyle='--', alpha=0.7)
    ax.axvline(df_tukey['mean'].iloc[midx] - df_tukey['halfwidths'].iloc[midx], color='0.7', linestyle='--')
    ax.axvline(df_tukey['mean'].iloc[midx] + df_tukey['halfwidths'].iloc[midx], color='0.7', linestyle='--')
    
    # Set axis labels and title
    ax.set_xlabel(r'$\Delta$BPM', fontsize=15)
    ax.set_ylabel('Landmarks', fontsize=15)
    ax.set_title('Tukey HSD test', fontsize=15)

    return ax


def plot_rank_error_roi(df, metric, dataset_name, groupby_col='region', top=None, ax=None):
    """
        Plot the average ranking of MAE per video for each region combination (e.g. forehead_cheeks, nose_muchtache, ...)
    """
    if metric == 'PCC': ascending = False 
    else: ascending = True

    # Ranked by video landmarks rank average
    df['rank'] = df.sort_values(by=['videoFilename', metric]).groupby('videoFilename').cumcount(ascending=ascending) + 1
    rankings = df.groupby(groupby_col)['rank'].agg(['mean','std']).sort_values(by='mean', ascending=False).reset_index()
    if top is not None: rankings = rankings[-top:]

    if ax is None:
        ax = plt.gca()

    bar = ax.errorbar(x=rankings['mean'], y=rankings[groupby_col], xerr=rankings['std'], fmt='o', color='black', ecolor='lightgray', elinewidth=2, capsize=0)
    labels = [item.get_text().replace('_', ' ')  for item in ax.get_yticklabels()]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    ax.set_title(f'Average ranking of {metric} for combination of regions per video ({dataset_name})')    

    return rankings, bar

# TODO
def plot_metric_rank(df, metric, dataset_name, groupby_col='landmarks_id', top=10, agg_fct='mean', ax=None, fig=None):
    """ 
        Rank by overal mean/median metric and top 10. Plot metric. 
    """
    ascending = False if metric == 'PCC' else True 
    y = df.groupby([groupby_col, 'landmarks', 'ROI'])[metric].agg(['median','mean','std']).sort_values(by=agg_fct, ascending=ascending)[:top].reset_index()
    z = df[df[groupby_col].isin(y[groupby_col].unique())]
    if ax is None:
        return y

    box = sns.boxplot(x=groupby_col, y=metric, order=y.landmarks_id, data=z, showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"}, showfliers=True)
    # box = sns.violinplot(x=groupby_col, y=metric,  order=y.landmarks_id, data=z, ax=ax)
    # box.legend_.remove()
    landmarks_mapping = y.sort_values(by=agg_fct, ascending=ascending)[['landmarks','landmarks_id']].drop_duplicates()
    landmarks_names = []
    for names in landmarks_mapping['landmarks'].values:
        names = [name.replace('left_', '').replace('right_', '').strip() for name in names]
        landmarks_names.append(', '.join([name.replace('_', ' ') for name in sorted(set(names))]))
    legend_text = '\n'.join(f'{key} - {value}' for key, value in zip(landmarks_mapping['landmarks_id'].values, landmarks_names))
    t = ax.text(.65,.2,legend_text,transform=ax.figure.transFigure)
    fig.subplots_adjust(right=.62)
    if metric == 'MAE': metric = r'$\Delta$' + 'BPM'
    box.set_xlabel('Landmark')
    box.set_ylabel(metric)
    title = f"{metric} values for combined landmarks, ordered by {agg_fct} ({dataset_name})"
    box.set_title(title)
    return y 


##################### Statistical tests ########################

def compare_pvalue(df, setting, SETTINGS, metric='MAE', kruskal=False, verbose=True):
    """
        Compare the p-value between the two settings (e.g. BEARD, MOTION)
        t-value: positive when the sample mean of a is greater than the sample mean of b and negative when the sample mean of a is less than the sample mean of b.
    """
    df[setting] = False
    for dataset in SETTINGS.keys():
        df.loc[(df['dataset'] == dataset.lower()) & (df['videoIdx'].isin(SETTINGS[dataset][setting])), setting] = True
    if verbose: print(df.groupby(setting)[metric].agg(['mean', 'count']))

    res_ttest = ttest_ind(df[df[setting]==False][metric], df[df[setting]==True][metric], equal_var=False) # Welch's test, not equal variance
    if res_ttest.pvalue < 0.05:
        res_less = ttest_ind(df[df[setting]==False][metric], df[df[setting]==True][metric], equal_var=False, alternative='less').pvalue # Welch's test, not equal variance
    else:
        res_less = None   

    if kruskal:
        stat, p = scipy.stats.kruskal(df[df[setting]==False][metric], df[df[setting]==True][metric]) # Kruskal-Wallis H test, median equality
        return res_ttest.statistic, res_ttest.pvalue, res_less, stat, p
    
    return res_ttest.statistic, res_ttest.pvalue, res_less
    

def test_pvalue(df, setting, SETTINGS, condition, kruskal=False):
    """
        Test the p-value for a given setting and condition
        Args:
            setting: str (BEARD, GLASSES)
            condition: str for description (e.g. 'cheeks, jaw')
    """
    tests = []
    test = compare_pvalue(df, setting, SETTINGS, metric='MAE', kruskal=kruskal, verbose=False)
    tests.append([setting, condition, 'MAE']+list(test))

    if kruskal:
        tests = pd.DataFrame(tests, columns=['Setting', 'Condition', 'Metric', 'ttest_stat', 'ttest_pvalue', 'ttest_less', 'Kruskal_stat', 'Kruskal_pvalue'])
        significant_cols = ['ttest_pvalue', 'ttest_less', 'Kruskal_pvalue']
    else:
        tests = pd.DataFrame(tests, columns=['Setting', 'Condition', 'Metric', 'ttest_stat', 'ttest_pvalue', 'ttest_less'])
        significant_cols = ['ttest_pvalue', 'ttest_less']

    for cols in significant_cols:
        tests[cols+'_significant'] = tests[cols].apply(lambda x: x != None and float(x) < 0.05)
        tests[cols] = tests[cols].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else x)

    return tests

