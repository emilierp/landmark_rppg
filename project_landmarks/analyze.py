import numpy as np
import pandas as pd
import constants
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pyVHR.utils.errors import getErrors
from scipy.stats import ttest_ind, zscore
from scipy.stats import f_oneway, kruskal


PALETTE = 'Spectral' # "Spectral"

def get_dataset_settings(datasets):

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

def get_df_setting(df, setting, SETTINGS):
    """
        Select rows in the df that correspond to specific setting (e.g. BEARD, MOTION, STILL, ...s)
    """
    x = pd.DataFrame()
    for dataset in SETTINGS.keys():
        x = pd.concat([x, df.loc[(df['dataset'] == dataset.lower()) & (df['videoIdx'].isin(SETTINGS[dataset][setting]))]])
    return x.reset_index(drop=True)

def format_data(df):
    """
        Add video setting and person columns to the dataframe
        Convert the metric column from list to single float
        Calculate overall score by taking the average of z-scores of MAE, PCC, timePCC, timeDTW
        The score has to be minimized, so lower score (negative z score, below average) is good
    """
     # Video setting: gym, still, ...    
    df.loc[df['dataset'] == 'lgi_ppgi', 'video'] =  df.loc[df['dataset'] == 'lgi_ppgi','videoFilename'].apply(lambda x: x.split('_')[1])
    df.loc[df['dataset'] == 'mr_nirp', 'video'] =  df.loc[df['dataset'] == 'mr_nirp','videoFilename'].apply(lambda x: x.split('_')[-1])
    df.loc[df['dataset'] == 'ubfc_phys', 'video'] =  'T1'
    # Person: Subject2 (MR_NIRP), alex (LGI_PPGI), ...
    df.loc[df['dataset'] == 'lgi_ppgi', 'person'] =  df.loc[df['dataset'] == 'lgi_ppgi','videoFilename'].apply(lambda x: x.split('_')[0])
    df.loc[df['dataset'] == 'mr_nirp', 'person'] =  df.loc[df['dataset'] == 'mr_nirp','videoFilename'].apply(lambda x: x.split('_')[0])
    df.loc[df['dataset'] == 'ubfc_phys', 'person'] =  df.loc[df['dataset'] == 'ubfc_phys', 'videoFilename']
    
    for col in ['MAE','PCC']:
        try:
            df[col] = df[col].apply(lambda x: x[0]).abs()
        except:
            pass
    df['timePCC'] = df['timePCC'].apply(lambda x: np.abs(x).mean())
    metrics = ['MAE','PCC','timePCC','timeDTW']
    df[metrics] = df.groupby(['videoFilename','landmarks'])[metrics].transform('mean') # average over methods
    df = df.drop_duplicates(['videoFilename','landmarks']).reset_index(drop=True).drop(columns=['method'])  # drop duplicates over methods

    # Calculate overall score
    for metric in metrics:
        # min max normalize
        df[f'{metric}_z'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        if 'PCC' in metric: df[f'{metric}_z'] = 1-df[f'{metric}_z']
        # df[f'{metric}_z'] = zscore(df[metric])
    df['FreqScore'] = (1/2)*(df['PCC_z'] + df['MAE_z']).astype('float64')
    df['TimeScore'] = (1/2)*(df['timePCC_z'] + df['timeDTW_z']).astype('float64')
    df['score'] = (1/2)*(df['MAE_z'] + df['timeDTW_z']).astype('float64')
    df['OS'] = (1/3)*(df['MAE_z'] + df['timePCC_z'] + df['timeDTW_z']).astype('float64')
    # df['OS'] = (1/3)*(df['MAE_z'] + df['timePCC_z']*-1 + df['timeDTW_z']).astype('float64')
    # df['score'] = (df['MAE'] + (1-df['timePCC']) + df['timeDTW']).astype('float64')

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

# TODO
def evaluate_rotation_segments(df):
    """
        Get all the video segments that correspond to rotation movements
        Calculate the new errors for each file
        TODO add time domain evaluation, is pain because i need to re calculate bvps so not doing this 
    """
    times_rotation = constants.get_lgi_ppgi_rotation_segments()
    filenames = df.query('dataset == "lgi_ppgi"').videoFilename.unique()
    fps = constants.get_fps('lgi_ppgi') # does not matter anyways

    for filename in filenames:
        df = df[df['videoFilename'] == filename].copy()
        times_rot = times_rotation[filename]

        for i, row in df.iterrows():
            bpmGT, bpmES, timesGT, timesES = row[['bpmGT', 'bpmES', 'timeGT', 'timeES']].values
            # select the video segments corresponding to rotation 
            GT = dict(zip(timesGT, bpmGT))
            ES = dict(zip(timesES, bpmES))
            bpmGT_rot = np.array([value for key,value in GT.items() if key in times_rot])
            bpmES_rot = np.array([value for key,value in ES.items() if key in times_rot])
            # Calculate new errors
            RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(None, fps, bpmES_rot, bpmGT_rot, times_rot, times_rot)
            df.loc[i,'RMSE'] = RMSE[0]
            df.loc[i,'MAE'] = MAE[0]
            df.loc[i,'MAX'] = MAX[0]
            df.loc[i,'PCC'] = np.abs(PCC[0])
            df.loc[i,'CCC'] = CCC[0]
            df.at[i, 'bpmGT'] = bpmGT_rot
            df.at[i, 'bpmES'] = bpmES_rot
            df.at[i, 'timeGT'] = times_rot
            df.at[i, 'timeES'] = times_rot
    
    return df

def get_all_landmarks_in_roi(SAMPLING):
    """
        Select all rows that correspond to all landmarks in one ROI
        Files are weird so i need to merge them weirdly
    """
    df_roi = pd.DataFrame()
    for dataset in DATASETS:
        df = pd.read_hdf(f'../results/landmarks/{dataset}/{SAMPLING}/{dataset}_landmarks_in_roi_{SAMPLING}.h5', key='df') # combinations of landmarks in one ROI
        df2 = pd.read_hdf(f'../results/landmarks/{dataset}/{SAMPLING}/{dataset}_18_{SAMPLING}.h5').query("ROI != 'temple'") # add all landmarks in temple
        df_roi = pd.concat([df, df2, df_roi])
    df_roi = df_roi.reset_index(drop=True)

    df_roi['landmarks_id'] = df_roi.landmarks.astype('category').cat.codes
    df_roi['nb_landmarks'] = df_roi.landmarks.apply(lambda x: len(x))
    all_roi = df_roi.loc[df_roi.groupby('ROI').nb_landmarks.idxmax()].landmarks_id.unique()
    df_roi = df_roi[df_roi.landmarks_id.isin(all_roi)].reset_index(drop=True)

    return df_roi


##################### PLOT ########################

def get_palette(df, hue='landmarks_id'):
    # Defining the color palette
    # palette = sns.color_palette(PALETTE, n_colors=len(df[hue].unique()))
    # palette = dict(zip(df[hue].unique(), palette))

    palette = sns.color_palette(PALETTE, n_colors=len(df['ROI'].unique()))
    palette = dict(zip(df['ROI'].unique(), palette))
    df['color'] = df['ROI'].apply(lambda x: palette[x])
    palette = dict(zip(df[hue].unique(), df.drop_duplicates(hue)['color']))

    return palette

def plot_boxplot_each_landmark(df, metric, dataset_name, groupby_col='landmarks',ax=None):
    # Determin median metric order
    # grouped = df[[groupby_col, metric]].groupby(groupby_col).mean().sort_values(by=metric) # mean
    grouped = df[[groupby_col, metric]].groupby(groupby_col).median().sort_values(by=metric) # sort by median
    if 'PCC' in metric:
        grouped = grouped.sort_values(by=metric, ascending=False) 
    palette = get_palette(df.sort_values(by=['ROI',groupby_col]), hue=groupby_col) # sort to get colors by ROI

    if ax is None:
        ax = plt.gca()
    # box = sns.boxplot(x=groupby_col, y=metric, data=df, order=grouped.index, palette=palette, hue=groupby_col, legend='brief', ax=ax, showfliers=False, showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"5"})
    box = sns.boxplot(x=groupby_col, y=metric, data=df, order=grouped.index, palette=palette, hue=groupby_col, legend='brief', ax=ax, showfliers=False,)
    # box = sns.violinplot(x=groupby_col, y=metric, data=df, order=grouped.index, palette=palette, legend='brief',  hue=groupby_col, ax=ax)
    box.legend_.remove()
    labels = [item.get_text().replace('_', ' ')  for item in box.get_xticklabels()]
    box.set_xticks(range(len(labels)))
    box.set_xticklabels(labels,rotation=90, fontsize=10)
    if metric == 'MAE': metric = r'$\Delta$' + 'BPM'
    if metric == 'timeDTW': metric = 'DTW'
    if metric == 'score': metric = 'OS'

    box.set_xlabel('Landmark')
    box.set_ylabel(metric)
    if dataset_name in ['MR_NIRP', 'LGI_PPGI']: dataset_name = dataset_name.upper()
    title = f"{metric} values for individual landmarks, ordered by median ({dataset_name})"
    title = f'{dataset_name}'
    if groupby_col == 'ROI': title = f"Average rankings of {metric} for the entire ROI ({dataset_name})"
    # box.set_title(title)

    return grouped, box
    
def plot_bar_ranking(df, metric, dataset_name, groupby_col='landmarks', ax=None):
    """
        Plot the average ranking of the landmarks per video
    """
    # Average rankings of the MAE values
    df['ldmk_MAE'] = df.groupby(groupby_col)[metric].transform('mean')
    # df_agg = df.drop_duplicates('ldmk_MAE').sort_values(['ldmk_MAE'])[[groupby_col, 'ldmk_MAE']] # average MAE per landmark
    
    # Ranked by video landmarks rank average
    rankings = df.groupby('videoFilename')[metric].rank(ascending=True).groupby(df[groupby_col]).mean().sort_values().reset_index()
    if 'PCC' in metric:
        rankings = df.groupby('videoFilename')[metric].rank(ascending=False).groupby(df[groupby_col]).mean().sort_values().reset_index()
    palette = get_palette(df.sort_values(by=['ROI',groupby_col]), hue=groupby_col)
    
    if ax is None:
        ax = plt.gca()
    bar = sns.barplot(x=groupby_col, y=metric, data=rankings, palette=palette, hue=groupby_col, ax=ax)
    labels = [item.get_text().replace('_', ' ')  for item in ax.get_xticklabels()]
    bar.set_xticks(range(len(labels)))
    bar.set_xticklabels(labels, rotation=90, fontsize=6)
    if metric == 'MAE': metric = r'$\Delta$' + 'BPM'
    bar.set_ylabel(metric)
    bar.set_xlabel('Landmark')
    if dataset_name in ['MR_NIRP', 'LGI_PPGI']: dataset_name = dataset_name.upper()
    title = f"Average rankings of the {metric} values for individual landmarks ({dataset_name})"
    if groupby_col == 'ROI': title = f"Average rankings of {metric} for the entire ROI ({dataset_name})"
    bar.set_title(title)

    # Legend with colors
    # patches = [Patch(color=v, label=k) for k, v in palette.items()]
    # plt.legend(title=groupby_col, labels=palette.keys(), handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, fontsize=8, frameon=False)


    return rankings, bar

# TODO
def plot_bar_ranking_2(df, metric, dataset_name, groupby_col='landmarks', ax=None):
    """
        Plot the average ranking of MAE per landmarks
    """
    # Average rankings of the MAE values
    df['ldmk_MAE'] = df.groupby(groupby_col)[metric].transform('mean')
    # df_agg = df.drop_duplicates('ldmk_MAE').sort_values(['ldmk_MAE'])[[groupby_col, 'ldmk_MAE']] # average MAE per landmark
    
    # Rank by average MAE per landmarks
    rankings = df[[groupby_col, 'ldmk_MAE']].drop_duplicates().sort_values(by='ldmk_MAE', ascending=True).reset_index(drop=True).reset_index().rename({'index':'rank'}, axis=1)
    rankings['rank'] = rankings['rank'] + 1
    # rankings = df.groupby('videoFilename')[metric].rank(ascending=True).groupby(df[groupby_col]).mean().sort_values().reset_index()
    if metric == 'PCC':
        rankings = df.groupby('videoFilename')[metric].rank(ascending=False).groupby(df[groupby_col]).mean().sort_values().reset_index()
    palette = get_palette(df.sort_values(by=['ROI',groupby_col]), hue=groupby_col)
    
    if ax is None:
        ax = plt.gca()
    bar = sns.barplot(x=groupby_col, y='rank', data=rankings, palette=palette, hue=groupby_col, ax=ax)
    labels = [item.get_text().replace('_', ' ')  for item in ax.get_xticklabels()]
    bar.set_xticks(range(len(labels)))
    bar.set_xticklabels(labels, rotation=90, fontsize=6)
    title = f"Average rankings of the {metric} values for individual landmarks ({dataset_name})"
    if groupby_col == 'ROI': title = f"Average rankings of {metric} for the entire ROI ({dataset_name})"
    bar.set_title(title)

    # Legend with colors
    # patches = [Patch(color=v, label=k) for k, v in palette.items()]
    # plt.legend(title=groupby_col, labels=palette.keys(), handles=patches, bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0, fontsize=8, frameon=False)

    return rankings, bar

def plot_rank_error_roi(df, metric, dataset_name, groupby_col='ROI', top=None, ax=None):
    """
        Plot the average ranking of MAE per video for each ROI combination (e.g. forehead_cheeks, nose_muchtache, ...)
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
    ax.set_title(f'Average ranking of {metric} for combination of ROIs per video ({dataset_name})')    

    return rankings, bar

def plot_rank_error_landmarks(df, metric, dataset_name, groupby_col='landmarks_id', top=None, ax=None, fig=None):
    """
        Plot the average ranking of MAE per video for each landmarks combination (e.g. glabella_chin, ...)
        Legend: Combine left and right landmarks as one
    """
    if metric == 'PCC': ascending = False 
    else: ascending = True

    # Ranked by video landmarks rank average
    df['rank'] = df.sort_values(by=['videoFilename', metric]).groupby('videoFilename').cumcount(ascending=ascending) + 1
    rankings = df.groupby(groupby_col)['rank'].agg(['mean','std']).sort_values(by='mean', ascending=False).reset_index()
    if top is not None: rankings = rankings[-top:]
    rankings = rankings.merge(df[['landmarks_id','landmarks']].drop_duplicates(), on='landmarks_id', how='left').sort_values(by='mean', ascending=ascending).reset_index(drop=True)
    rankings[groupby_col] = rankings.index + 1
    rankings = rankings.sort_values(by='mean', ascending=not ascending).reset_index(drop=True)

    if fig is None: fig, ax = plt.gcf(), plt.gca()
    bar = ax.errorbar(x=rankings['mean'], y=rankings[groupby_col].astype('str'), fmt='o', color='black', capsize=0)

    # bar = ax.errorbar(x=rankings['mean'], y=rankings[groupby_col].astype('str'), xerr=rankings['std'], fmt='o', color='black', ecolor='lightgray', elinewidth=2, capsize=0)
    ax.set_title(f'Average ranking of {metric} for combination of ROIs per video ({dataset_name})')  
    ax.set_xlabel(f'Average ranking of {metric}')  
    ax.set_ylabel(f'Landmarks combination')

    # Legend: combine right and left landmarks  
    landmarks_mapping = rankings.sort_values(by='mean', ascending=ascending)[['landmarks','landmarks_id']].drop_duplicates().reset_index(drop=True)
    landmarks_names = []
    for names in landmarks_mapping['landmarks'].values:
        names = [name.replace('left_', '').replace('right_', '').strip() for name in names]
        landmarks_names.append(', '.join([name.replace('_', ' ') for name in sorted(set(names))]))
    legend_text = '\n'.join(f'{key} - {value}' for key, value in zip(landmarks_mapping['landmarks_id'].values, landmarks_names))
    t = ax.text(.65,.2,legend_text,transform=ax.figure.transFigure)
    fig.subplots_adjust(right=.62)

    return rankings, bar, fig

def plot_each_landmark(df, metric, dataset_name, groupby_col='landmarks', axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    grouped, box = plot_boxplot_each_landmark(df, metric, dataset_name, groupby_col, ax=axs[0])
    rankings, bar = plot_bar_ranking(df, metric, dataset_name, groupby_col, ax=axs[1])

    return df, grouped, rankings, box, bar

def plot_metric_rank(df, metric, dataset_name, groupby_col='landmarks_id', top=10, agg_fct='mean', ax=None, fig=None):
    """ 
        Rank by overal mean/median metric and top 10. Plot metric
    """

    ascending = False if metric == 'PCC' else True 
    y = df.groupby([groupby_col, 'landmarks'])[metric].agg(['median','mean','std']).sort_values(by=agg_fct, ascending=ascending)[:top].reset_index()
    y[groupby_col] = y.index + 1
    y = y.sort_values(by=agg_fct, ascending=ascending).reset_index(drop=True)

    # bar = ax.errorbar(x=y[groupby_col].astype('str'), y=y[agg_fct], fmt='o', color='black', capsize=0)
    # bar = ax.errorbar(x=y[groupby_col].astype('str'),  y=y['mean'], yerr=y['std'], fmt='o', color='black', ecolor='lightgray', elinewidth=2, capsize=0)
    bar = ax.bar(x=y[groupby_col].astype('str'), height=y[agg_fct], yerr=y['std'], color='lightblue', capsize=0)
    ax.set_xticks(range(0,top+1))
    ax.set_xlabel('Landmarks')
    ax.set_ylabel(metric)
    landmarks_mapping = y.sort_values(by=agg_fct, ascending=ascending)[['landmarks','landmarks_id']].drop_duplicates()
    landmarks_names = []
    for names in landmarks_mapping['landmarks'].values:
        names = [name.replace('left_', '').replace('right_', '').strip() for name in names]
        landmarks_names.append(', '.join([name.replace('_', ' ') for name in sorted(set(names))]))
    legend_text = '\n'.join(f'{key} - {value}' for key, value in zip(landmarks_mapping['landmarks_id'].values, landmarks_names))
    t = ax.text(.65,.2,legend_text,transform=ax.figure.transFigure)
    fig.subplots_adjust(right=.62)
    ax.set_title(f'Average {metric} for top {top} combination of ROIs (Still) (ranked by metric)')  

    return y 


# TODO
def plot_bubble_roi(df, metric, dataset_name, size=[1200,400]):
    """
        Plot MAE for different combinations of landmarks within a ROI, for all ROIs
        Scratch: If plotting against time requirement, use x=ldmk_time
    """

    df = df.copy()
    df['ldmk_MAE'] = df.groupby('landmarks')['MAE'].transform('mean')
    df['ldmk_size'] = df['landmarks'].apply(lambda x: len(x))
    if len(df.loc[0,'ROI'].split('_'))> 1: # testing combination of ROIs
        df['landmarks'] = df['ROI'].apply(lambda x: ', '.join((x.split('_'))))
    else: # testion single ROI and multiple landmarks
        df['landmarks'] = df.landmarks.apply(lambda x: ', '.join(x).replace('_',' ').replace('(','').replace(')',''))    
    df['landmarks_id'] = df['landmarks'].astype('category').cat.codes
    df['landmarks'] = df['landmarks_id'].astype('str') + ': ' + df['landmarks'] 
    df = df.sort_values(by='landmarks_id').reset_index(drop=True)

    # Defining consistent color palette between seaborn and plotly
    palette = get_palette(df.sort_values(by=['ROI']), hue='ROI')
    rgb_colors = [f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})' for color in palette.values()] 
    palette = dict(zip(df['ROI'].unique(), rgb_colors))
    
    fig = px.scatter(df, x="ROI", y="ldmk_MAE", color="ROI", size='ldmk_size', hover_data=['landmarks'],
                    template='none', color_discrete_map=palette, width=size[0], height=size[1],
                    labels={'ldmk_MAE':'MAE (bpm)', 'landmarks':'Landmarks', 'ldmk_size':'Number of landmarks'},
                    title=f"MAE for each ROI and different landmarks ({dataset_name})",) #  text='landmarks_id',
    # fig.update_traces(textposition='middle center', textfont_size=12, textfont_family='Courier New') 
    fig.update_xaxes(showline=True, linecolor='black', mirror=True) # linewidth=2, 
    fig.update_yaxes(showline=True, linecolor='black', mirror=True)
    
    return df, fig


def plot_best_ldmk_in_all_roi(df, metric, setting, verbose=False, fig=None, ax=None):
    """
        1. Rank the landmarks by best landmarks per video and ROI
        2. Take the most common best landmarks per ROI
        3. Plot the metric for the best landmarks per ROI
    """

    if metric == 'PCC' or metric == 'timeDTW': ascending = False 
    else: ascending = True

    # Rank by best landmarks per video and ROI 
    df['rank'] = df.sort_values(by=['videoFilename', metric]).groupby(['videoFilename','ROI']).cumcount(ascending=ascending) + 1

    # most common best ranking landmarks per ROI
    best_landmarks = df.query('rank == 1').groupby('ROI').landmarks.value_counts().sort_values(ascending=False).groupby('ROI').head(1).reset_index()
    if verbose: print(best_landmarks)

    # Take all rows that correspond to the best landmarks per ROI
    df = df[df['landmarks'].isin(best_landmarks.landmarks.unique())]
    df = df.assign(ROI_MAE=df.groupby('ROI')[metric].transform('mean')).sort_values(by='ROI_MAE').reset_index(drop=True)
    best_landmarks = best_landmarks.merge(df[['landmarks','ROI_MAE']].drop_duplicates(), on='landmarks', how='left').sort_values(by='ROI_MAE').reset_index(drop=True)

    palette = get_palette(df.sort_values(by=['ROI','landmarks']), hue='ROI')
    box = sns.boxplot(x='ROI', y=metric, data=df, palette=palette, hue='ROI', ax=ax)
    landmarks_names = []
    for names in best_landmarks['landmarks'].values:
        names = [name.replace('left_', '').replace('right_', '').strip() for name in names]
        landmarks_names.append(', '.join([name.replace('_', ' ') for name in sorted(set(names))]))
    legend_text = '\n'.join(f'{key} - {value}' for key, value in zip(best_landmarks['ROI'].values, landmarks_names))
    t = ax.text(.65,.2,legend_text,transform=ax.figure.transFigure)
    fig.subplots_adjust(right=.62)
    box.set_title(f'Landmarks that most performed best for each ROI ({setting})')

    return df, box


##################### Sanity checks ########################

def plot_zscore_hist(df):
    plt.hist(df['MAE_z'], alpha=0.5, label='MAE')
    plt.hist(df['timeDTW_z'], alpha=0.5,    label='timeDTW')
    plt.hist(df['PCC_z'], alpha=0.5,       label='PCC')
    plt.hist(df['score'], alpha=0.5,     label='score')
    plt.legend()

def compare_p_value(df, setting, SETTINGS, metric='MAE', verbose=True):
    """
        Compare the p-value between the two settings (e.g. BEARD, MOTION)
    """
    df[setting] = False
    for dataset in SETTINGS.keys():
        df.loc[(df['dataset'] == dataset.lower()) & (df['videoIdx'].isin(SETTINGS[dataset][setting])), setting] = True
    if verbose: print(df.groupby(setting)[metric].agg(['mean', 'count']))

    res = ttest_ind(df[df[setting]==False][metric], df[df[setting]==True][metric], equal_var=False) # Welch's test, not equal variance
    print(f"{setting}: p-value is {res.pvalue}, so p-value < 0.05 is {res.pvalue < 0.05} (if True then the difference is significant)")
    if res.pvalue < 0.05:
        res = ttest_ind(df[df[setting]==False][metric], df[df[setting]==True][metric], equal_var=False, alternative='less') # Welch's test, not equal variance
        print(f"{setting}: p-value = {res.pvalue} < 0.05 is {res.pvalue < 0.05}: (if True then False_mean < True_mean)")

    stat, p = f_oneway(df[df[setting]==False][metric], df[df[setting]==True][metric],) # ANOVA
    print(f"{setting}: p-value is {p}, so p-value < 0.05 is {p < 0.05} (if True then the difference is significant)")

    return res


def compare_pvalue(df, setting, SETTINGS, metric='MAE', verbose=True):
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
    # stat, p = f_oneway(df[df[setting]==False][metric], df[df[setting]==True][metric],) # ANOVA
    stat, p = kruskal(df[df[setting]==False][metric], df[df[setting]==True][metric]) # Kruskal-Wallis H test, median equality

    return res_ttest.statistic, res_ttest.pvalue, res_less, stat, p

def test_pvalue(df, setting, SETTINGS, condition):
    """
        Test the p-value for a given setting and condition
        Args:
            setting: str (BEARD, GLASSES)
            condition: str for description (e.g. 'cheeks, jaw')
    """
    tests = []
    for metric in ['MAE', 'timeDTW', 'score']:
        test =  compare_pvalue(df, setting, SETTINGS, metric=metric, verbose=False)
        tests.append([setting, condition, metric]+list(test))

    tests = pd.DataFrame(tests, columns=['Setting', 'Condition', 'Metric', 'ttest_stat', 'ttest_pvalue', 'ttest_less', 'Kruskal_stat', 'Kruskal_p'])
    for cols in ['ttest_pvalue', 'ttest_less', 'Kruskal_p']:
        tests[cols+'_significant'] = tests[cols].apply(lambda x: x != None and float(x) < 0.05)
        tests[cols] = tests[cols].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else x)

    return tests