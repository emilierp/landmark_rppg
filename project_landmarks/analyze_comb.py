import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import constants
import pyVHR

top_landmarks = ['glabella', 'upper_nasal_dorsum', 'lower_medial_forehead', 'soft_triangle', 'malar', 'lower_lateral_forehead', 'nasal_tip']



def eda_combination(x, sym_ldmks, agg_fct='mean', metric='score'):
    from IPython.display import display

    z = x.groupby(['landmarks_id', 'landmarks', 'ROI'])[metric].agg(['median','mean','std']).sort_values(by=agg_fct, ascending=True).reset_index()
    assert z.shape[0] == len(x.landmarks.unique())  
    z['landmarks_names'] = z['landmarks'].apply(lambda x: set([name.replace('left_', '').replace('right_', '').strip() for name in x]))
    z['landmarks_len'] = z['landmarks_names'].apply(lambda x: len(x))
    print(f"Overall: Mean value for OS going from {z['mean'].min()} to {z['mean'].max()}")
    # print(f"Overall: Median value for OS going from {z['median'].min()} to {z['median'].max()}")

    top = int(len(z)*0.1)
    print(f"Top {top} landmarks (10%) from {len(z)} landmarks")  
    y = z[:top].copy()

    for length in sorted(x.landmarks_len.unique()):
        # print(f"Landmarks with {length} landmarks: {y.query('landmarks_len == @length').shape[0] / len(y)} {y.query('landmarks_len == @length').shape[0] / z.query('landmarks_len == @length').shape[0]}") 
        print(f"{int(y.query('landmarks_len == @length').shape[0] * 100 / len(y))}" + ' \%  & ', end='')

    print(f"Mean value for OS going from {y['mean'].min()} to {y['mean'].max()}")
    # print(f"Median value for OS going from {y['median'].min()} to {y['median'].max()}")

    rois = pyVHR.extraction.CustomLandmarks().get_face_regions().keys()
    for roi in rois:
        y[roi] = y['ROI'].apply(lambda x: roi in x).sum() / y.shape[0]
        
    display(y[rois].drop_duplicates().sort_values(by=0, axis=1, ascending=False))

    rois = z.ROI.unique().tolist()
    for roi in rois:
        y[roi] = (y['ROI'] == roi).sum() / y.shape[0]
        
    display(y[rois].drop_duplicates().sort_values(by=0, axis=1, ascending=False))

    for ldmk in sym_ldmks:
        y[ldmk] = y['landmarks_names'].apply(lambda x: ldmk.replace('_', ' ') in x).sum() / y.shape[0]
    display(y[sym_ldmks].drop_duplicates().sort_values(by=0, axis=1, ascending=False))

    return z, y


def plot_metric_rank2(df, metric, dataset_name, groupby_col='landmarks_id', top=10, agg_fct='mean', ax=None, fig=None):
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


def agg_ldmk_comb(y, agg_col, case='ind_ldmk'):
    t = y[agg_col].drop_duplicates().sort_values(by=0, axis=1, ascending=False).copy().T.reset_index().rename({0: 'Frequency Distribution', 'index':'Landmarks'}, axis=1)
    if case == 'ind_ldmk':
        rois = pyVHR.extraction.CustomLandmarks().get_face_regions()
        for roi in list(rois.keys()):
            rois[roi] = [name.replace('left_', '').replace('right_', '') for name in rois[roi]]
            t.loc[t['Landmarks'].isin(rois[f'{roi}']),'ROI'] = roi    
        t['Landmarks'] = t['Landmarks'].apply(lambda x: x.replace('_', ' '))
    if case == 'comb_ldmk':
        t['ROI'] = t['Landmarks'].apply(lambda x: ', '.join([roi[0].upper() for roi in x.split('_')]))

    return t

def freq_ldmk_comb(t_list, col='Landmarks', hue_col='setting', figsize=(10,5)):
    """
        col: 'Landmarks' or 'ROI'BytesWarning
    """

    palette = ['mediumvioletred', 'mediumblue', 'mediumseagreen'][:len(t_list)]

    t = pd.concat(t_list)
    plt.figure(figsize=figsize)
    if col == 'Landmarks':
        sns.barplot(x='Frequency Distribution', y='Landmarks', hue=hue_col, data=t.sort_values('ROI'), palette=palette)
    if col == 'ROI':
        sns.barplot(x='Frequency Distribution', y='ROI', hue=hue_col, data=t[t['Frequency Distribution'] > 0], palette=palette)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.xlabel('Frequency Distribution', fontsize=15)
    plt.ylabel('Landmarks', fontsize=15)
    plt.legend(fontsize=15)
    # plt.figure()
    # rois = constants.get_rois()
    # palette = sns.color_palette(PALETTE, n_colors=len(rois.keys()))
    # palette = dict(zip(sorted(rois.keys()), palette))
    # sns.catplot(x='Frequency Distribution', y='Landmarks', hue='ROI', col='setting', data=t.sort_values('ROI'), kind='bar', palette=palette)

    return t
