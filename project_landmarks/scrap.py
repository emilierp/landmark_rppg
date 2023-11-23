##### PLOTS #####

def plot_bubble_roi(df, metric, dataset_name, roi, fig, axs, title=None, size=[1500,500]):

    df = df.copy()
    if df[metric].dtype == 'object':
        df[metric] = df[metric].apply(lambda x: x[0])
    if dataset_name == 'MR_NIRP':
        df['video'] = df.videoFilename.apply(lambda x: x.split('_')[1])
    if dataset_name == 'LGI_PPGI':
        df['video'] = df['videoFilename'].apply(lambda x: x.split('_')[-1])
    df['ldmk_time'] = df.groupby('landmarks')['TIME_REQUIREMENT'].transform('sum')
    df['ldmk_MAE'] = df.groupby('landmarks')['MAE'].transform('mean')
    df['ldmk_size'] = df['landmarks'].apply(lambda x: len(x))
    if len(df.loc[0,'dataset'].split('_'))> 3: # testing combination of ROIs
        df['landmarks'] = df['dataset'].apply(lambda x: ', '.join((x.split('_')[2:])))
    else: # testion single ROI and multiple landmarks
        df['landmarks'] = df.landmarks.apply(lambda x: ', '.join(x).replace('_',' ').replace('(','').replace(')',''))
    df['landmarks_id'] = df['landmarks'].astype('category').cat.codes
    df['landmarks'] = df['landmarks_id'].astype('str') + ': ' + df['landmarks'] 
    df = df.sort_values(by='landmarks_id').reset_index(drop=True)

    palette = get_palette(df, 'landmarks_id')
    ax = sns.scatterplot(data=df, x='ldmk_time', y='ldmk_MAE', hue='landmarks_id', size='ldmk_size', sizes=(100, 400), 
                        palette=palette, edgecolors='grey', linewidths=0.3, ax=axs)
    ax.set(xlabel='Time  requirement (s)', ylabel='MAE (bpm)', title=f"MAE vs Time requirement for each landmark ({dataset_name.upper()})")

    # annotate
    groups = df.drop_duplicates(subset=['landmarks_id'])
    for i, txt in enumerate(df.landmarks_id.unique()):
        ax.annotate(txt, (groups.ldmk_time.iloc[i], groups.ldmk_MAE.iloc[i]), fontsize=9)

    # legend
    ax.legend_.remove()
    handles, labels = get_handle_labels(df, ax, redundant=False)
    fig.legend(handles[:len(df.landmarks.unique())+1],labels[:len(df.landmarks.unique())+1], loc='right', bbox_to_anchor=(2.05, 0.5), ncol=1)
    fig.legend(handles[len(df.landmarks.unique())+2:],labels[len(df.landmarks.unique())+2:],loc='right', ncol=len(df.ldmk_size.unique())+1,
                bbox_to_anchor=(1.8, 0.2),title='Number of landmarks')
    

    return df, fig, ax