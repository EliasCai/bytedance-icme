# -- coding:utf-8 --

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter

# cluster the video by feature embedding
# covert label to one hot


def uid_features(df):

    ss = StandardScaler()

    # 1. number of view
    group = df['uid'].astype(str).value_counts()
    clip_75 = group.describe()['75%']
    df_view = group.reset_index().astype('int')
    df_view = df_view.rename(columns={'index': 'uid', 'uid': 'uid_view'})

    # 2. mean of finish and like
    df_target = df.groupby(
        'uid').mean().loc[:, ['finish', 'like']].reset_index()

    df_target = df_target.rename(columns={
        'finish': 'uid_finish',
        'like': 'uid_like'
    })
    # 3. Standardize data
    df_view['uid_view'] = df_view['uid_view'].clip(0, clip_75)
    df_view['uid_view'] = ss.fit_transform(df_view['uid_view'].values.reshape(
        -1, 1))

    df_feature = df_view.merge(df_target, on='uid', how='left')

    return df_feature


def author_features(df):

    ss = StandardScaler()

    # 1. number of view
    group = df['author_id'].astype(str).value_counts()
    clip_75 = group.describe()['75%']
    df_view = group.reset_index().astype('int')
    df_view = df_view.rename(columns={
        'index': 'author_id',
        'author_id': 'author_view'
    })

    # 2. mean of finish and like
    df_target = df.groupby(
        'author_id').mean().loc[:, ['finish', 'like']].reset_index()
    df_target = df_target.rename(columns={
        'finish': 'author_finish',
        'like': 'author_like'
    })

    # 3. Standardize data
    df_view['author_view'] = df_view['author_view'].clip(0, clip_75)
    df_view['author_view'] = ss.fit_transform(
        df_view['author_view'].values.reshape(-1, 1))

    df_feature = df_view.merge(df_target, on='author_id', how='left')

    return df_feature


def normalize_features(df1):

    df = df1.copy()
    ss = StandardScaler()

    dict_music_id = dict([(-1, 0), (25, 1), (110, 2), (468, 3), (33, 4), (57,5),
                          (43, 6), (238, 7), (307, 8)]) # yapf: disable

    c = Counter(df['device'].sample(frac=0.2).values)  # use sample to speed up
    common_device = c.most_common(2000)  #
    # dict_device = dict([(30545, 0), (30971, 1), (751, 2), (44097, 3), (1693,4),
    # (4108, 5), (17745, 6), (2052, 7), (2254, 8)]) # yapf: disable
    dict_device = dict([(k[0],v+1) for v, k in enumerate(common_device)])
    df['duration_time'] = ss.fit_transform(df['duration_time'].values.reshape(
        -1, 1))

    df['music_id'] = df['music_id'].map(
        lambda x: dict_music_id[x] if x in dict_music_id else 9)

    df['device'] = df['device'].map(
        lambda x: dict_device[x] if x in dict_device else 0)

    df['channel'] = df['channel'].clip(0, 2)  # convert value > 3 to 2

    concat_list = [
        # pd.get_dummies(df['channel'], prefix='channel'),
        # pd.get_dummies(df['device'], prefix='device'),
        # pd.get_dummies(df['music_id'], prefix='music_id')
    ]
    concat_list.append(df)
    df = pd.concat(concat_list, axis=1)
    # from features import normalize_features; df2 = normalize_features(df)
    return df


if __name__ == "__main__":
    pass
