# -- coding:utf-8 --

import pandas as pd
import numpy as np
import json
from contextlib import contextmanager
import time


def read_chunk(reader, chunkSize):
    chunks = []
    # while True:
    for i in range(10):
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            print("Iteration is stopped.")
            break
    df = pd.concat(chunks, ignore_index=True)

    return df


def read_final_track2_train(chunkSize=100000):
    loop = True
    path = 'input/final_track2_train.txt'

    cols = [
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
        'finish', 'like', 'music_id', 'device', 'time', 'duration_time'
    ]
    reader = pd.read_csv(path, iterator=True, sep='\t')
    df = read_chunk(reader, chunkSize)
    df.columns = cols
    return df


def read_track2_title(chunkSize=4000000, maxlen=10):

    path = 'input/track2_title.txt'  # 3114071 rows

    item_id = np.zeros((chunkSize, 1)).astype(np.int)
    seq = np.zeros((chunkSize, maxlen)).astype(np.int)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= chunkSize:
                break
            content = json.loads(line)
            words = list(map(int, list(content['title_features'].keys())))
            col_num = min(10, len(words))
            item_id[i] = int(content['item_id'])
            seq[i, :col_num] = words[:col_num]
    return item_id[:i], seq[:i]


def map_title(df, item_id, seq):

    match = pd.DataFrame({
        'item_id': item_id.reshape(-1),
        'idx': range(item_id.shape[0])
    })
    match = df.merge(match, on='item_id', how='left')
    idx_null = match['idx'].isnull()
    idx = match[~idx_null]['idx'].astype(int).values
    return seq[idx]


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


if __name__ == "__main__":

    with timer("read_final_track2_train"):
        df = read_final_track2_train(100000)

        df_feat, df_model = train_test_split(
            df, random_state=SEED, shuffle=False,
            test_size=0.5)  # half for feature, half for traing model

    with timer("creating features of uid and author"):
        df_uid_feature = uid_features(df_feat)
        df_author_feature = author_features(df_feat)

    # with timer("read_track2_title"):
    # item_id, seq = read_track2_title()

    # with timer("map title with train data"):
    # seq_order = map_title(df, item_id, seq)

    with timer("normalize the features"):
        df_model = normalize_features(df_model)
        df_model = df_model.merge(df_uid_feature, on='uid', how='left')
        df_model = df_model.merge(
            df_author_feature, on='author_id', how='left')
        df_model = df_model.fillna(df_model.mean())

    df_train, df_valid = train_test_split(
        df_model, random_state=SEED, shuffle=False, test_size=0.2)

    col_feat = [
        'duration_time', 'uid_view', 'uid_finish', 'uid_like', 'author_view',
        'author_finish', 'author_like'
    ]
    x_train, x_valid = df_train.loc[:, col_feat].values, df_valid.loc[:, col_feat].values
    finish_train, finish_valid = df_train['finish'].values, df_valid['finish'].values
    like_train, like_valid = df_train['like'].values, df_valid['like'].values
    # for col in df.columns:
    # print(df[col].drop_duplicates().count(), df[col].min(), df[col].max())
    # print(df[col].astype(str).describe())

    # df.groupby('uid').count()['user_city'].describe()
    # df['user_city'].astype(str).value_counts().describe()
    # df.loc[:,['uid','author_id']].drop_duplicates()
