# -- coding:utf-8 --
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, roc_auc_score

from features import uid_features, author_features, normalize_features
from data_io import read_final_track2_train, read_track2_title, map_title, timer

SEED = 2019


def train_ridge(x_train, x_valid, y_train, y_valid):

    clf = Ridge()
    clf.fit(x_train, y_train) 
    
    y_pred = clf.predict(x_valid)
    # print(auc(y_valid, y_pred))
    print(roc_auc_score(y_valid, y_pred))
    return clf

if __name__ == "__main__":

    with timer("1. reading final_track2_train.txt"):
            df = read_final_track2_train(100000)

            df_feat, df_model = train_test_split(
                df, random_state=SEED, shuffle=False,
                test_size=0.5)  # half for generating feature, half for traing model
    
    with timer("2. creating features of uid and author"):
        df_uid_feature = uid_features(df_feat)
        df_author_feature = author_features(df_feat)

    # with timer("read_track2_title"):
    # item_id, seq = read_track2_title()

    # with timer("map title with train data"):
    # seq_order = map_title(df, item_id, seq)

    with timer("3. normalizing the features"):
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
    
    model = train_ridge(x_train, x_valid, finish_train, finish_valid)