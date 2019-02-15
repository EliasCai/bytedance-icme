# -- coding:utf-8 --

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from features import uid_features, author_features, music_features, normalize_features
from data_io import read_final_track2_train, read_final_track2_test, read_track2_title
from data_io import map_title, timer

SEED = 2019


def train_ridge(x_train, x_valid, y_train, y_valid):
    # print('linear_model')
    preds = []
    clf = RidgeCV(alphas=[1, 0.1, 0.01, 0.001])
    # clf = LassoCV(alphas = [1, 0.1, 0.01, 0.001])
    # clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    preds.append(y_pred.reshape(-1, 1))

    # preds = np.hstack(preds)
    # print(roc_auc_score(y_valid, preds.mean(1)))

    # y_pred = clf.predict(x_valid)
    print(roc_auc_score(y_valid, y_pred))

    return clf


def tf_roc(): # t

    x_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.8, 0.9, 1]
    y_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    x_2 = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    y_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    x_placeholder = tf.placeholder(tf.float64, [10])
    y_placeholder = tf.placeholder(tf.bool, [10])
    auc = tf.metrics.auc(labels=y_placeholder, predictions=x_placeholder)
    initializer = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    with tf.Session() as sess:

        for i in range(3):
            sess.run(initializer)
            auc_value, update_op = sess.run(
                auc, feed_dict={
                    x_placeholder: x_1,
                    y_placeholder: y_1
                })
            print('auc_1: ' + str(auc_value) + ", update_op: " +
                  str(update_op))
            print(roc_auc_score(y_1, x_1))
            sess.run(initializer)
            auc_value, update_op = sess.run(
                auc, feed_dict={
                    x_placeholder: x_2,
                    y_placeholder: y_2
                })
            print('auc_2: ' + str(auc_value) + ", update_op: " +
                  str(update_op))

            print(roc_auc_score(y_2, x_2))


def draw_roc(y_valid, y_pred):

    fpr, tpr, _ = roc_curve(y_valid, y_pred)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')


if __name__ == "__main__":

    with timer("1. reading final track2 data"):
        df = read_final_track2_train(100000)
        df_test = read_final_track2_test(100000)

        df_feat, df_model = train_test_split(
            df, random_state=SEED, shuffle=True, test_size=0.3
        )  # some for generating feature, some for traing model

    with timer("2. creating features of uid and author"):
        df_uid_feature = uid_features(df_feat)
        df_author_feature = author_features(df_feat)
        # df_music_feature = music_features(df_feat)

    # with timer("read_track2_title"):
    # item_id, seq = read_track2_title()

    # with timer("map title with train data"):
    # seq_order = map_title(df, item_id, seq)

    with timer("3. normalizing the features"):
        df_model, df_test = normalize_features(df_model, df_test)

        df_model = df_model.merge(df_uid_feature, on='uid', how='left')
        df_test = df_test.merge(df_uid_feature, on='uid', how='left')

        df_model = df_model.merge(
            df_author_feature, on='author_id', how='left')
        df_test = df_test.merge(df_author_feature, on='author_id', how='left')
        # df_model = df_model.merge(df_music_feature, on='music_id', how='left')

        df_model = df_model.fillna(df_model.mean())
        df_test = df_test.fillna(df_test.mean())

        df_train, df_valid = train_test_split(
            df_model, random_state=SEED, shuffle=False, test_size=0.2)

    with timer("4. training the model"):
        col_feat = [
            'duration_time', 'uid_view', 'uid_finish', 'uid_like',
            'author_view', 'author_finish', 'author_like'
        ]
        x_train = df_train.loc[:, col_feat].values
        x_valid = df_valid.loc[:, col_feat].values


        finish_train, finish_valid = \
            df_train['finish'].values, df_valid['finish'].values
        like_train, like_valid = \
            df_train['like'].values, df_valid['like'].values

        y_train, y_valid = finish_train, finish_valid
        model_finish = train_ridge(x_train, x_valid, y_train, y_valid)
        y_train, y_valid = like_train, like_valid
        model_like = train_ridge(x_train, x_valid, y_train, y_valid)
        # y_pred = model.predict(x_valid)
        # print(roc_auc_score(y_valid, y_pred))

        # draw_roc(y_valid, y_pred)
    with timer("5. predicting the result of test set"):
        x_test = df_test.loc[:, col_feat].values
        df_test['finish_probability'] = model_finish.predict(x_test)
        df_test['like_probability'] = model_like.predict(x_test)
        df_test['finish_probability'] = df_test['finish_probability'].clip(0, 1)
        df_test['like_probability'] = df_test['like_probability'].clip(0, 1)
        df_test.loc[:,['uid','item_id','finish_probability','like_probability']].to_csv('output/result.csv',index=False)
