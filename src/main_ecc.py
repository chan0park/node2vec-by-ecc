from utils import *
from plot import *
import pickle
import os
import argparse
import random

parser = argparse.ArgumentParser(description='Control mode of code.')
parser.add_argument('-load_all', action='store_true', default=False)
parser.add_argument('-marktime', action='store_true', default=False)
parser.add_argument('-plot_density', action='store_true', default=False)
parser.add_argument('-shade', action='store_true')
parser.add_argument('-classification', action='store_true', default=False)
parser.add_argument('-regression', action='store_true', default=False)
parser.add_argument('-classification_30', action='store_true', default=False)
parser.add_argument('-regression_30', action='store_true', default=False)
parser.add_argument('-embedding_split_10', action='store_true', default=False, help="Split users in 10 groups and make separate edgelist file")
parser.add_argument('-split_n', action='store', default=10, help='Number of bins')
args = parser.parse_args()

if args.load_all:
    df_30 = check_and_import('./df/df_30')
    if df_30 is None:
        df_30 = import_30("./data/30_eventinfo.csv")
        df_30 = mark_timewindow(df_30)
        save_pickle(df_30, 'df_30')

if args.marktime:
    df_ml = mark_timewindow(df_ml)
    df_30 = mark_timewindow(df_30) 

df_ml_iu = check_and_import('./df/df_ml')
if df_ml_iu is None:
    df_ml_iu = import_ml("./data/ratings.csv")
    df_ml_iu = mark_timewindow(df_ml_iu)
    df_ml_iu.columns = ['uid','id','feedback','timestamp','timewindow']
    save_pickle(df_ml_iu, 'df_ml_iu')

df_30_iu = check_and_import('./df/df_30_iu')
if df_30_iu is None:
    # df_30_iu = df_30.groupby(["uid", "id", "ym"]).size().reset_index(name="feedback")
    df_30_iu = df_30.groupby(["uid", "id", "timewindow"]).size().reset_index(name="feedback")
    save_pickle(df_30_iu, 'df_30_iu')

df_30_ue, df_30_ie = check_and_import('./df/df_30_ue'), check_and_import('./df/df_30_ie')
if df_30_ue is None or df_30_ie is None:
    df_30_i = df_30_iu.groupby(["id","timewindow"]).size().reset_index(name="unum")
    df_30_i = calculate_ir(df_30_i)
    save_pickle(df_30_i, 'df_30_i')
    df_30_userinfo = import_30_userinfo('./data/30_users.txt')
    save_pickle(df_30_userinfo, 'df_30_userinfo')
    df_30_ue = calculate_ue(df_30_i, df_30_iu)
    save_pickle(df_30_ue, 'df_30_ue')
    df_30_ie = calculate_ie(df_30_iu, df_30_ue)
    save_pickle(df_30_ie, 'df_30_ie')


df_ml_ue, df_ml_ie = check_and_import('./df/df_ml_ue'), check_and_import('./df/df_ml_ie')
if df_ml_ue is None or df_ml_ie is None:
    df_ml_i = df_ml_iu.groupby(["id","timewindow"]).size().reset_index(name="unum")
    df_ml_i = calculate_ir(df_ml_i)
    save_pickle(df_ml_i, 'df_ml_i')
    df_ml_ue = calculate_ue(df_ml_i, df_ml_iu)
    save_pickle(df_ml_ue, 'df_ml_ue')
    df_ml_ie = calculate_ie(df_ml_iu, df_ml_ue)
    save_pickle(df_ml_ie, 'df_ml_ie')

if args.embedding_split:
    df_30_ue = mark_n(df_30_ue, 'ue', args.split_n)
    split_and_save_edgelist(df_30_iu, df_30_ue, args.split_n, './graph/30/', '')
    df_ml_ue = mark_n(df_ml_ue, 'ue', args.split_n)
    split_and_save_edgelist(df_ml_iu, df_ml_ue, args.split_n, './graph/ml/', '')

if args.plot_density:
    # Density - IR, UE, IE
    plot_ir_density(df_30_i, df_ml_i)
    plot_ue_density(df_30_ue, df_ml_ue)
    plot_ie_density(df_30_ie, df_ml_ie)
    # UE by age
    plot_ue_age(df_30_userinfo, df_30_ue)

if args.classification_30 or args.regression_30:
    df_30_iteminfo = df_30_iu.groupby(["uid","id"]).size().reset_index(name="size").groupby(["id"]).size().reset_index(name="usernum")
    df_30_ie = pd.merge(df_30_ie, df_30_iteminfo[df_30_iteminfo.usernum>4][['id']], on=["id"], how="inner")
    df_30_iu = pd.merge(df_30_iu, df_30_iteminfo[df_30_iteminfo.usernum>4][['id']], on=["id"], how="inner")
    df_30_userinfo = df_30_iu.groupby(["uid","id"]).size().reset_index(name="size").groupby(["uid"]).size().reset_index(name="itemnum")
    df_30_ue = pd.merge(df_30_ue, df_30_userinfo[df_30_userinfo.itemnum>1000][['uid']], on=["uid"], how="inner")
    df_30_iu = pd.merge(df_30_iu, df_30_userinfo[df_30_userinfo.itemnum>1000][['uid']], on=["uid"], how="inner")
    df_30_iu_uvec = calculate_iu_uvec(df_30_iu, df_30_ie, df_30_ue)
    save_pickle(df_30_iu_uvec, 'df_30_iu_uvec')

    # df_30_iu_uvec = check_and_import('./df/df_30_iu_uvec')
    # if df_30_iu_uvec is None:
    #     df_30_iu_uvec = calculate_iu_uvec(df_30_iu, df_30_ie, df_30_ue)
    #     save_pickle(df_30_iu_uvec, 'df_30_iu_uvec')

    import gensim
    song2vec = gensim.models.Word2Vec.load('./df/song2vec_model_5')
    
    # print(len(df_30_iu_uvec))
    X, y = make_data(df_30_iu_uvec, item_vec=song2vec)
    df_30_data = list(zip(X, y))
    X_sample, y_sample = zip(*random.sample(df_30_data, 1000000))
    X_sample2, y_sample2 = zip(*random.sample(df_30_data, 1000))
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size = 0.3, random_state = 0)

    # df_30_userinfo2 = df_30_iu.groupby(["uid","id"]).size().reset_index(name="size").groupby(["uid"]).size().reset_index(name="itemnum")



if args.classification or args.regression:
    # Filter out users
    df_ml_userinfo = df_ml_iu.groupby(["uid","id"]).size().reset_index(name="size").groupby(["uid"]).size().reset_index(name="itemnum")
    df_ml_ue = pd.merge(df_ml_ue, df_ml_userinfo[df_ml_userinfo.itemnum>1000][['uid']], on=["uid"], how="inner")

    # Filter out Items
    df_ml_iteminfo = df_ml_iu.groupby(["uid","id"]).size().reset_index(name="size").groupby(["id"]).size().reset_index(name="usernum")
    df_ml_ie = pd.merge(df_ml_ie, df_ml_iteminfo[df_ml_iteminfo.usernum>1000][['id']], on=["id"], how="inner")
    df_ml_iu = pd.merge(df_ml_iu, df_ml_iteminfo[df_ml_iteminfo.usernum>1000][['id']], on=["id"], how="inner")

    # calculating each user's ie distribution
    df_ml_iu_uvec = check_and_import('./df/df_ml_iu_uvec')
    if df_ml_iu_uvec is None:
        df_ml_iu_uvec = calculate_iu_uvec(df_ml_iu, df_ml_ie, df_ml_ue)
        save_pickle(df_ml_iu_uvec, 'df_ml_iu_uvec')

    X, y = make_data(df_ml_iu_uvec)
    df_ml_data = list(zip(X, y))
    X_sample, y_sample = zip(*random.sample(df_ml_data, 1000000))
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size = 0.3, random_state = 0)


if args.classification:
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
    y_train2 = [str(x) for x in y_train]
    y_test2 = [str(x) for x in y_test]
    classifier.fit(X_train, y_train2)
    classifier.score(X_test, y_test2)

if args.regression:
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor.fit(X_train, y_train)
    regressor.score(X_test, y_test)

if args.regression_30:
    from sklearn.ensemble import RandomForestRegressor
    regressor3 = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regressor3.fit(X_train, y_train)
    # print(regressor2.score(X_test, y_test))