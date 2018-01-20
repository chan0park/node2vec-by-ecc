#!/usr/bin/python3
import ast
import os
import pickle
import pandas as pd
import math
import time
import numpy

def import_ml(ml_path="./ml/ratings.csv", headercol=None):
    with open(ml_path, 'r') as file:
        ml_data = file.readlines()

    ml_data = [x.strip('\n').split(',') for x in ml_data]
    if not headercol:
        df = pd.DataFrame(ml_data[1:], columns=ml_data[0])
    if headercol:
        df = pd.DataFrame(ml_data, columns=headercol)
    return df


def import_30(path="./data/30_eventinfo.csv", headercol=['eid','timestamp','playtime','uid','tid']):
    with open(path,'r') as file:
        data = file.readlines()

    data = [x.replace("\"","").strip('\n').split(',') for x in data]
    if not headercol:
        df = pd.DataFrame(data[1:], columns=data[0])
    if headercol:
        df = pd.DataFrame(data, columns=headercol)
    return df

def preprocessing(df_i, th=1):
    df = df_i[df_i.unum > 1]
    return df

def list_to_z_score(inp_list):
    z_list = inp_list - (inp_list.mean())/inp_list.std(ddof=0)
    return z_list

def df_to_z_score(df, col, result):
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df

def calculate_ir(df_i, df_iu):
    ir = [-math.log(x) for x in df_i['unum']]
    ir = list_to_z_score(ir)
    df_i['ir'] = ir
    df_iu.join(df_i)
    e = df_iu.groupby(["uid"])
    
    
def check_and_import(path):
    if os.path.isfile(path):
        with open(path,'rb') as file:
            df = pickle.load(file)
        return df
    else:
        return None

def septile_for_plot():
    return 

def mark_septile():
    return

def mark_septile_ie():
    return

def mark_septile_tir():
    return

def mark_100(df, col_name):
    df = df.sort(col_name, ascending=1)
    new_col = col_name+'_100'
    repeat = math.floor(len(df)/100)
    array = [i for i in range(1,101) for _ in range(repeat)]
    array = array + [100 for i in range(len(df)-len(array))]
    df[new_col] = array
    return df


def mark_timewindow(df):
    temp = df['timestamp']
    ym = [int(str(time.localtime(int(x)).tm_year)+"%02d"%time.localtime(int(x)).tm_mon) for x in temp]
    df['ym'] = ym
    return df

def save_pickle(obj, name):
    with open('./df/'+name, 'wb') as file:
        pickle.dump(obj, file)


def calculate_uie_dist(df_iu, df_ie, df_ue):
    df_ie = mark_100(df_ie,'zie')
    df_iue = pd.merge(df_iu, df_ie[['id','zie_100']], on="id", how="inner")
    df_udist_temp_weight = df_iue.groupby(["uid", "zie_100"])['feedback'].sum().reset_index(name="dist")
    df_udist_temp = df_iue.groupby(["uid", "zie_100"])['feedback'].size().reset_index(name="dist")
    userlist = list(df_ue['uid'])
    df_udist = [[i,j] for i in userlist for j in range(1,101)]
    df_udist = pd.DataFrame(df_udist, columns=['uid','zie_100'])
    df_udist = pd.merge(df_udist, df_udist_temp, on=["uid","zie_100"], how="left")
    return df_udist

def calculate_uvec(df_ue, df_udist):  
    userlist = list(df_ue['uid'])
    temp_dist = numpy.nan_to_num(df_udist['dist'])
    user_vec = []
    for i in range(int(len(temp_dist)/100)):
        tmp = temp_dist[i*100:(i+1)*100]
        tmp = [x/sum(tmp) for x in tmp]
        #make data  uvec+id(101) /feedback
        user_vec.append([userlist[i], tmp])
    df_uvec = pd.DataFrame(user_vec, columns=['uid','uvec'])
    return df_uvec

def make_data(df_iu_uvec, item_vec=None):
    temp_uvec = list(df_iu_uvec['uvec'])
    temp_id = list(df_iu_uvec['id'])
    if item_vec is None:
        temp_id = [int(x) for x in temp_id]
    data_y = list(df_iu_uvec['feedback'])
    data_x = []
    for i in range(len(temp_uvec)):
        if item_vec is None:
            data_x.append(temp_uvec[i]+[temp_id[i]])
        else:
            try:
                data_x.append(temp_uvec[i]+list(item_vec[temp_id[i]]))
            except:
                data_x.append(None)
        # try:
        #     data_x.append(temp_uvec[i]+[temp_id[i]])
        # except:
        #     print(temp_uvec[i])
        #     print(temp_id[i])
        #     raise
    if item_vec is not None:
        data_x, data_y = zip(*((x, y) for x, y in zip(data_x, data_y) if x is not None))
    return data_x, data_y


def calculate_iu_uvec(df_iu, df_ie, df_ue):
    df_udist = calculate_uie_dist(df_iu, df_ie, df_ue)
    df_uvec = calculate_uvec(df_ue, df_udist)
    df_iu_uvec = pd.merge(df_iu, df_uvec, on="uid", how="inner")
    return df_iu_uvec