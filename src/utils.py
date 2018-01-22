#!/usr/bin/python3
import ast
import os
import pickle
import pandas as pd
import math
import time
import numpy

def import_ml(ml_path="./data/ratings.csv", headercol=None):
    with open(ml_path, 'r') as file:
        ml_data = file.read().splitlines()
    ml_data = [x.strip('\n').split(',') for x in ml_data]
    if not headercol:
        df = pd.DataFrame(ml_data[1:], columns=ml_data[0])
    if headercol:
        df = pd.DataFrame(ml_data, columns=headercol)
    return df


def import_30(path="./data/30_eventinfo.csv", headercol=['eid','timestamp','playtime','uid','id']):
    with open(path,'r') as file:
        data = file.readlines()
    data = [x.replace("\"","").strip('\n').split(',') for x in data]
    if not headercol:
        df = pd.DataFrame(data[1:], columns=data[0])
    if headercol:
        df = pd.DataFrame(data, columns=headercol)
    return df

def import_30_userinfo(path='./data/30_users.txt'):
    with open(path,'r') as file:
        data = file.read().splitlines()
    data = [x.split('\t') for x in data]
    target_cols = ['country','age','gender']
    import ast
    user_data = []
    for row in data:
        temp_data = []
        temp_dict = ast.literal_eval(row[3])
        temp_data.append(row[1])
        for col in target_cols:
            temp_data.append(temp_dict[col])
        user_data.append(temp_data)
    user_data = pd.DataFrame(user_data, columns=['uid']+target_cols) 
    return user_data

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

def calculate_ir(df_i):
    ir = [-math.log(x) for x in df_i['unum']]
    df_i['ir'] = ir
    # ir = list_to_z_score(ir)
    df_i['ir'] = list_to_z_score(df_i.ir)
    return df_i

def calculate_ue(df_i, df_iu):
    df_iu = pd.merge(df_iu, df_i[['id','timewindow','ir']], on=["id",'timewindow'], how="inner")
    df_u = df_iu.groupby(['uid'])['feedback'].sum().reset_index(name="feedback_sum")
    df_iu = pd.merge(df_iu, df_u, on="uid",how="inner")
    df_iu['temp'] = df_iu['feedback']*df_iu['ir']
    df_ue = df_iu.groupby(['uid','feedback_sum'])['temp'].sum().reset_index(name="weighted_sum")
    df_ue['ue'] = df_ue['weighted_sum']/df_ue['feedback_sum']
    df_ue = df_ue[['uid','ue']]
    df_ue['ue'] = list_to_z_score(df_ue.ue)
    return df_ue

def calculate_ie(df_iu, df_ue):
    df_i = df_iu.groupby(['id'])['feedback'].sum().reset_index(name="feedback_sum")
    df_iu = pd.merge(df_iu, df_i[['id','feedback_sum']], on=["id"], how="inner")
    df_iu = pd.merge(df_iu, df_ue, on="uid", how="inner")
    df_iu['temp'] = df_iu['feedback']*df_iu['ue']
    df_ie = df_iu.groupby(['id','feedback_sum'])['temp'].sum().reset_index(name="weighted_sum")
    df_ie['ie'] = df_ie['weighted_sum']/df_ie['feedback_sum']
    df_ie = df_ie[['id','ie']]
    df_ie['ie'] = list_to_z_score(df_ie.ie)
    return df_ie




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

def mark_10(df, col_name):
    df = df.sort_values(col_name, ascending=1)
    new_col = col_name+'_10'
    repeat = int(math.floor(len(df)/10))
    array = [i for i in range(1,11) for _ in range(repeat)]
    array = array + [10 for i in range(len(df)-len(array))]
    df[new_col] = array
    return df

def mark_n(df, col_name, n):
    df = df.sort_values(col_name, ascending=1)
    new_col = col_name+'_'+str(n)
    repeat = int(math.floor(len(df)/n))
    array = [i for i in range(1,n+1) for _ in range(repeat)]
    array = array + [n for i in range(len(df)-len(array))]
    df[new_col] = array
    return df


def mark_timewindow(df):
    temp = df['timestamp']
    ym = [int(str(time.localtime(int(x)).tm_year)+"%02d"%time.localtime(int(x)).tm_mon) for x in temp]
    df['timewindow'] = ym
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


def split_and_save_edgelist(df_iu, df_ue, split_n, edge_list_path='./graph/', file_prefix=''):
    import os
    if not os.path.isdir(edge_list_path):
        os.mkdir(edge_list_path)
    for i in range(1, split_n+1):
        user_list = list(df_ue[df_ue["ue_"+str(split_n)] == i]['uid'])
        df_iu_temp = df_iu[df_iu.uid.isin(user_list)]
        df_iu_temp = df_iu_temp[['uid','id','feedback']]
        df_iu_temp = df_iu_temp.values.tolist()
        for data in df_iu_temp:
            # data[0] = data[0]
            data[1] = '9999999'+data[1]
            data[2] = str(data[2])
        
        with open(edge_list_path+file_prefix+'ue_{}.edgelist'.format(i),'w') as file:
            file.write("\n".join([' '.join(x) for x in df_iu_temp]))

def save_edgelist(df_iu, edge_list_path='./graph/', file_prefix=''):
    df_iu = df_iu[['uid','id','feedback']]
    df_iu = df_iu.values.tolist()
    for data in df_iu:
        data[1] = '9999999'+data[1]
        data[2] = str(data[2])
    with open(edge_list_path+file_prefix+'ue.edgelist','w') as file:
        file.write("\n".join([' '.join(x) for x in df_iu]))

    

