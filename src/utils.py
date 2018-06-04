#!/usr/bin/python3
import ast
import os
import pickle
import pandas as pd
import math
import time
import numpy as np
import random

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

def list_to_zero_one(inp_list):
    zo_list = (inp_list - min(inp_list)) / (max(inp_list)- min(inp_list))
    return zo_list

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

def calculate_ir_from_iu(df_iu):
    df_i = df_iu.groupby(["id","timewindow"]).size().reset_index(name="unum")
    ir = [-math.log(x) for x in df_i['unum']]
    df_i['ir'] = [-math.log(x) for x in df_i['unum']]
    df_i = df_i.groupby(["id"], as_index=False).mean()
    df_i = df_i[["id","ir"]]
    df_i['ir'] = list_to_z_score(df_i.ir)
    # df_i['ir'] = list_to_zero_one(df_i.ir)
    return df_i

def calculate_ue_from_iu(df_iu):
    df_i = df_iu.groupby(["id","timewindow"]).size().reset_index(name="unum")
    df_i = calculate_ir(df_i)
    df_iu = pd.merge(df_iu, df_i[['id','timewindow','ir']], on=["id",'timewindow'], how="inner")
    df_u = df_iu.groupby(['uid'])['feedback'].sum().reset_index(name="feedback_sum")
    df_iu = pd.merge(df_iu, df_u, on="uid",how="inner")
    df_iu['temp'] = df_iu['feedback']*df_iu['ir']
    df_ue = df_iu.groupby(['uid','feedback_sum'])['temp'].sum().reset_index(name="weighted_sum")
    df_ue['ue'] = df_ue['weighted_sum']/df_ue['feedback_sum']
    df_ue = df_ue[['uid','ue']]
    df_ue['ue'] = list_to_z_score(df_ue.ue)
    # df_ue['ue'] = list_to_zero_one(df_ue.ue)
    return df_ue

def calculate_ie_from_iu(df_iu):
    df_ue = calculate_ue_from_iu(df_iu)
    df_i = df_iu.groupby(['id'])['feedback'].sum().reset_index(name="feedback_sum")
    df_iu = pd.merge(df_iu, df_i[['id','feedback_sum']], on=["id"], how="inner")
    df_iu = pd.merge(df_iu, df_ue, on="uid", how="inner")
    df_iu['temp'] = df_iu['feedback']*df_iu['ue']
    df_ie = df_iu.groupby(['id','feedback_sum'])['temp'].sum().reset_index(name="weighted_sum")
    df_ie['ie'] = df_ie['weighted_sum']/df_ie['feedback_sum']
    df_ie = df_ie[['id','ie']]
    df_ie['ie'] = list_to_z_score(df_ie.ie)
    # df_ie['ie'] = list_to_zero_one(df_ie.ie)
    return df_ie

def calculate_ire_from_iu(df_iu):
    df_ir = calculate_ir_from_iu(df_iu)
    df_ie = calculate_ie_from_iu(df_iu)
    df_ie = pd.merge(df_ie, df_ir, on=["id"], how="inner")
    df_ie['ire'] = df_ie['ie'] * df_ie['ir']
    # df_ie['ie'] = list_to_z_score(df_ie.ie)
    df_ie['ire'] = list_to_zero_one(df_ie.ire)
    df_ie = df_ie[['id','ire']]
    return df_ie

def calculate_ier_from_iu(df_iu):
    df_ir = calculate_ir_from_iu(df_iu)
    df_ie = calculate_ie_from_iu(df_iu)
    df_ie = pd.merge(df_ie, df_ir, on=["id"], how="inner")
    df_ie['ier'] = df_ie['ie'] / df_ie['ir']
    inf_index = [i for i, x in enumerate(df_ie.ier) if np.isinf(x)]
    for idx in inf_index:
        df_ie.at[idx, 'ier'] = 0
    # df_ie['ie'] = list_to_z_score(df_ie.ie)
    df_ie['ier'] = list_to_zero_one(df_ie.ier)
    df_ie = df_ie[['id','ier']]
    return df_ie

def check_and_import(path):
    if os.path.isfile(path):
        with open(path,'rb') as file:
            df = pickle.load(file)
        return df
    else:
        return None

def septile_for_plot(df_iu,seg_point, prefix):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_iu = df_iu.sort_values('timewindow', ascending=0)
    if prefix=='30':
        df_iu1 = df_iu[df_iu.timewindow <= seg_point]
        df_iu2 = df_iu[df_iu.timewindow > seg_point]
    elif prefix=='ml':
        df_iu1 = df_iu[(df_iu.timewindow >201100) & (df_iu.timewindow<201299)]
        df_iu2 = df_iu[(df_iu.timewindow >201300) & (df_iu.timewindow<201499)]
    
    df_ir1 = calculate_ir_from_iu(df_iu1)
    df_ir1 = df_ir1.rename(columns={'ir':'ir1'})
    df_ir2 = calculate_ir_from_iu(df_iu2)
    df_ir2 = df_ir2.rename(columns={'ir':'ir2'})
    df_ue1 = calculate_ue_from_iu(df_iu1)
    df_ue1 = df_ue1.rename(columns={'ue':'ue1'})
    df_ue2 = calculate_ue_from_iu(df_iu2)
    df_ue2 = df_ue2.rename(columns={'ue':'ue2'})
    df_ie1 = calculate_ie_from_iu(df_iu1)
    df_ie1 = df_ie1.rename(columns={'ie':'ie1'})
    df_ie2 = calculate_ie_from_iu(df_iu2)
    df_ie2 = df_ie2.rename(columns={'ie':'ie2'})
    df_i1 = df_iu1.groupby(['id'])['feedback'].size().reset_index(name="unum")
    df_i1 = df_i1[df_i1.unum>4]
    df_i2 = df_iu2.groupby(['id'])['feedback'].size().reset_index(name="unum")
    df_i2 = df_i2[df_i2.unum>4]
    df_u1 = df_iu1.groupby(['uid'])['feedback'].size().reset_index(name="inum")
    df_u1 = df_u1[df_u1.inum>4]
    df_u2 = df_iu2.groupby(['uid'])['feedback'].size().reset_index(name="inum")
    df_u2 = df_u2[df_u2.inum>4]
    df_ir_both = pd.merge(df_ir1, df_ir2, on=['id'], how='inner')
    df_ir_both = pd.merge(df_ir_both, df_i1, on=['id'], how='inner')
    df_ir_both = df_ir_both.drop(columns=['unum'])
    df_ir_both = pd.merge(df_ir_both, df_i2, on=['id'], how='inner')
    df_ir_both = df_ir_both.drop(columns=['unum'])
    df_ir_both = mark_septile(df_ir_both, 'ir1')
    df_ir_both = mark_septile(df_ir_both, 'ir2')
    df_ir_both_cnt = df_ir_both.groupby(['ir1_7','ir2_7'])['id'].size().reset_index(name='count')
    df_ir_both_cnt_sum = df_ir_both_cnt.groupby(['ir1_7'])['count'].sum().reset_index(name="count_sum")
    df_ir_both_cnt = pd.merge(df_ir_both_cnt, df_ir_both_cnt_sum, how='inner')
    df_ir_both_cnt['ratio'] = df_ir_both_cnt['count'] *1.0 / df_ir_both_cnt['count_sum']
    df_ie_both = pd.merge(df_ie1, df_ie2, on=['id'], how='inner')
    df_ie_both = pd.merge(df_ie_both, df_i1, on=['id'], how='inner')
    df_ie_both = df_ie_both.drop(columns=['unum'])
    df_ie_both = pd.merge(df_ie_both, df_i2, on=['id'], how='inner')
    df_ie_both = df_ie_both.drop(columns=['unum'])
    df_ie_both = mark_septile(df_ie_both, 'ie1')
    df_ie_both = mark_septile(df_ie_both, 'ie2')
    df_ie_both_cnt = df_ie_both.groupby(['ie1_7','ie2_7'])['id'].size().reset_index(name='count')
    df_ie_both_cnt_sum = df_ie_both_cnt.groupby(['ie1_7'])['count'].sum().reset_index(name="count_sum")
    df_ie_both_cnt = pd.merge(df_ie_both_cnt, df_ie_both_cnt_sum, how='inner')
    df_ie_both_cnt['ratio'] = df_ie_both_cnt['count'] *1.0 / df_ie_both_cnt['count_sum']
    df_ue_both = pd.merge(df_ue1, df_ue2, on=['uid'], how='inner')
    df_ue_both = pd.merge(df_ue_both, df_u1, on=['uid'], how='inner')
    df_ue_both = df_ue_both.drop(columns=['inum'])
    df_ue_both = pd.merge(df_ue_both, df_u2, on=['uid'], how='inner')
    df_ue_both = df_ue_both.drop(columns=['inum'])
    df_ue_both = mark_septile(df_ue_both, 'ue1')
    df_ue_both = mark_septile(df_ue_both, 'ue2')
    df_ue_both_cnt = df_ue_both.groupby(['ue1_7','ue2_7'])['uid'].size().reset_index(name='count')
    df_ue_both_cnt_sum = df_ue_both_cnt.groupby(['ue1_7'])['count'].sum().reset_index(name="count_sum")
    df_ue_both_cnt = pd.merge(df_ue_both_cnt, df_ue_both_cnt_sum, how='inner')
    df_ue_both_cnt['ratio'] = df_ue_both_cnt['count'] *1.0 / df_ue_both_cnt['count_sum']
    cmap = sns.cubehelix_palette(8)
    plt.rc('axes', labelsize=12)
    ax = sns.heatmap(df_ir_both_cnt.pivot_table(index='ir2_7', columns='ir1_7', values='ratio'), annot=True, fmt=".2f", cmap=cmap, cbar=False)
    ax.invert_yaxis()
    x0,x1=ax.get_xlim()
    y0,y1=ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    if prefix == '30':
        plt.xlabel('Item Rarity Septile (first 6 months)')
        plt.ylabel('Item Rarity Septile (second 6 months)')
    elif prefix =='ml':
        plt.xlabel('Item Rarity Septile (11\'-12\')')
        plt.ylabel('Item Rarity Septile (13\'-14\')')

    plt.savefig(prefix+'_ir_septile.png')
    plt.clf()
    ax = sns.heatmap(df_ie_both_cnt.pivot_table(index='ie2_7', columns='ie1_7', values='ratio'), annot=True, fmt=".2f", cmap=cmap, cbar=False)
    ax.invert_yaxis()
    ax.xaxis
    x0,x1=ax.get_xlim()
    y0,y1=ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    if prefix == '30':
        plt.xlabel('Item Eccentricity Septile (first 6 months)')
        plt.ylabel('Item Eccentricity Septile (second 6 months)')
    elif prefix =='ml':
        plt.xlabel('Item Eccentricity Septile (11\'-12\')')
        plt.ylabel('Item Eccentricity Septile (13\'-14\')')

    plt.tight_layout()
    plt.savefig(prefix+'_ie_septile.png')
    plt.clf()
    ax = sns.heatmap(df_ue_both_cnt.pivot_table(index='ue2_7', columns='ue1_7', values='ratio'), annot=True, fmt=".2f", cmap=cmap, cbar=False)
    ax.invert_yaxis()
    if prefix == '30':
        plt.xlabel('User Eccentricity Septile (first 6 months)')
        plt.ylabel('User Eccentricity Septile (second 6 months)')
    elif prefix =='ml':
        plt.xlabel('User Eccentricity Septile (11\'-12\')')
        plt.ylabel('User Eccentricity Septile (13\'-14\')')

    plt.savefig(prefix+'_ue_septile.png')
    plt.clf()
    return df_ir_both_cnt, df_ie_both_cnt, df_ue_both_cnt

def mark_septile(df, col_name):
    df = df.sort_values(col_name, ascending=1)
    new_col = col_name+'_7'
    repeat = int(math.floor(len(df)/7))
    array = [i for i in range(1,8) for _ in range(repeat)]
    array = array + [7 for i in range(len(df)-len(array))]
    df[new_col] = array
    return df

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
    temp_dist = np.nan_to_num(df_udist['dist'])
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


def random_sample_users(df, ratio):
    '''
    randomly sample users of input ratio to reduce size of the df
    '''
    userlist = [x[0] for x in list(df.groupby("uid")['uid'])]
    sample_userlist = [userlist[i] for i in random.sample(xrange(len(userlist)), int(ratio*len(userlist))) ]
    df = df[df.uid.isin(sample_userlist)]
    return df

def emb_file_to_user_dict(emb_path):
    with open(emb_path,'r') as file:
        emb = file.read().splitlines()
    emb_meta = emb[0]
    emb = emb[1:]
    user_emb = [x.split() for x in emb if not x.startswith('9999999')]
    user_dict = {}
    for user in user_emb:
        user_dict[user[0]] = [float(x) for x in user[1:]]
    return user_dict

def find_similar_users(emb_path, ratio):
    user_dict = emb_file_to_user_dict(emb_path)
    userlist = user_dict.keys()
    target_user = random.choice(userlist)
    target_user_vec = user_dict[target_user]
    cos_sim = []
    from scipy import spatial
    for user in userlist:
        temp_cos = 1 - spatial.distance.cosine(target_user_vec, user_dict[user])
        cos_sim.append((user, temp_cos))
    from operator import itemgetter
    cos_sim.sort(key=itemgetter(1), reverse=True)
    user_num = int(ratio*float(len(userlist)))
    similar_users = [x[0] for x in cos_sim[:user_num]]
    return similar_users
    
def save_edgelist_of_users(df_iu, user_list, edge_list_path='./graph/', file_prefix=''):
    if not os.path.isdir(edge_list_path):
        os.mkdir(edge_list_path)
    df_iu_temp = df_iu[df_iu.uid.isin(user_list)]
    df_iu_temp = df_iu_temp[['uid','id','feedback']]
    df_iu_temp = df_iu_temp.values.tolist()
    for data in df_iu_temp:
        # data[0] = data[0]
        data[1] = '9999999'+data[1]
        data[2] = str(data[2])
    
    with open(edge_list_path+file_prefix+'.edgelist','w') as file:
        file.write("\n".join([' '.join(x) for x in df_iu_temp]))


    

