from utils import *
from plot import *
import time

df_30 = check_and_import('./df/df_30')
if df_30 is None:
    df_30 = import_30("./30/eventinfo.csv")
    df_30 = mark_timewindow(df_30)
    save_pickle(df_30, 'df_30')
df_ml = check_and_import('./df/df_ml')
if df_ml is None:
    df_ml = import_ml()

df_ml = mark_timewindow(df_ml)
df_30 = mark_timewindow(df_30) 

df_ml_1 = df_ml[(df_ml.timewindow>=201100) & (df_ml.timewindow<=201212)]
df_ml_2 = df_ml[(df_ml.timewindow>=201300) & (df_ml.timewindow<=201412)]
df_30_1 = df_30[(df_30.timewindow>=201402) & (df_30.timewindow<=201407)]
df_30_2 = df_30[(df_30.timewindow>=201408) & (df_30.timewindow<=201501)]

df_ml_ue_1, df_ml_ie_1, df_ml_ir_1 = calculate_all(df_ml_1, "movie")
df_ml_ue_2, df_ml_ie_2, df_ml_ir_2 = calculate_all(df_ml_2, "movie")
df_30_ue_1, df_30_ie_1, df_30_ir_1 = calculate_all(df_30_1, "music")
df_30_ue_2, df_30_ie_2, df_30_ir_2 = calculate_all(df_30_2, "music")

df_30_ue_schedl_1 = calculate_all_schedl(df_30_1, "music")
df_30_ue_schedl_2 = calculate_all_schedl(df_30_2, "music")
df_ml_ue_schedl_1 = calculate_all_schedl(df_ml_1, "movie")
df_ml_ue_schedl_2 = calculate_all_schedl(df_ml_2, "movie")

# df_30_iu_1 = df_30_1.groupby(["uid", "id", "timewindow"]).size().reset_index(name="feedback")
# df_30_iu_2 = df_30_2.groupby(["uid", "id", "timewindow"]).size().reset_index(name="feedback")

# df_ml_ir_1 = calculate_ir(df_ml_1)
# df_ml_ir_2 = calculate_ir(df_ml_2)
# df_30_ir_1 = calculate_ir(df_30_iu_1)
# df_30_ir_2 = calculate_ir(df_30_iu_2)

df_30_tir_1 = df_30_ir_1.groupby(['id'])['zir'].mean().reset_index(name="tzir")
df_30_tir_2 = df_30_ir_2.groupby(['id'])['zir'].mean().reset_index(name="tzir")
df_ml_tir_1 = df_ml_ir_1.groupby(['id'])['zir'].mean().reset_index(name="tzir")
df_ml_tir_2 = df_ml_ir_2.groupby(['id'])['zir'].mean().reset_index(name="tzir")


df_ml_ue_1 = df_ml_ue_1[df_ml_ue_1.tinum>4]
df_ml_ue_2 = df_ml_ue_2[df_ml_ue_2.tinum>4]
df_30_ue_1 = df_30_ue_1[df_30_ue_1.tinum>4]
df_30_ue_2 = df_30_ue_2[df_30_ue_2.tinum>4] 

df_ml_ie_1 = df_ml_ie_1[df_ml_ie_1.tunum>4]
df_ml_ie_2 = df_ml_ie_2[df_ml_ie_2.tunum>4]
df_30_ie_1 = df_30_ie_1[df_30_ie_1.tunum>4]
df_30_ie_2 = df_30_ie_2[df_30_ie_2.tunum>4]

df_ml_ue_1.rename(columns = {'zue':'zue1'}, inplace = True)
df_ml_ue_2.rename(columns = {'zue':'zue2'}, inplace = True)
df_30_ue_1.rename(columns = {'zue':'zue1'}, inplace = True)
df_30_ue_2.rename(columns = {'zue':'zue2'}, inplace = True)
df_ml_ue12 = pd.merge(df_ml_ue_1[['uid','zue1']], df_ml_ue_2[['uid','zue2']], on="uid", how="inner")    
df_30_ue12 = pd.merge(df_30_ue_1[['uid','zue1']], df_30_ue_2[['uid','zue2']], on="uid", how="inner")
df_ml_ue12 = mark_septile(df_ml_ue12)
df_30_ue12 = mark_septile(df_30_ue12)
df_30_ue12_2 = df_30_ue12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_ue12_2 = df_ml_ue12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_ue12_3 = df_ml_ue12_2.groupby("septile1").sum()
df_ml_ue12_3['septile1'] = range(1,8)
df_30_ue12_3 = df_30_ue12_2.groupby("septile1").sum()
df_30_ue12_3['septile1'] = range(1,8)
df_ml_ue12_3.rename(columns = {'freq':'sum'}, inplace = True)
df_30_ue12_3.rename(columns = {'freq':'sum'}, inplace = True)

df_ml_ue12_2 = pd.merge(df_ml_ue12_2, df_ml_ue12_3[['septile1','sum']])
df_30_ue12_2 = pd.merge(df_30_ue12_2, df_30_ue12_3[['septile1','sum']])
df_ml_ue12_2['p'] = df_ml_ue12_2['freq']/df_ml_ue12_2['sum']
df_30_ue12_2['p'] = df_30_ue12_2['freq']/df_30_ue12_2['sum']

septile_ml = df_ml_ue12_2.pivot("septile2","septile1","p")
septile_30 = df_30_ue12_2.pivot("septile2","septile1","p")

septile_ml_schedl, septile_30_schedl = septile_for_plot(df_ml_ue_schedl_1, df_ml_ue_schedl_2, df_30_ue_schedl_1, df_30_ue_schedl_2)

df_ml_ie_1.rename(columns = {'zie':'zie1'}, inplace = True)
df_ml_ie_2.rename(columns = {'zie':'zie2'}, inplace = True)
df_30_ie_1.rename(columns = {'zie':'zie1'}, inplace = True)
df_30_ie_2.rename(columns = {'zie':'zie2'}, inplace = True)
df_ml_ie12 = pd.merge(df_ml_ie_1[['id','zie1']], df_ml_ie_2[['id','zie2']], on="id", how="inner")    
df_30_ie12 = pd.merge(df_30_ie_1[['id','zie1']], df_30_ie_2[['id','zie2']], on="id", how="inner")
df_ml_ie12 = mark_septile_ie(df_ml_ie12)
df_30_ie12 = mark_septile_ie(df_30_ie12)
df_30_ie12_2 = df_30_ie12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_ie12_2 = df_ml_ie12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_ie12_3 = df_ml_ie12_2.groupby("septile1").sum()
df_ml_ie12_3['septile1'] = range(1,8)
df_30_ie12_3 = df_30_ie12_2.groupby("septile1").sum()
df_30_ie12_3['septile1'] = range(1,8)
df_ml_ie12_3.rename(columns = {'freq':'sum'}, inplace = True)
df_30_ie12_3.rename(columns = {'freq':'sum'}, inplace = True)

df_ml_ie12_2 = pd.merge(df_ml_ie12_2, df_ml_ie12_3[['septile1','sum']])
df_30_ie12_2 = pd.merge(df_30_ie12_2, df_30_ie12_3[['septile1','sum']])
df_ml_ie12_2['p'] = df_ml_ie12_2['freq']/df_ml_ie12_2['sum']
df_30_ie12_2['p'] = df_30_ie12_2['freq']/df_30_ie12_2['sum']

septile_ml_ie = df_ml_ie12_2.pivot("septile2","septile1","p")
septile_30_ie = df_30_ie12_2.pivot("septile2","septile1","p")

df_ml_tir_1.rename(columns = {'tzir':'tzir1'}, inplace = True)
df_ml_tir_2.rename(columns = {'trir2':'tzir2'}, inplace = True)
df_30_tir_1.rename(columns = {'tzir':'tzir1'}, inplace = True)
df_30_tir_2.rename(columns = {'tzir':'tzir2'}, inplace = True)
df_ml_tir12 = pd.merge(df_ml_tir_1[['id','tzir1']], df_ml_tir_2[['id','tzir2']], on="id", how="inner")    
df_30_tir12 = pd.merge(df_30_tir_1[['id','tzir1']], df_30_tir_2[['id','tzir2']], on="id", how="inner")
df_ml_tir12 = mark_septile_tir(df_ml_tir12)
df_30_tir12 = mark_septile_tir(df_30_tir12)
df_30_tir12_2 = df_30_tir12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_tir12_2 = df_ml_tir12.groupby(["septile1","septile2"]).size().reset_index(name="freq")
df_ml_tir12_3 = df_ml_tir12_2.groupby("septile1").sum()
df_ml_tir12_3['septile1'] = range(1,7)
df_30_tir12_3 = df_30_tir12_2.groupby("septile1").sum()
df_30_tir12_3['septile1'] = range(1,8)
df_ml_tir12_3.rename(columns = {'freq':'sum'}, inplace = True)
df_30_tir12_3.rename(columns = {'freq':'sum'}, inplace = True)

df_ml_tir12_2 = pd.merge(df_ml_tir12_2, df_ml_tir12_3[['septile1','sum']])
df_30_tir12_2 = pd.merge(df_30_tir12_2, df_30_tir12_3[['septile1','sum']])
df_ml_tir12_2['p'] = df_ml_tir12_2['freq']/df_ml_tir12_2['sum']
df_30_tir12_2['p'] = df_30_tir12_2['freq']/df_30_tir12_2['sum']

septile_ml_tir = df_ml_tir12_2.pivot("septile2","septile1","p")
septile_30_tir = df_30_tir12_2.pivot("septile2","septile1","p")

# UE heatmap
plot_ue_30_heatmap(septile_30_schedl)
plot_ue_ml_heatmap(septile_ml_schedl)

# IE heatmap
plot_ie_30_heatmap(septile_30_ie)
plot_ie_ml_heatmap(septile_ml_ie)

# # IR heatmap
plot_ir_30_heatmap(septile_30_tir)
plot_ir_ml_heatmap(septile_ml_tir)