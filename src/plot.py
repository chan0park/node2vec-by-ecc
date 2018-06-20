import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def plot_ue_30_heatmap(septile_30_schedl):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_30_schedl, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('User Eccentricity Septile (first 6 months)', fontsize=16)
    plt.ylabel('User Eccentricity Septile (second 6 months)', fontsize=16)
    plt.show()

def plot_ue_ml_heatmap(septile_ml_schedl):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_ml_schedl, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('User Eccentricity Septile (11\'- 12\')', fontsize=16)
    plt.ylabel('User Eccentricity Septile (13\'-14\')', fontsize=16)
    plt.show()

def plot_ie_30_heatmap(septile_30_ie):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_30_ie, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('Item Eccentricity Septile (first 6 months)', fontsize=16)
    plt.ylabel('Item Eccentricity Septile (second 6 months)', fontsize=16)
    plt.show()

def plot_ie_ml_heatmap(septile_ml_ie):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_ml_ie, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('Item Eccentricity Septile (11\'- 12\')', fontsize=16)
    plt.ylabel('Item Eccentricity Septile (13\'-14\')', fontsize=16)
    plt.show()

def plot_ir_30_heatmap(septile_30_tir):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_30_tir, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('Item Rarity Septile (first 6 months)', fontsize=16)
    plt.ylabel('Item Rarity Septile (second 6 months)', fontsize=16)
    plt.show()

def plot_ir_ml_heatmap(septile_ml_tir):
    plt.figure(figsize=(6,6))
    ax = sns.heatmap(septile_ml_tir, annot=True, cmap=sns.cubehelix_palette(150), cbar=False)
    ax.invert_yaxis()
    plt.xlabel('Item Rarity Septile (11\'- 12\')', fontsize=16)
    plt.ylabel('Item Rarity Septile (13\'-14\')', fontsize=16)
    plt.show()

# Density - IR
def plot_ir_density(df_30_i, df_ml_i):
    df_30_tir = df_30_i.groupby(['id'])['zir'].mean().reset_index(name="tzir")
    df_ml_tir = df_ml_i.groupby(['id'])['zir'].mean().reset_index(name="tzir")
    plt.figure(figsize=(8,6))
    ax1 = sns.kdeplot(df_30_tir['tzir'], shade=args.shade, label="30Music")
    ax1.set(xlim=(-1,1.5))
    ax2 = sns.kdeplot(df_ml_tir['tzir'], shade=args.shade, label="MovieLens")
    ax2.set(xlim=(-1,1.5))
    plt.xlabel('Standardized Item Rarity', fontsize=16)
    plt.ylabel('Density Function', fontsize=16)
    plt.legend(fontsize="large")
    plt.show()

# Density - UE
def plot_ue_density(df_30_ue, df_ml_ue):
    plt.figure(figsize=(8,6))
    ax1 = sns.kdeplot(df_30_ue['zue'], shade=args.shade, label="30Music", bw=0.25)
    ax1.set(xlim=(-4,4))
    ax2 = sns.kdeplot(df_ml_ue['zue'], shade=args.shade, label="MovieLens", bw=0.25)
    ax2.set(xlim=(-4,4))
    plt.xlabel('User Eccentricity', fontsize=16)
    plt.ylabel('Density Function', fontsize=16)
    plt.legend(fontsize="large")
    plt.show()

# # Density - IE
def plot_ie_density(df_30_ie, df_ml_ie):
    plt.figure(figsize=(8,6))
    ax1 = sns.kdeplot(df_30_ie['zie'], shade=args.shade, label="30Music", bw=0.15)
    ax1.set(xlim=(-4,4))
    ax2 = sns.kdeplot(df_ml_ie['zie'], shade=args.shade, label="MovieLens", bw=0.15)
    ax2.set(xlim=(-4,4))
    plt.xlabel('Item Eccentricity', fontsize=16)
    plt.ylabel('Density Function', fontsize=16)
    plt.legend(fontsize="large")
    plt.show()

# # UE by age
def plot_ue_age(df_30_userinfo, df_30_ue):
    df_30_userinfo_ue = pd.merge(df_30_userinfo, df_30_ue[['uid','zue']], on="uid", how="inner")
    df_30_userinfo_ue['age'] = pd.to_numeric(df_30_userinfo_ue['age'])

    ax0 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=10) & (df_30_userinfo_ue.age<20)]['zue'], shade=args.shade, label="10s")
    ax1 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=20) & (df_30_userinfo_ue.age<30)]['zue'], shade=args.shade, label="20s")
    ax2 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=30) & (df_30_userinfo_ue.age<40)]['zue'], shade=args.shade, label="30s")
    ax3 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=40) & (df_30_userinfo_ue.age<50)]['zue'], shade=args.shade, label="40s")
    ax4 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=50) & (df_30_userinfo_ue.age<60)]['zue'], shade=args.shade, label="50s")
    ax5 = sns.kdeplot(df_30_userinfo_ue[(df_30_userinfo_ue.age>=60) & (df_30_userinfo_ue.age<70)]['zue'], shade=args.shade, label="60s")
    ax0.set(xlim=(-3,3))
    ax0.set(ylim=(0,0.55))
    plt.xlabel('User Eccentricity', fontsize=16)
    plt.ylabel('Density Function', fontsize=16)
    plt.legend(fontsize="large")
    plt.show()

def plot_long_tail(df_iu):
    df_i = df_iu.groupby(["id"]).size().reset_index(name="unum")
    unum_list = list(df_i.unum)
    unum_list.sort(reverse=True)
    y_pos = []
    x_pos = list(range(0,100,5))
    step = int(np.ceil(len(unum_list)/20.0))
    for i in range(0, len(unum_list), step):
        chunk = unum_list[i:i+step]
        y_pos.append(float(sum(chunk))/sum(unum_list))
    
    plt.figure(figsize=(8,6))
    plt.bar(x_pos[:15], y_pos[:15], width=5, align='edge', color='brown')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.ylabel('Consumed user number / Total user number', fontsize=15, labelpad=15)
    plt.xlabel('Items Ranked by Popularity (%)', fontsize=15, labelpad=10)
    plt.tight_layout()
    plt.show()

plot_long_tail(df_ml_iu)
