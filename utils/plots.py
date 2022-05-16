import os
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import seaborn as sns


sns.set('paper')
sns.set_style('whitegrid', {'axes.grid': False})
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'legend.fontsize': 'x-small',
    'legend.handlelength': 1,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.8,
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small'}
plt.rcParams.update(params)

ratio = 0.5

props = ['qed', 'SAS', 'logP']

feats = {
    'atoms': ['C', 'F', 'N', 'O', 'Other'],
    'bonds': ['SINGLE', 'DOUBLE', 'TRIPLE'],
    'rings': ['Tri', 'Quad', 'Pent', 'Hex']
}

#MODEL = 'OURS'
MODEL = 'DEL'


def plot_property(df, name, prop, ax=None):
    new_names = dict([(p, p.upper()) for p in props])
    df.rename(columns=new_names, inplace=True)
    sns.distplot(df[prop.upper()][df.who==name], hist=False, label=name, ax=ax)
    ax = sns.distplot(df[prop.upper()][df.who==MODEL], hist=False, label=MODEL, ax=ax)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    if prop=='LOGP':
        ax.set_xlim(-10,10)
    
    
def plot_property_multi(df, names, prop, ax=None):
    new_names = dict([(p, p.upper()) for p in props])
    df.rename(columns=new_names, inplace=True)
    for name in names:
        sns.distplot(df[prop.upper()][df.who==name], hist=False, label=name, ax=ax, kde_kws={"lw": 0.5})
    #ax = sns.distplot(df[prop.upper()][df.who==MODEL], hist=False, label=MODEL, ax=ax, kde_kws={"lw": 1})
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    ax.set_ylabel('')
    ax.legend()
    if prop=='logP':
        ax.set_xlim(-10,10)


def plot_count(df, name, feat, ax=None):
    s1 = df[feats[feat]][df.who==name].mean(axis=0)
    s2 = df[feats[feat]][df.who==MODEL].mean(axis=0)
    data = pd.DataFrame([s1, s2], index=[name, MODEL])
    ax = data.plot(kind='bar', stacked=True, ax=ax, rot=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.9),
          ncol=len(feats[feat]), framealpha=0, borderpad=1, title=feat.upper())


def plot_count_multi(df, names, feat, ax=None):
    s=[]
    for name in names:
        sn = df[feats[feat]][df.who==name].mean(axis=0)
        s.append(sn)
    data = pd.DataFrame(s, index=names)
    ax = data.plot(kind='bar', stacked=True, ax=ax, rot=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.9),
          ncol=len(feats[feat]), framealpha=0, borderpad=1, title=feat.upper())
    

def plot_counts(df, dataset_name, dirsave='./'):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i, f in enumerate(feats):
        plot_count(df, dataset_name, f, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    fig.savefig(f'counts_{dataset_name}.pdf')
    fig.savefig( os.path.join(dirsave, f'counts_{dataset_name}.pdf' ), bbox_inches='tight' )


def plot_counts_multi(df, dataset_name, dataset_name_long, dirsave='./'):
    names = df.who.unique()
    # put dataset name as first
    new_names = [dataset_name]
    for n in names:
        if n!= dataset_name:
            new_names.append(n)
    
    fig, axs = plt.subplots(1, 3, figsize=(8,4))
    for i, f in enumerate(feats):
        plot_count_multi(df, new_names, f, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    #fig.savefig(f'counts_{ dataset_name_long}.pdf')
    fig.savefig( os.path.join(dirsave, f'counts_{ dataset_name_long}.pdf' ), bbox_inches='tight' )


def plot_props(df, dataset_name, dirsave='./'):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i, p in enumerate(props):
        plot_property(df, dataset_name, p, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    
    fig.savefig( os.path.join(dirsave, f'props_{dataset_name}.pdf' ), bbox_inches='tight' )


def plot_props_multi(df, dataset_name, dataset_name_long, dirsave='./'):
    names = df.who.unique()
    # put dataset name as first
    new_names = [dataset_name]
    for n in names:
        if n!= dataset_name:
            new_names.append(n)
    
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    for i, p in enumerate(props):
        plot_property_multi(df, new_names, p, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
        if p=='logP':
            axs.flat[i].set_xlim(-10,10)
            #axs.flat[i].legend()
    
    fig.savefig( os.path.join(dirsave, f'props_{dataset_name_long}.pdf' ), bbox_inches='tight' )


def plot_props_multi_compare_diff_experiments(df, dataset_name, dataset_name_long, dirsave='./'):
    names = df.who.unique()    
    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    for i, p in enumerate(props):
        plot_property_multi(df, names, p, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    
    fig.savefig( os.path.join(dirsave, f'props_{dataset_name_long}.pdf' ), bbox_inches='tight' )
    

def plot_paper_figures(run_dir, DEL=False):
    #dataset_name = "ZINC" if "ZINC" in run_dir else "PCBA"
    if 'ZINC' in run_dir:
        dataset_name='ZINC'
    if 'PCBA' in run_dir:
        dataset_name='PCBA'
    if 'ZINCMOSES' in run_dir:
        dataset_name='ZINCMOSES'
    if 'CHEMBL' in run_dir:
        dataset_name='CHEMBL'
    print(run_dir)
    if DEL:
        suffix='_del'
    df = pd.read_csv(os.path.join(run_dir, 'results/samples'+suffix+'/aggregated.csv'))
    plot_counts(df, dataset_name, dirsave=os.path.join(run_dir, 'results/samples'+suffix+'/'))
    plot_props(df, dataset_name, dirsave=os.path.join(run_dir, 'results/samples'+suffix+'/'))


def plot_figures(run_dir, DEL=False, filename='aggregated'):
    #dataset_name = "ZINC" if "ZINC" in run_dir else "PCBA"
    if 'ZINC' in run_dir:
        dataset_name='ZINC'
    if 'PCBA' in run_dir:
        dataset_name='PCBA'
    if 'ZINCMOSES' in run_dir:
        dataset_name='ZINCMOSES'
    if 'CHEMBL' in run_dir:
        dataset_name='CHEMBL'
    print(run_dir)
    if DEL:
        suffix='_del'
    df = pd.read_csv(os.path.join(run_dir, 'results/samples'+suffix+'/'+filename+'.csv'))
    plot_counts_multi(df, dataset_name, dataset_name+'_'+filename, dirsave=os.path.join(run_dir, 'results/samples'+suffix+'/'))
    plot_props_multi(df, dataset_name, dataset_name+'_'+filename, dirsave=os.path.join(run_dir, 'results/samples'+suffix+'/'))
    

def plot_loss(run_dir, group='batch'):
    df = pd.read_csv(os.path.join(run_dir, 'results/performance/loss_details.csv'))
    
    # if group=='epoch':
        # df = df.groupby('epoch').mean()
    
    #fig, axs = plt.subplots(5, 1, sharex=True)
    fig, axs = plt.subplots(6, 1)
    fig.set_size_inches(8,10)
    # axs[0].plot(df['CE_loss'], color='blue')
    # axs[0].set_ylabel('CE Loss')
    # axs[1].plot(df['KL_loss'], color='red')
    # axs[1].set_ylabel('KL Loss')
    # axs[2].plot(df['MSE_loss'], color='orange')
    # axs[2].set_ylabel('MSE Loss')
    axs[3].plot(df['loss'], color='purple')
    axs[3].set_ylabel('Loss') #r'CE_Loss + $\beta$ MSE_Loss + $\alpha$ KL_loss'
    # axs[4].plot(df['beta'], color='olive')
    # axs[4].set_ylabel(r'$\beta$')
    # axs[4].set_xlabel(group)
    # axs[4].set_ylim(0,df['beta'].max()+0.05)
    # axs[5].plot(df['alpha'], color='hotpink')
    # axs[5].set_ylabel(r'$\alpha$')
    # axs[5].set_xlabel(group)
    # axs[5].set_ylim(0,df['alpha'].max()+0.05)
    filename=os.path.join(run_dir, 'results/performance/fig_loss_details_'+group+'.pdf')
    fig.savefig(filename,bbox_inches='tight')
    
# def plot_pareto_fronts(run_dir, fronts=[0,1,2], color=None, with_bo=False):
#     pop_final = pd.read_csv(os.path.join(run_dir, 'results/samples_del/new_pop_final.csv'))
#     z = pop_final['qed'].to_numpy()
#     x = pop_final['SAS'].to_numpy()
#     y = pop_final['logP'].to_numpy()
#     rank = pop_final['rank'].to_numpy()         
    
#     if color is None:
#         colors=['red', 'blue', 'cyan', 'olive', 'purple', 'brown', 'lime', 'orchid']
    
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     #for f in fronts[::-1]:
#     for f in fronts:
#         ind = (rank==f)
#         c = colors[f]
#         ax.scatter(x[ind], y[ind], z[ind], color=c, marker='.', s=2, label='DEL Pareto Front '+str(f+1))
    
#     bo_file = os.path.join(run_dir, 'results/bo/samples_qehvi.csv')
#     with_bo = with_bo & os.path.isfile(bo_file)
#     if with_bo:
#         bo_results = pd.read_csv(bo_file)
#         zbo = bo_results['qed'].to_numpy()
#         xbo = bo_results['SAS'].to_numpy()
#         ybo = bo_results['logP'].to_numpy()
#         batch = bo_results['batch'].to_numpy()
        
#         colors_bo = ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke']
#         batchmax= batch.max()
#         #for f in fronts[::-1]:
#         for f in fronts:
#             ind = (batch==(batchmax-f)) & (zbo!=10)
#             ax.scatter(xbo[ind], ybo[ind], zbo[ind], color=colors_bo[f], marker='*', s=4, label='MOBO Pareto Front '+str(f+1))
    
#     ax.set_xlim( 0,10)
#     ax.set_ylim( -5,5)
#     ax.set_zlim( 0,1)
    
#     ax.set_xlabel('SAS')
#     ax.set_ylabel('logP')
#     ax.set_zlabel('QED')
    
#     ax.legend()
    
    
#     for angle in range(0,360, 30):
#         ax.view_init(azim=angle)
#         filename=os.path.join(run_dir, 'results/samples_del/fig_pateto'+str(angle)+'.pdf')
#         if with_bo:
#             filename=os.path.join(run_dir, 'results/bo/fig_pateto_with_bo'+str(angle)+'.pdf')
#         #fig.savefig(filename,bbox_inches='tight')
#         fig.savefig(filename)
    
    
def plot_pareto_fronts(run_dir, fronts=[0,1,2], color=None, with_bo=False):
    pop_final = pd.read_csv(os.path.join(run_dir, 'results/samples_del/new_pop_final.csv'))
    z = pop_final['qed'].to_numpy()
    x = pop_final['SAS'].to_numpy()
    y = pop_final['logP'].to_numpy()
    rank = pop_final['rank'].to_numpy()         
    
    if color is None:
        colors=['red', 'blue', 'cyan', 'olive', 'purple', 'brown', 'lime', 'orchid']
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #for f in fronts[::-1]:
    for f in fronts:
        ind = (rank==f)
        c = colors[f]
        ax.scatter(x[ind], y[ind], z[ind], color=c, marker='.', s=2, label='DEL Front '+str(f+1))
    
    bo_file = os.path.join(run_dir, 'results/bo/samples_qehvi.csv')
    with_bo = with_bo & os.path.isfile(bo_file)
    bo_file2 = os.path.join(run_dir, 'results/bo/samples_qparego.csv')
    if with_bo:
        bo_results = pd.read_csv(bo_file)
        zbo = bo_results['qed'].to_numpy()
        xbo = bo_results['SAS'].to_numpy()
        ybo = bo_results['logP'].to_numpy()
        batch = bo_results['batch'].to_numpy()
        colors_bo = ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke']
        batchmax = batch.max()
        
        bo_results2 = pd.read_csv(bo_file2)
        zbo2 = bo_results2['qed'].to_numpy()
        xbo2 = bo_results2['SAS'].to_numpy()
        ybo2 = bo_results2['logP'].to_numpy()
        #batch2 = bo_results2['batch'].to_numpy()
        colors_bo2 = ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray', 'gainsboro', 'whitesmoke']
        
        #for f in fronts[::-1]:
        for f in fronts:
            ind = (batch==(batchmax-f)) & (zbo!=10)
            ax.scatter(xbo[ind], ybo[ind], zbo[ind], color=colors_bo[f], marker='*', s=4, label='qEHVI Front '+str(f+1))
        for f in fronts:
            ind = (batch==(batchmax-f)) & (zbo!=10)
            ax.scatter(xbo2[ind], ybo2[ind], zbo2[ind], color=colors_bo2[f], marker='D', s=2, label='qParEGO Front '+str(f+1))
    
    ax.set_xlim( 0,10)
    ax.set_ylim( -5,5)
    ax.set_zlim( 0,1)
    
    ax.set_xlabel('SAS')
    ax.set_ylabel('logP')
    ax.set_zlabel('QED')
    
    ax.legend()
    
    
    for angle in range(0,360, 30):
        ax.view_init(azim=angle)
        filename=os.path.join(run_dir, 'results/samples_del/fig_pateto'+str(angle)+'.pdf')
        if with_bo:
            filename=os.path.join(run_dir, 'results/bo/fig_pateto_with_bo'+str(angle)+'.pdf')
        #fig.savefig(filename,bbox_inches='tight')
        fig.savefig(filename)    
    
