import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset

from sklearn import metrics
from fcmeans import FCM
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.decomposition import PCA
#init_notebook_mode(connected=True)
# %matplotlib inline
#warnings.filterwarnings('ignore')

import seaborn as sns




df_label= pd.read_csv("/home/ertugrul/clusters_2/combined_MPM_P_labels_561.csv", header = 0)
df_inpath= pd.read_csv("/home/ertugrul/clusters_2/initiate_path.csv", header = 0)
df_nv19path= pd.read_csv("/home/ertugrul/clusters_2/nvalt19_path.csv", header = 0)
df_path = pd.concat([df_inpath, df_nv19path], axis=0)

df_199 = pd.read_csv("/home/ertugrul/clusters_2/initiate_result.csv")
df_363 = pd.read_csv("/home/ertugrul/clusters_2/nvalts19_result.csv")
df_conc = pd.concat([df_199, df_363], axis=0)

df_corr= pd.read_csv("/home/ertugrul/clusters_2/selected_features_corr_var_561.csv", header = 0)

print("df_lab:", df_label.shape)
print("df_inpath:", df_inpath.shape)
print("df_nv19path:", df_nv19path.shape)
print("df_path:", df_path.shape)
print("df_conc:", df_conc.shape)
print("df_corr:", df_corr.shape)

pd.set_option('display.max_rows', None)
#print('imagename:', df_path['Image'][1][0][43:])

# Drop rows which has NaN values
df_label.drop([269], inplace = True)
df_label.reset_index(drop = True, inplace=True)
path_list = df_path["Image"].to_list()
path_list_p = []
for i in sorted(path_list[199:]):
    path_list_p.append((i[44:58]))
#image_names = df2["image"].to_list()
#df1= pd.read_csv("/home/ertugrul/fs_var_final.csv", header = 0)
df_label_Plist = sorted(df_label["ID_date"][199:].to_list())

#print("path_list_p", path_list_p)
#print("df_label_Plist", df_label_Plist)
#print(df_label_Plist)

# remove row which is in the label list but not in the image list
list_dropped = []
for i in path_list_p:
    for j in df_label_Plist:
        if i == j:
            list_dropped.append(j)
print("list_dropped", len(list_dropped))

for i in path_list_p:
    if i in df_label_Plist:
        df_label_Plist.remove(i)
df_label.drop([204], inplace = True)
df_label.reset_index(drop = True, inplace=True)
#print("Diifference",df_label_Plist )
#print("df_label",df_label.iloc[199:205, :] )

print("df_lab:", df_label.shape)
print("df_corr:", df_corr.shape)    
#print(df_label_Plist)
X = df_corr.iloc[:, 2:]
df = df_corr.iloc[:, 2:]
#X = df1.iloc[:, 5:]
#df = df1.iloc[:, 5:]

cl_dist = {'Name' : ['KMeans','Fuzzy_Cmeans','Affinity Propagation', 'GaussianMixture']}
cl_df = pd.DataFrame(cl_dist)
cl=pd.Series(['KMeans','Fuzzy_Cmeans','Affinity Propagation', 'GaussianMixture'])

ss = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
#X =np.ascontiguousarray(X)
X = np.asarray(ss, order='C')
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

#X_result = pd.DataFrame(columns = cl)
for i in range(0,cl.size) :
    if cl[i]=='KMeans':
        resultk = cluster.KMeans(n_clusters= 3).fit_predict(X)
        df[cl[i]] = pd.DataFrame(resultk)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, resultk, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, resultk)       
              
    elif cl[i] == 'Fuzzy_Cmeans':
        model =  FCM(n_clusters=3)
        model.fit(X)
        result = model.predict(X)
        df[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)

    elif cl[i] == 'Affinity Propagation':
        result = cluster.AffinityPropagation(damping=0.9, preference= -500).fit_predict(X)
        df[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)

    elif cl[i] == 'GaussianMixture' :
        gmm = mixture.GaussianMixture( n_components=3, covariance_type='full')
        gmm.fit(X)
        resultg = gmm.predict(X)
        df[cl[i]] = pd.DataFrame(resultg)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, resultg, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, resultg)
   
print("Kmeans:", df["KMeans"].value_counts())
print("Fuzzy_Cmeans:", df["Fuzzy_Cmeans"].value_counts())
print("Affinity Propagation:", df["Affinity Propagation"].value_counts())
print("GaussianMixture:", df["GaussianMixture"].value_counts())

print("cl_df:", cl_df)


finalDf = pd.concat([df, principalDf], axis = 1)
finalDf["label"] = df_label["label"]
print("label_counts:", df_label["label"].value_counts())
#print("resultg:", resultg)
print(finalDf.columns)

df_cluster =  finalDf[["KMeans", "label"]]
df_cluster.to_csv("cluster_count.csv")

print("col_len:", len(finalDf.columns))
print("columns:", finalDf.columns)
#print(finalDf.columns)

# 4 X 2 plot
fig,ax = plt.subplots(2, 2, figsize=(4, 7)) 
x = 0
y = 0
z = 0
classes = [0, 1, 2]
for i in cl:
    result= ax[x,y].scatter(finalDf.iloc[:, 109], finalDf.iloc[:, 110],  c=finalDf.iloc[:, 105 + z])
    ax[x,y].set_title(i + " Cluster Result")
    ax[x, y].set_xlabel("PC1")
    ax[x, y].set_ylabel("PC2")
    ax[x, y].legend(handles=result.legend_elements()[0], labels=classes,  loc=1, fontsize = 10)
    y = y + 1
    z = z + 1
    if y == 2:
        x = x + 1
        y = 0
       # plt.subplots_adjust(bottom=-0.5, top=1.5)
plt.tight_layout()
#plt.show() 

fig,ax = plt.subplots(1, 1, figsize=(4, 6))
result = ax.scatter(finalDf.iloc[:, 109], finalDf.iloc[:, 110],  c=finalDf.iloc[:, 111]) 
ax.set_title("Cluster Result with Labels")
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

ax.set_xlim((-13, 15))
ax.set_ylim((-12, 15))
ax.legend(handles=result.legend_elements()[0], labels=classes,  loc=1, fontsize = 10)

#   plt.tight_layout()
plt.show()
 
