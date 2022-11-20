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



#df_corr= pd.read_csv("/home/ertugrul/CT_diff.csv", header = 0)

df= pd.read_csv("/home/ertugrul/cluster_3_CT_diff/CT_feat_diff_222.csv", header = 0)
#pd.set_option('display.max_rows', No


X = df.iloc[:, 1:105]
df2 = df.iloc[:, 1:105]

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
        result = cluster.KMeans(n_clusters= 3).fit_predict(X)
        df2[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)       
              
    elif cl[i] == 'Fuzzy_Cmeans':
        model =  FCM(n_clusters=3)
        model.fit(X)
        result = model.predict(X)
        df2[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)

    elif cl[i] == 'Affinity Propagation':
        result = cluster.AffinityPropagation(damping=0.9, preference= -500).fit_predict(X)
        df2[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)

    elif cl[i] == 'GaussianMixture' :
        gmm = mixture.GaussianMixture( n_components=3, covariance_type='full')
        gmm.fit(X)
        result = gmm.predict(X)
        df2[cl[i]] = pd.DataFrame(result)
        cl_df.loc[cl_df.Name == cl[i], 'Silhouette-Coeff'] = metrics.silhouette_score(X, result, metric='euclidean')
        cl_df.loc[cl_df.Name == cl[i], 'Calinski-Harabaz'] = metrics.calinski_harabasz_score(X, result)
   
print("Kmeans:", df2["KMeans"].value_counts())
print("Fuzzy_Cmeans:", df2["Fuzzy_Cmeans"].value_counts())
print("Affinity Propagation:", df2["Affinity Propagation"].value_counts())
print("GaussianMixture:", df2["GaussianMixture"].value_counts())

print("cl_df:", cl_df)


finalDf = pd.concat([df2, principalDf], axis = 1)
finalDf["label"] = df["label"]

df_cluster =  finalDf[["KMeans", "label"]]
df_cluster.to_csv("cluster_count_CTdiff.csv")

print("final_dfcol_len:", len(finalDf.columns))
print("final_dfcolumns:", finalDf.columns)
#print(finalDf.columns)

# 4 X 2 plot
fig,ax = plt.subplots(2, 2, figsize=(4, 7)) 
x = 0
y = 0
z = 0
classes = [0, 1, 2]
for i in cl:
    result= ax[x,y].scatter(finalDf.iloc[:, 108], finalDf.iloc[:, 109],  c=finalDf.iloc[:, 104 + z])
    ax[x,y].set_title(i + " Cluster Result Differences")
    ax[x, y].set_xlabel("PC1")
    ax[x, y].set_ylabel("PC2")
    ax[x, y].legend(handles=result.legend_elements()[0], labels=classes,  loc=3, fontsize = 10)
    y = y + 1
    z = z + 1
    if y == 2:
        x = x + 1
        y = 0
       # plt.subplots_adjust(bottom=-0.5, top=1.5)
plt.tight_layout()
#plt.show() 

fig,ax = plt.subplots(1, 1, figsize=(4, 6))
result = ax.scatter(finalDf.iloc[:, 108], finalDf.iloc[:, 109],  c=finalDf.iloc[:, 110]) 
ax.set_title("Cluster Result with feature differences")
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(handles=result.legend_elements()[0], labels=classes,  loc=3, fontsize = 10)

#   plt.tight_layout()
plt.show()
 
