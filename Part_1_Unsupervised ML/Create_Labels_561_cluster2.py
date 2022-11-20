import imp
from operator import index
import os
import pandas as pd
import numpy as np


#pd.set_option("display.max_rows", None)

df1= pd.read_excel("/SHARED/active_Ertugrul/excel_files/initiate_processed_os.xlsx")

df2=pd.read_excel("/SHARED/active_Ertugrul/excel_files/nvalt19_processed_os.xlsx")


df = pd.concat([df1, df2], axis=0)
df = df.sort_values(by="ID")
df.reset_index(inplace=True)
df.drop(["index"], axis = 1,  inplace= True)


#dummy = np.empty((561, 107))
#dummy[:] = np.nan

#df_diff = pd.DataFrame(dummy, columns = df.columns)

df['label'] = [np.nan]*df.shape[0]
print('df.shape:', df)

def calculate_change(v2, v1):
    change = (v2 - v1)/v1
    return change
print(df.columns)
PI_groups = df.groupby("ID")
for patient, frame in PI_groups:
    frame = frame.sort_values(by = "date")

    if frame.shape[0] > 1:
        v1 = frame["vol_ai"][frame.index[0]]
        for j in range(len(frame.index)):
            if j < (len(frame.index)-1):
                for i in range(len(frame.index)):
                    if i < (len(frame.index)-1):
                        p1 = calculate_change(frame['vol_ai'][frame.index[i+1]], v1)
                        if p1 < -0.2 and (abs(frame['vol_ai'][frame.index[i+1]] - v1) >24):
                            df.iloc[frame.index[0], 8] = 1
                            frame.iloc[0, 8] = 1
                            break
                        if p1 > 0.2 and (abs(frame['vol_ai'][frame.index[i+1]] - v1) >24):
                            df.iloc[frame.index[0], 8] = 2
                            frame.iloc[0, 8] = 2
                        else:
                            frame.iloc[0, 8] = 0
                            df.iloc[frame.index[0], 8] = 0

                if j >0 :
                    if calculate_change(frame["vol_ai"][frame.index[j+1]], frame["vol_ai"][frame.index[j]]) > 0.2 and (abs(frame['vol_ai'][frame.index[j+1]] - abs(frame['vol_ai'][frame.index[j]])) >24):
                        df.iloc[frame.index[j], 8] = 2
                        frame.iloc[j, 8] = 2
                    elif calculate_change(frame["vol_ai"][frame.index[j+1]], frame["vol_ai"][frame.index[j]]) < -0.2 and (abs(frame['vol_ai'][frame.index[j+1]] - abs(frame['vol_ai'][frame.index[j]])) >24):
                        df.iloc[frame.index[j], 8] = 1
                        frame.iloc[j, 8] = 1
                    else:
                        frame.iloc[j, 8] = 0
                        df.iloc[frame.index[j], 8] = 0


    #print(frame)

# remove row which is in the label list but not in the image list
# Drop rows which has NaN values

df_2000 = pd.read_csv("/home/ertugrul/cluster_2_561/561_feature_full.csv")

df.drop([204, 269], inplace = True)
df = df.reset_index(drop=True)
df = df.drop("increase", axis = 1)
df_561_fet_full = pd.read_csv("/home/ertugrul/cluster_2_561/selected_features_corr_var_561.csv")
df_561_fet_full["label"] = df["label"] 
df_561_fet_full[["Image", "Mask"]] = df_2000[["Image", "Mask"]]

df_561_fet_full = df_561_fet_full.dropna()
print(df_561_fet_full.columns)
print(df_561_fet_full.shape)

df_561_fet_full.to_csv("df_426__withlabel_mask.csv")

df = df.dropna()
df.reset_index(drop = True, inplace=True)

print(df ["label"].value_counts())





#
#df["Image"] =df_561_fet_full["Image"].apply(lambda x: x[43:62])
df.to_csv("Labels_561_cluster2_v3.csv")
