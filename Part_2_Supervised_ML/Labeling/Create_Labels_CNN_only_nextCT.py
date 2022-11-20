import imp
from operator import index
import os
import pandas as pd
import numpy as np


#pd.set_option("display.max_rows", None)


df1=pd.read_excel("/processing/ertugrul/MPM/v2/initiate_processed_os.xlsx")
df2= pd.read_excel("/processing/ertugrul/MPM/v2/nivomes_processed_os.xlsx")
df3= pd.read_excel("/processing/ertugrul/MPM/v2/pemmela_processed_os.xlsx")
df4=pd.read_excel("/processing/ertugrul/MPM/v2/nvalt19_processed_os.xlsx")

df1 = df1[["ID_date", "vol_ai"]]
df2 = df2[["ID_date", "vol_ai"]]
df3 = df3[["ID_date", "vol_ai"]]
df4 = df4[["ID_date", "vol_ai"]]
print(df1.shape, df2.shape, df3.shape, df4.shape)

df_merg = pd.concat([df1, df2, df3], axis = 0)
df_merg["ID"] = df_merg["ID_date"].apply(lambda x: x[:8])
df_merg["date"] = df_merg["ID_date"].apply(lambda x: x[9:])

df4["ID"] = df4["ID_date"].apply(lambda x: x[:5])
df4["date"] = df4["ID_date"].apply(lambda x: x[6:])

df_merged = pd.concat([df_merg, df4], axis = 0)
df_merged = df_merged[["ID", "date", "vol_ai"]]

df_merged = df_merged.sort_values(by="ID")
df_merged.reset_index(inplace=True, drop=True)

df_merged['label'] = [np.nan]*df_merged.shape[0]

def calculate_change(v2, v1):
    change = (v2 - v1)/v1
    return change

PI_groups = df_merged.groupby("ID")

z=0
for patient, frame in PI_groups:
    z+=1
    frame = frame.sort_values(by = "date")

    if frame.shape[0] > 1:    
        for j in range(len(frame.index)):
            if j < (len(frame.index)-1):

                if j==0:
                    v1 = frame["vol_ai"][frame.index[0]]
                    for i in range(len(frame.index)):
                        if i < (len(frame.index)-1) and v1!=0:
                            p1 = calculate_change(frame['vol_ai'][frame.index[i+1]], v1)
                            if p1 < -0.2 and (abs(frame['vol_ai'][frame.index[i+1]] - v1) >24):

                                df_merged.iloc[frame.index[0], 3] = 1
                                frame.iloc[0, 3] = 1
                                break
                            if p1 > 0.2 and (abs(frame['vol_ai'][frame.index[i+1]] - v1) >24):
                                df_merged.iloc[frame.index[0], 3] = 2
                                frame.iloc[0, 3] = 2
                            else:
                                frame.iloc[0, 3] = 0
                                df_merged.iloc[frame.index[0], 3] = 0

                if j >0 :
                    if calculate_change(frame["vol_ai"][frame.index[j+1]], frame["vol_ai"][frame.index[j]]) > 0.2 and (abs(frame['vol_ai'][frame.index[j+1]] - abs(frame['vol_ai'][frame.index[j]])) >24):
                        df_merged.iloc[frame.index[j], 3] = 2
                        frame.iloc[j, 3] = 2
                    elif calculate_change(frame["vol_ai"][frame.index[j+1]], frame["vol_ai"][frame.index[j]]) < -0.2 and (abs(frame['vol_ai'][frame.index[j+1]] - abs(frame['vol_ai'][frame.index[j]])) >24):
                        df_merged.iloc[frame.index[j], 3] = 1
                        frame.iloc[j, 3] = 1
                    else:
                        frame.iloc[j, 3] = 0
                        df_merged.iloc[frame.index[j], 3] = 0

print("Patients_number = {}".format(z))
print("label_countts:", df_merged["label"].value_counts())

# Drop rows which has NaN values
df_merged = df_merged.dropna()
df_merged.reset_index(drop=True, inplace=True)

ID_date = []
for i in range(df_merged.shape[0]):
    if df_merged["ID"][i][0]== "M":
        ID_date.append(df_merged["ID"][i] + "-" + df_merged["date"][i])
    elif df_merged["ID"][i][0]== "P":
                ID_date.append(df_merged["ID"][i] + "_" + df_merged["date"][i])



df_merged["ID_date"] = ID_date


#Transfer columns which contain Mask and Images paths
df_1000 = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/df1003_w_paths_ID.csv")

#df_1000["ID1"] = 1003*[0]
#df_1000["date"] = 1003*[0]

#for i, j in enumerate(df_1000["ID"]):
#    if j[0]=="M":
 #       df_1000.iloc[i, 5] = j[9:]
  #      df_1000.iloc[i, 4] = j[:8]
   # elif j[0]=="P":
    #    df_1000.iloc[i, 5] = j[6:]
      #  df_1000.iloc[i, 4] = j[:5]

df_1000 = df_1000[df_1000["ID"].isin(df_merged["ID_date"].values)]
#df_1000 = df_1000[df_1000["ID1"].isin(df_merged["ID"].values)]
df_1000.reset_index(drop=True, inplace= True)
df_1000 = df_1000[['Image', 'Mask', 'ID']]


print(df_1000.shape)
print("xxxxxxxxxxxxxxxxxxxxx")
df_1000["label"] = 823*[3]
for i in range(df_1000.shape[0]):
    for j in range(df_merged.shape[0]):
        if df_1000["ID"][i]==df_merged["ID_date"][j]:
            df_1000.iloc[i, 3] = df_merged["label"][j]

print(df_1000)
print(df_merged)
print(df_1000.columns)

print(df_1000["label"].value_counts())



#df_merged[["Image", "Mask"]] = df_1000[["Image", "Mask"]]

df_1000.to_csv("CNN_823_paths_labels.csv")
