import imp
from operator import index
import os
import pandas as pd
import numpy as np


#pd.set_option("display.max_rows", None)

df1= pd.read_excel("/SHARED/active_Ertugrul/excel_files/initiate_processed_os.xlsx")
df2=pd.read_excel("/SHARED/active_Ertugrul/excel_files/nvalt19_processed_os.xlsx")
df_merged = pd.concat([df1, df2], axis=0)

print("df_merged shape", df_merged.shape)


df_merged["ID"] = df_merged["ID_date"].apply(lambda x: x[:8])
df_merged["date"] = df_merged["ID_date"].apply(lambda x: x[9:])


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
    frame = frame.sort_values(by = "date")
    z+=1
    
    if frame.shape[0] > 1:
        v1 = frame["vol_ai"][frame.index[0]]
        for i in range(len(frame.index)):
            if i < (len(frame.index)-1):
                p1 = calculate_change(frame['vol_ai'][frame.index[i+1]], v1)
                if p1 < -0.2 and (abs(frame['vol_ai'][frame.index[i+1]] - v1) >24):
                    frame.iloc[0, 3] = 1
                    df_merged.iloc[frame.index[0], 3] = 1
                    break
                for i in range(len(frame.index)):
                    if i < (len(frame.index)-1):
                        cr = calculate_change(frame["vol_ai"][frame.index[i+1]], frame["vol_ai"][frame.index[i]])
                        cv1 = abs(frame['vol_ai'][frame.index[i+1]] - v1)
                        cv2 = abs(frame['vol_ai'][frame.index[i+1]] - frame['vol_ai'][frame.index[i]])

                        if (p1 > 0.2 and cv1 >24)  or ( cr > 0.2 and cv2 >24):
                            frame.iloc[0, 3] = 2
                            df_merged.iloc[frame.index[0], 3] = 2
                        elif  (-0.2 < p1 < 0.2 and cv1 <24) or ( -0.2 < cr < 0.2  and cv2 <24):

                            frame.iloc[0, 3] = 0
                            df_merged.iloc[frame.index[0], 3] = 0

print("label_counts:", df_merged["label"].value_counts())
#print(df_merged)


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
df_path1 = pd.read_csv("/processing/ertugrul/Part_2/CNN_wholeCT_datagenerator/236_BL/133/initiate_path.csv")
df_path2 = pd.read_csv("/processing/ertugrul/Part_2/CNN_wholeCT_datagenerator/236_BL/133/nvalt19_path.csv")
df_path = pd.concat([df_path1, df_path2], axis=0)

df_164 = df_path[df_path["ID"].isin(df_merged["ID_date"].values)]
df_164.reset_index(drop=True, inplace= True)
df_164 = df_164[['Image', 'Mask', 'ID']]

df_164["label"] = 164*[3]
for i in range(df_164.shape[0]):
    for j in range(df_merged.shape[0]):
        if df_164["ID"][i]==df_merged["ID_date"][j]:
            df_164.iloc[i, 3] = df_merged["label"][j]

print(df_164)
print(df_164.columns)

print(df_164["label"].value_counts())


df_164.to_csv("Labels_164_BL_CNN.csv")