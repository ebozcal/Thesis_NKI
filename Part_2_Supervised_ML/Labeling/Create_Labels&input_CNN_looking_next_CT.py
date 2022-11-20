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
#print(df1.shape, df2.shape, df3.shape, df4.shape)

# Merge 3 MPM files in order to compare it with CT images and masks files
df_merg = pd.concat([df1, df2, df3], axis = 0)
#Create ID and date columns by seperating ID_date columns
df_merg["ID"] = df_merg["ID_date"].apply(lambda x: x[:8])
df_merg["date"] = df_merg["ID_date"].apply(lambda x: x[9:])

df4["ID"] = df4["ID_date"].apply(lambda x: x[:5])
df4["date"] = df4["ID_date"].apply(lambda x: x[6:])
# Merge 3 MPM files with  P file in order to compare it with CT images and masks files
df_merged = pd.concat([df_merg, df4], axis = 0)
df_merged = df_merged[["ID", "date", "vol_ai"]]

#Sort values by ID
df_merged = df_merged.sort_values(by="ID")
df_merged.reset_index(inplace=True, drop=True)

#Create label columns to be filled in line with the calculation based on the vol_AI change
df_merged['label'] = [np.nan]*df_merged.shape[0]
df_merged['vc'] = [np.nan]*df_merged.shape[0]
print("df_merged columns1:", df_merged.columns)

#Define the function to calculate the percentage of the volume change
def calculate_change(v2, v1):
    change = (v2 - v1)/v1
    return change

# Create frames based on the CT ID's
PI_groups = df_merged.groupby("ID")
z=0
l=0
vc = []
for patient, frame in PI_groups:
    frame = frame.sort_values(by = "date")
    z +=1
    if frame.shape[0] > 1:
        # Loops for the CT to be labelled
        for i in range(len(frame.index)):
            # Loop for the CT's to be checked what is the vol AI change percentage in order to label the subject CT above
            if i+1 < (len(frame.index)):
                v2 = frame['vol_ai'][frame.index[i+1]]
                p1 = calculate_change(v2, frame['vol_ai'][frame.index[i]])
                vc = abs(frame['vol_ai'][frame.index[i+1]] - frame['vol_ai'][frame.index[i]])
                if p1 < -0.1 and vc >24:
                    frame.iloc[i, 3] = 1
                    df_merged.iloc[frame.index[i], 3] = 1
                    df_merged.iloc[frame.index[i], 4] = p1
                elif  p1 > 0.1 and vc >24:
                    frame.iloc[i, 3] = 2
                    df_merged.iloc[frame.index[i], 3] = 2
                    df_merged.iloc[frame.index[i], 4] = p1
                elif  -0.1 < p1 < 0.1 or vc <24:
                    frame.iloc[i, 3] = 0
                    df_merged.iloc[frame.index[i], 3] = 0
                    df_merged.iloc[frame.index[i], 4] = p1

 
    #if  z<5:
     #   print(frame)
      #  print("----------------------------------------------")
       # print(df_merged.iloc[l:l+15])
        #print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")

    #l += frame.shape[0]
#print("label_counts:", df_merged["label"].value_counts())


#print("Patients_number = {}".format(z))
print("label_countts:", df_merged["label"].value_counts())

# Drop rows which has NaN values
df_merged = df_merged.dropna()
df_merged.reset_index(drop=True, inplace=True)
print("df_merged columns2:", df_merged.columns)

ID_date = []
for i in range(df_merged.shape[0]):
    if df_merged["ID"][i][0]== "M":
        ID_date.append(df_merged["ID"][i] + "-" + df_merged["date"][i])
    elif df_merged["ID"][i][0]== "P":
                ID_date.append(df_merged["ID"][i] + "_" + df_merged["date"][i])



df_merged["ID_date"] = ID_date


#Transfer columns which contain Mask and Images paths
df_1000 = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/df1003_w_paths_ID.csv")

df_1000 = df_1000[df_1000["ID"].isin(df_merged["ID_date"].values)]
df_1000.reset_index(drop=True, inplace= True)
df_1000 = df_1000[['Image', 'Mask', 'ID']]


#print(df_1000.shape)
#print("xxxxxxxxxxxxxxxxxxxxx")
df_1000["label"] = 762*[3]
df_1000["vc"] = 762*[3]

for i in range(df_1000.shape[0]):
    for j in range(df_merged.shape[0]):
        if df_1000["ID"][i]==df_merged["ID_date"][j]:
            df_1000.iloc[i, 3] = df_merged["label"][j]
            df_1000.iloc[i, 4] = df_merged["vc"][j]


print(df_1000)
#print(df_merged)
#print(df_1000.columns)

print(df_1000["label"].value_counts())
print(df_1000)




#df_merged[["Image", "Mask"]] = df_1000[["Image", "Mask"]]

df_1000.to_csv("CNN_762_paths_labels_input_10_p1.csv")
