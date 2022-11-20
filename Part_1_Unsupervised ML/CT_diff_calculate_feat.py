import pandas as pd
import numpy as np

df = pd.read_csv("/home/ertugrul/selected_features_corr_var_561.csv")
df_label= pd.read_csv("/home/ertugrul/Labels_561_cluster2_v2.csv", header = 0)
#pd.set_option('display.max_rows', None)
df = df.iloc[:, 3:]
df[["ID" , "date", "label"]] = df_label[["ID" , "date",  "label"]]

#pd.set_option('display.max_columns', None)


PI_groups = df.groupby("ID")

dummy = np.empty((561, 107))
dummy[:] = np.nan

df_diff = pd.DataFrame(dummy, columns = df.columns)

print("df_diff_first_shape", df_diff.shape)
print("df_diff_columns", df_diff.columns)

for p, frame in PI_groups:
    frame = frame.sort_values(by="date")
    for i in range(len(frame.index)):
        if i < (len(frame.index)-1) and len(frame.index) >1:     
            time_diff = abs((pd.Timestamp(df.iloc[frame.index[i+1], 105]) - pd.Timestamp(df.iloc[frame.index[i], 105])).days)
            for j in range(len(df_diff.columns)-3):
                if time_diff <85 and i+2 < (len(frame.index)-1):
                    df_diff.iloc[frame.index[i], j] = df.iloc[frame.index[i+1], j] - df.iloc[frame.index[i], j]
                    df_diff.iloc[frame.index[i], 106] = frame.iloc[i+2, 106]
                    new_ID = (frame.iloc[i+2, 104] + "_" + str(i))
               
                    df_diff.iloc[frame.index[i], 104] = new_ID                  

df_diff["date"] = df_label[ "date"]
df_diff = df_diff.dropna()
df_diff.reset_index(drop = True, inplace=True)

print(df_diff["label"].value_counts())

df_diff.to_csv("CT_feat_diff_222.csv")
print("df_diff_last_shape:", df_diff.shape)
print(df_diff.columns)

