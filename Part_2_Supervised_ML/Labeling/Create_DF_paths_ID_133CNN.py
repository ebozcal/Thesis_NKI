import imp
from operator import index
import os
import pandas as pd
import numpy as np


#pd.set_option("display.max_rows", None)


df1=pd.read_excel("/SHARED/active_Ertugrul/excel_files/v2/initiate_processed_os.xlsx")
df2= pd.read_excel("/SHARED/active_Ertugrul/excel_files/v2/nivomes_processed_os.xlsx")


list1_image = os.listdir("/SHARED/active_Kevin/mpm/studies/initiate/ct/")
list1_mask = os.listdir("/SHARED/active_Kevin/mpm/studies/initiate/v10/")

list2_image = os.listdir("/SHARED/active_Kevin/mpm/studies/nivomes/ct/")
list2_mask = os.listdir("/SHARED/active_Kevin/mpm/studies/nivomes/v10/")
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

matched_list1_ID=[]
matched_list_1 = []
for i in sorted(list1_image):
    for j in df2["ID_date"]:
         if j == i[:17]:
            matched_list1_ID.append(j)
            matched_list_1.append(i)
matched_list_1 = [os.path.join( "/SHARED/active_Kevin/mpm/studies/nivomes/ct/", x)  for x in matched_list_1]

print("matched_list_mpm_ID = {}".format(len(matched_list1_ID)))
print("matched_list_mpm_path = {}".format(len(matched_list_1)))

#xxxxxxxxxxxxxxxxxxxxxxxxx
matched_list2_ID=[]
matched_list_2 = []
for i in sorted(list2_image):
    for j in df2["ID_date"]:
         if j == i[:17]:
            matched_list2_ID.append(j)
            matched_list_2.append(i)
matched_list_2 = [os.path.join("/SHARED/active_Kevin/mpm/studies/pemmela/ct/", x)  for x in matched_list_2]

print("matched_list_mpm_ID = {}".format(len(matched_list3_ID)))
print("matched_list_mpm_path = {}".format(len(matched_list_3)))

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
matched1_ID_mask=[]
matched1_path_mask = []
for i in sorted(list1_mask):
    for j in df1["ID_date"]:
        #z =1
        #while z <5:
         #   print(df_mpm["ID_date"])
        if j == i[:17]:
            #print("matched_list_mpm_ID = {}, matched_Image_mpm_path = {}".format(j, i))
            matched1_ID_mask.append(j)
            matched1_path_mask.append(i)
matched1_path_mask = [os.path.join("/SHARED/active_Kevin/mpm/studies/initiate/v10/", x)  for x in matched1_path_mask]

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
matched2_ID_mask=[]
matched2_path_mask = []
for i in sorted(list2_mask):
    for j in df2["ID_date"]:
        #z =1
        #while z <5:
         #   print(df_mpm["ID_date"])
        if j == i[:17]:
            #print("matched_list_mpm_ID = {}, matched_Image_mpm_path = {}".format(j, i))
            matched2_ID_mask.append(j)
            matched2_path_mask.append(i)
matched2_path_mask = [os.path.join("/SHARED/active_Kevin/mpm/studies/nivomes/v10/", x)  for x in matched2_path_mask]


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Merged_path_list = matched_list_1 + matched_list_2 
Merged_mask_list = matched1_path_mask + matched2_path_mask 
Merged_ID_list = matched1_ID_mask + matched2_ID_mask

dict_path_ID = {"Image":Merged_path_list,  "Mask":Merged_mask_list, "ID":Merged_ID_list}
df_paths_ID = pd.DataFrame(dict_path_ID)

print(df_paths_ID)

df_paths_ID.to_csv("df1003_paths_ID.csv")

