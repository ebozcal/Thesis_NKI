import imp
from operator import index
import os
import pandas as pd
import numpy as np


#pd.set_option("display.max_rows", None)


df1=pd.read_excel("/processing/ertugrul/MPM/v2/initiate_processed_os.xlsx")
df2= pd.read_excel("/processing/ertugrul/MPM/v2/nivomes_processed_os.xlsx")
df3= pd.read_excel("/processing/ertugrul/MPM/v2/pemmela_processed_os.xlsx")
df_p=pd.read_excel("/processing/ertugrul/MPM/v2/nvalt19_processed_os.xlsx")
 

#print("df1 :{}, df2 : {}, df3 : {}, df4 : {}".format(df1.shape, df2.shape, df3.shape, df4.shape))

list1_image = os.listdir("/processing/ertugrul/MPM/initiate/ct/")
list1_mask = os.listdir("/processing/ertugrul/MPM/initiate/v10/")

#print("list1_image_len = {}".format(len(list1_image)))
#print("list1_mask_len = {}".format(len(list1_mask)))

list2_image = os.listdir("/processing/ertugrul/MPM/nivomes/ct/")
list2_mask = os.listdir("/processing/ertugrul/MPM/nivomes/v10/")

#print("list2_image_len = {}".format(len(list2_image)))
#print("list2_mask_len = {}".format(len(list2_mask)))

list3_image = os.listdir("/processing/ertugrul/MPM/pemmela/ct/")
list3_mask = os.listdir("/processing/ertugrul/MPM/pemmela/v10/")

#print("list3_image_len = {}".format(len(list3_image)))
#print("list3_mask_len = {}".format(len(list3_mask)))

#list_mpm_images = list1_image + list2_image + list3_image
#list_mpm_masks = list1_mask + list2_mask + list3_mask

list_p_images = os.listdir("/processing/ertugrul/MPM/nvalt19/ct/")
list_p_masks = os.listdir("/processing/ertugrul/MPM/nvalt19/v10/")

#xxxxxxxxxxxxxxxxxxxxxx
matched_list1_ID=[]
matched_list_1 = []
for i in sorted(list1_image):
    for j in df1["ID_date"]:
         if j == i[:17]:
            matched_list1_ID.append(j)
            matched_list_1.append(i)

print("matched_list_mpm_ID = {}".format(len(matched_list1_ID)))
print("matched_list_mpm_path = {}".format(len(matched_list_1)))
matched_list_1 = [os.path.join("/processing/ertugrul/MPM/initiate/ct/", x)  for x in matched_list_1]

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

matched_list2_ID=[]
matched_list_2 = []
for i in sorted(list2_image):
    for j in df2["ID_date"]:
         if j == i[:17]:
            matched_list2_ID.append(j)
            matched_list_2.append(i)
matched_list_2 = [os.path.join( "/processing/ertugrul/MPM/nivomes/ct/", x)  for x in matched_list_2]

print("matched_list_mpm_ID = {}".format(len(matched_list2_ID)))
print("matched_list_mpm_path = {}".format(len(matched_list_2)))

#xxxxxxxxxxxxxxxxxxxxxxxxx
matched_list3_ID=[]
matched_list_3 = []
for i in sorted(list3_image):
   
    for j in df3["ID_date"]:
         print(j)
         print(i[:17])

         if j == i[:17]:
            matched_list3_ID.append(j)
            matched_list_3.append(i)
matched_list_3 = [os.path.join("/processing/ertugrul/MPM/pemmela/ct/", x)  for x in matched_list_3]

print("matched_list_mpmpemmela_ID = {}".format(len(matched_list3_ID)))
print("matched_list_mpmpemmela_path = {}".format(len(matched_list_3)))
#xxxxxxxxxxxxxxxxxxxxxxxxx

matched_list_p_ID=[]
matched_list_p_path =[]
for i in sorted(list_p_images):
    for j in df_p["ID_date"]:
        if j == i[:14]:
            #print("list_p_ID = {}, Image_path = {}".format(j, i))
            matched_list_p_ID.append(j)
            matched_list_p_path.append(i)
matched_list_p_path = [os.path.join("/processing/ertugrul/MPM/nvalt19/ct/", x)  for x in matched_list_p_path]


print("matched_list_pnvalt_ID = {}".format(len(matched_list_p_ID)))
print("matched_list_pnvalt_path = {}".format(len(matched_list_p_path)))

#zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

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
matched1_path_mask = [os.path.join("/processing/ertugrul/MPM/initiate/v10/", x)  for x in matched1_path_mask]

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
matched2_path_mask = [os.path.join("/processing/ertugrul/MPM/nivomes/v10/", x)  for x in matched2_path_mask]

#xxxxxxxxxxxxxxxxxxxxxxxxxx
matched3_ID_mask=[]
matched3_path_mask = []
for i in sorted(list3_mask):
    for j in df3["ID_date"]:
        #z =1
        #while z <5:
         #   print(df_mpm["ID_date"])
        if j == i[:17]:
            #print("matched_list_mpm_ID = {}, matched_Image_mpm_path = {}".format(j, i))
            matched3_ID_mask.append(j)
            matched3_path_mask.append(i)
matched3_path_mask = [os.path.join("/processing/ertugrul/MPM/pemmela/v10/", x)  for x in matched3_path_mask]

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
matched_list_p_ID_mask=[]
matched_list_p_path_masks =[]
for i in sorted(list_p_masks):
    for j in df_p["ID_date"]:
        if j == i[:14]:
            #print("list_p_ID = {}, Image_path = {}".format(j, i))
            matched_list_p_ID_mask.append(j)
            matched_list_p_path_masks.append(i)
matched_list_p_path_masks = [os.path.join("/processing/ertugrul/MPM/nvalt19/v10/", x)  for x in matched_list_p_path_masks]


print("matched_list_p_ID = {}".format(len(matched_list_p_ID)))
print("matched_list_p_path_masks = {}".format(len(matched_list_p_path_masks)))

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Merged_path_list = matched_list_1 + matched_list_2 + matched_list_3 + matched_list_p_path
Merged_mask_list = matched1_path_mask + matched2_path_mask + matched3_path_mask + matched_list_p_path_masks
Merged_ID_list = matched1_ID_mask + matched2_ID_mask + matched3_ID_mask + matched_list_p_ID

dict_path_ID = {"Image":Merged_path_list,  "Mask":Merged_mask_list, "ID":Merged_ID_list}
df_paths_ID = pd.DataFrame(dict_path_ID)

print(df_paths_ID)

df_paths_ID.to_csv("df1003_w_paths_ID.csv")

