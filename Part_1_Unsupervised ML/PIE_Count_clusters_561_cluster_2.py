import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/processing/ertugrul/Part_1/Cluster_561CT/cluster_count_561_v2.csv")

print(df)

label_list = df['label'].to_list()
kmeans = df['KMeans'].to_list()    
print("KMeans:", df["KMeans"].value_counts())

cluster_00 = 0
cluster_01 =0
cluster_02=0
cluster_10 = 0
cluster_11 =0
cluster_12=0
cluster_20 = 0
cluster_21 =0
cluster_22=0
#for i in range(len(resultk)):
 #   if label_list[i] == 1 and resultk[i] == 0:
  #      cluster_a_uni +=1
   # if label_list[i] == 2 and resultk[i] == 1:
    #    cluster_b_uni +=1
    #if label_list[i] == 0 and resultk[i] == 2:
     #   cluster_c_uni +=1

for i in range(len(kmeans)):
    if kmeans[i] == 0:
        if label_list[i] == 0:
            cluster_00 +=1
        if label_list[i] == 1:
            cluster_01 +=1
        if label_list[i] == 2:
            cluster_02 +=1

    if kmeans[i] == 1:
        if label_list[i] == 0:
            cluster_10 +=1
        if label_list[i] == 1:
            cluster_11 +=1
        if label_list[i] == 2:
            cluster_12 +=1

    if kmeans[i] == 2:
        if label_list[i] == 0:
            cluster_20 +=1
        if label_list[i] == 1:
            cluster_21 +=1
        if label_list[i] == 2:
            cluster_22 +=1
            
   

   
print("cluster_00" , cluster_00)
print ("cluster_01" , cluster_01)
print("cluster_02" , cluster_02)
print("cluster_10" , cluster_10)
print("cluster_11" , cluster_11)
print("cluster_12" , cluster_12)
print("cluster_20" , cluster_20)
print("cluster_21" ,cluster_21 )
print("cluster_22" , cluster_22)

data0 = [93, 14, 22]
data1 = [142, 34, 98]
#data = [0, 9, 19]
#data3 = [13, 4, 6]
labels = ["Stable", "Response", "Progression"]
#labels = ["1st Cluster", "2nd Cluster", "3rd Cluster"]
#plt.pie(data1, labels=labels, autopct = "%1.0f%%", textprops={'fontsize': 50})
#plt.title("Classes in the 2nd Cluster", textprops={'fontsize': 50})
#plt.pie(data3, labels=labels , autopct = "%1.0f%%")
#plt.title("Distribution of Responses in the Clusters")


fig, (ax1,  ax2)= plt.subplots(1, 2, figsize=(7, 4))
fig.subplots_adjust(right =1, left = 0.05, top=0.85,  bottom=0.20)
ax1.pie(data0, labels=labels, autopct = "%1.0f%%", textprops={'fontsize': 11})
ax2.pie(data1, labels=labels, autopct = "%1.0f%%", textprops={'fontsize': 11})
ax1.set_title("Classes in the 1st Cluster")
ax2.set_title("Classes in the 2nd Cluster")

plt.show()