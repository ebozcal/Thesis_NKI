import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/ertugrul/cluster_3_CT_diff/cluster_count_CTdiff.csv")

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

#data0 = [116, 8, 28]
data1 = [56, 3, 8]
#data = [0, 9, 19]
#data3 = [15, 32, 9]
labels = ["Stable", "Response", "Progression"]
#labels = ["1st Cluster", "2nd Cluster", "3rd Cluster"]
plt.pie(data1, labels=labels, autopct = "%1.0f%%" )
plt.title("Classes in the 2nd Cluster")
#plt.pie(data3, labels=labels , autopct = "%1.0f%%")
#plt.title("Distribution of Responses in the Clusters")
plt.show()
