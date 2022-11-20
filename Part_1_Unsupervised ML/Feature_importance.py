from cgitb import text
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/processing/ertugrul/Part_1/Cluster_561CT/cluster_count_561_v2.csv")
df2 = pd.read_csv("/processing/ertugrul/Part_1/Cluster_561CT/df_561_fet_corrvarr_withlabel.csv")

print(df['KMeans'].value_counts())
df['Binary Cluster 0'] = df['KMeans'].map({1:0, 0:1, 2:1})
print("\n", df["Binary Cluster 0"].value_counts())

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1)
clf.fit(df2.iloc[:, 4:109].values, df["Binary Cluster 0"].values)

# Index sort the most important features
sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1] # Reverse sort

# Get the most important features names and weights
most_important_features = np.take_along_axis(
    np.array(df2.iloc[:, 4:109].columns.tolist()), 
    sorted_feature_weight_idxes, axis=0)
most_important_weights = np.take_along_axis(
    np.array(clf.feature_importances_), 
    sorted_feature_weight_idxes, axis=0)

# Show
#print(list(zip(most_important_features, most_important_weights)))
df3 = pd.DataFrame({"features":list(most_important_features), "weights":list(most_important_weights)})
#print(df3.iloc[:20, :])

shape_value = 0
firstorder_value = 0
texture_value = 0

print(type(df3["weights"].values))
for i, (f, w) in enumerate(zip(df3["features"].to_list(), df3["weights"].to_list())):
    if "shape" in f.split("_"):
        shape_value += w
    elif "firstorder" in f.split("_"):
        firstorder_value += w
    else:
        texture_value += w

print("shape_value:{}".format(shape_value))
print("firstorder_value:{}".format(firstorder_value))
print("texture_value:{}".format(texture_value))

data= [shape_value, firstorder_value, texture_value]
labels = ["shape", "firstorder", "texture"]
plt.pie(data, labels = labels, autopct = "%1.0f%%")
plt.title("Distribution of feature importance")
plt.show()