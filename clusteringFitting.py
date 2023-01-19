# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:19:13 2023

@author: gandh
"""

#import useful libraries
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import numpy as np
import sklearn.metrics as skmet
from scipy.optimize import curve_fit

"""
Created function that reads a file, Update NAN values with 0 and transposed matrix
This function returns converted data to dataframe and transposed dataframe data
"""
def readFile(x):
    fileData = pd.read_csv(x);
    transposedFileData = pd.read_csv(x, header=None, index_col=0).T
    transposedFileData = transposedFileData.rename(columns={"Country Name":"Year"})
    transposedFileData = transposedFileData.fillna(0.0)
    return fileData, transposedFileData

powerConsume_Df, transposedpoCon_df = readFile("pcMain.csv")

# plotting the scatter graph for data
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(transposedpoCon_df['Year'], transposedpoCon_df['Pakistan'])
plt.title('Graph without clustering')
plt.xlabel('Year')
plt.ylabel('Power consumption per capita')
plt.show()

# extract columns for fitting
df_fit = transposedpoCon_df[["Year", "Pakistan"]].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements


for ic in range(2, 7):
# set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
# extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))

# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the select the
# l-th number from the colour table.
plt.scatter(df_fit["Year"], df_fit["Pakistan"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(4):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Year")
plt.ylabel("Power consumption per capita")
plt.title("4 clusters")
plt.show()
#-----------------------
# Plot for five clusters
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(df_fit)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the select the
# l-th number from the colour table.
plt.scatter(df_fit["Year"], df_fit["Pakistan"], c=labels, cmap="Accent")
# colour map Accent selected to increase contrast between colours
# show cluster centres
for ic in range(5):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("Year")
plt.ylabel("Power consumption per capita")
plt.title("5 clusters")
plt.show()


"""curve fit
------------


"""
def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f
x , y = transposedpoCon_df['Year'], transposedpoCon_df['Pakistan']
popt, _ = curve_fit(objective, x, y)

# summarize the parameter values
a, b, c, d, e, f = popt
# plot input vs output
plt.scatter(x, y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f)
# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')
plt.xlabel("Year")
plt.ylabel("Power consumption per capita")
plt.title("Curve fit")
plt.show()
