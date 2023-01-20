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
import itertools as iter

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

"""
creating function to plot scatter graph
"""
def plot_scatterGraph(x,y,title):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Power consumption per capita')
    plt.show()

"""
creating function to built clustering graph for pakistan
"""
def plot_clusteredGraphPak(data_fit):
    for ic in range(2, 7):
    # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(data_fit)
    # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(data_fit, labels))

    # Plot for four clusters using cluster library
    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(data_fit)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))

    # Individual colours can be assigned to symbols. The label l is used to the select the
    # l-th number from the colour table.
    plt.scatter(data_fit["Year"], data_fit["Pakistan"], c=labels, cmap="Accent")

    # showing the center of the cluster
    for ic in range(4):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)

    plt.xlabel("Year")
    plt.ylabel("Power consumption per capita")
    plt.title("Clustering graph for Pakistan")
    plt.show()

"""
creating function to built clustering graph for India
"""
def plot_clusteredGraphInd(data_fitIn):
    for ic in range(2, 7):
    # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(data_fitIn)
    # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(data_fitIn, labels))

    # Plot for four clusters using cluster library
    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(data_fitIn)

    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))

    # Individual colours can be assigned to symbols. The label l is used to the select the
    # l-th number from the colour table.
    plt.scatter(data_fitIn["Year"], data_fitIn["India"], c=labels, cmap="Accent")

    # showing the center of the cluster
    for ic in range(4):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)

    plt.xlabel("Year")
    plt.ylabel("Power consumption per capita")
    plt.title("Clustering graph for India")
    plt.show()

#function to fit the graph
def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

"""
defining function to plot curve fit for pakistan
"""
def plot_curvefitPak(transposedpoCon_df, data_fit):
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

    """
    finding the upper and lower limit
    """

    ci = 1.96 * np.std(data_fit["Pakistan"])/np.sqrt(len(data_fit["Year"]))
    #fig, ax = plt.subplots()
    plt.plot(data_fit["Year"],data_fit["Pakistan"])
    plt.fill_between(data_fit["Year"], (data_fit["Pakistan"]-ci), (data_fit["Pakistan"]+ci), color='b', alpha=.1)

    """
    plotting curve fit
    """
    plt.plot(x_line, y_line, '--', color='r')
    plt.xlabel("Year")
    plt.ylabel("Power consumption per capita")
    plt.title("Curve fit of pakistan")
    plt.legend(['Actual', 'Predicted values'])
    plt.show()

"""
defining function to plot curve fit for India
"""
def plot_curvefitInd(transposedpoCon_df, data_fitIn):
    x , y = transposedpoCon_df['Year'], transposedpoCon_df['India']
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

    """
    finding the upper and lower limit
    """

    ci = 1.96 * np.std(data_fitIn["India"])/np.sqrt(len(data_fitIn["Year"]))
    #fig, ax = plt.subplots()
    plt.plot(data_fitIn["Year"],data_fitIn["India"])
    plt.fill_between(data_fitIn["Year"], (data_fitIn["India"]-ci), (data_fitIn["India"]+ci), color='b', alpha=.1)

    """
    plotting curve fit
    """
    plt.plot(x_line, y_line, '--', color='r')
    plt.xlabel("Year")
    plt.ylabel("Power consumption per capita")
    plt.title("Curve fit of India")
    plt.legend(['Actual', 'Predicted values'])
    plt.show()

#calling the fucting to read the file
powerConsume_Df, transposedpoCon_df = readFile("pcMain.csv")

#defining the transposed dataframe to built clustered graph
data_fit = transposedpoCon_df[["Year", "Pakistan"]].copy()
data_fitIn = transposedpoCon_df[["Year", "India"]].copy()

"""
from taking the data for pakistan and india we are
plotting the scatter graph for data
"""
plot_scatterGraph(transposedpoCon_df['Year'], transposedpoCon_df['Pakistan'], "Scatter graph for Pakistan")
plot_scatterGraph(transposedpoCon_df['Year'], transposedpoCon_df['India'], "Scatter graph for India")

"""
plotting the clustered graph for india and pakistan
"""
plot_clusteredGraphPak(data_fit)
plot_clusteredGraphInd(data_fitIn)

"""
plotting the curve fit graph for india pakistan
"""
plot_curvefitPak(transposedpoCon_df, data_fit)
plot_curvefitInd(transposedpoCon_df, data_fitIn)
