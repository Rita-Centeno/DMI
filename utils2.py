import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors
from sklearn.base import clone
from utils1 import set_plot_properties

# KMEANS -------------------------------------------------------------------------------------------------------------
def plot_inertia_and_silhouette(data, k_min=2, k_max=15):
    """
    Plot the inertia (dispersion) and silhouette score for different numbers of clusters.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input data for clustering.
        k_min (int, optional): The minimum number of clusters to evaluate. Defaults to 2.
        k_max (int, optional): The maximum number of clusters to evaluate. Defaults to 15.

    Returns:
        None
    """
    dispersions = []
    scores = []

    k_clusters = range(k_min, k_max + 1)

    for k in k_clusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersions.append(kmeans.inertia_)  # Calculate the dispersion (inertia) for each number of clusters
        kmeans.predict(data)
        scores.append(silhouette_score(data, kmeans.labels_, metric='euclidean'))  # Calculate the silhouette score

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(k_clusters, dispersions, marker='o')  # Plot the inertia (dispersion)
    set_plot_properties(ax1, 'Number of clusters', 'Dispersion (inertia)')
    ax2.plot(k_clusters, scores, marker='o')  # Plot the silhouette score
    set_plot_properties(ax2, 'Number of clusters', 'Silhouette score')

    plt.show()


# HIERARCHICAL -------------------------------------------------------------------------------------------------------
def plot_dendrogram(data, linkage_method, columns=None, cut_line=None):
    """
    Plot a dendrogram for hierarchical clustering.
    """
    # Select the specified columns
    if columns is None:
        columns = data.columns
    
    data_filtered = data[columns]
    
    # Fit the AgglomerativeClustering model
    model = AgglomerativeClustering(linkage=linkage_method, distance_threshold=0, n_clusters=None).fit(data_filtered)

    # Create the plot
    fig, ax = plt.subplots()
    plt.title('Hierarchical Clustering Dendrogram')

    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create the linkage matrix
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the dendrogram
    dendrogram(linkage_matrix, truncate_mode='level', p=20)#p=50)

    # Add a cut line if provided
    if cut_line is not None:
        plt.axhline(y=cut_line, color='black', linestyle='-')

    # Display the plot
    plt.show()


# DBSCAN -------------------------------------------------------------------------------------------------------------~
def plot_kdist_graph(df, feats, n_neighbors=150):
  ''' K-distance graph to find out the right eps value. For each data point, 
  calculates the average distance to its n_neighbors'''

  neigh = NearestNeighbors(n_neighbors=n_neighbors)
  neigh.fit(df[feats])
  distances, _ = neigh.kneighbors(df[feats])

  ## We sort the average distances of the points and plot
  distances = np.sort(distances[:, -1])
  plt.ylabel("%d-NN Distance" % n_neighbors)
  plt.xlabel("Points sorted by distance")
  plt.plot(distances)
  plt.show()


# CLUSTER ANALYSIS ---------------------------------------------------------------------------------------
def clusters_comparison(data, solution1, solution2):
    """
    Compare the clusters of two solutions using a confusion matrix.

    Args:
        data (pandas.DataFrame): The input data containing the cluster assignments for both solutions.
        solution1 (str): The column name representing the cluster assignments of the first solution.
        solution2 (str): The column name representing the cluster assignments of the second solution.

    Returns:
        pandas.DataFrame: The confusion matrix comparing the clusters of the two solutions.
    """
    # Determine the number of unique clusters in each solution
    length1, length2 = len(data[solution1].unique()), len(data[solution2].unique())

    # Determine the maximum number of clusters
    n = max(length1, length2)

    # Compute the confusion matrix
    confusion = confusion_matrix(data[solution1], data[solution2])

    # Create a DataFrame for the confusion matrix with appropriate row and column labels
    df = pd.DataFrame(
        confusion,
        index=['{} {} Cluster'.format(solution1, i) for i in np.arange(0, n)],
        columns=['{} {} Cluster'.format(solution2, i) for i in np.arange(0, n)]
    )

    # Return the subset of the confusion matrix corresponding to the number of clusters in each solution
    return df.iloc[:length1, :length2]


def groupby_mean(data, variable, gradient=False, n_features=13, ax=0):
    """
    Group the data by a variable and calculate the mean for each group.

    Args:
        data (pandas.DataFrame): The input data.
        variable (str): The variable used for grouping.

    Returns:
        pandas.DataFrame: The transposed DataFrame containing the mean values for each group,
                          with an additional column for the overall mean and the count of observations.
    """
    # Group the data by the specified variable and calculate the mean for each group
    grouped_data = data.groupby(variable).mean().round(2)

    # Transpose the grouped data
    transposed_data = grouped_data.T

    # Select the first n_features rows
    result = transposed_data.iloc[:n_features, :]

    if not gradient:
        # Calculate the overall mean for all numeric columns
        overall_mean = data.mean(numeric_only=True).round(2)

        # Add the overall mean as a new column
        result["data"] = overall_mean

        # Calculate the count of observations in each group and overall
        counts = data.groupby(variable).size()
        counts["data"] = data.shape[0]  # Add the total number of observations

        # Add the counts as the last row in the result
        result.loc["Counts"] = counts
    
    else:
        result = result.style.background_gradient(axis=ax)
        # counts = data.groupby(variable).size()
        # counts["data"] = data.shape[0]
        print(f'# observations per cluster: {data.groupby(variable).size().tolist()}')

    return result


def compare_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """
    Creates a DataFrame with the mean value of each column for different 
    clusters and for all observations.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the data with the 
    designated clusters.
    cluster_col (str): The name of the column containing the cluster information.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the mean values of each column, 
    grouped by the cluster column and the general mean for each variable.

    """
    #Compute the mean of the variable without segmentation
    general_mean = pd.DataFrame(df.mean().T).rename(columns={0:'general_mean'})
    #Find mean values of the variable per cluster
    clusters_mean = pd.DataFrame(df.groupby(cluster_col).mean().T)
    
    return clusters_mean.join(general_mean)


# DIMENSIONALITY REDUCTION ---------------------------------------------------------------------------------------
def visualize_dimensionality_reduction(transformation, targets):
    """
    Visualize the dimensionality reduction results using a scatter plot.

    Args:
        transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
        targets (numpy.ndarray or list): The target labels or cluster assignments.

    Returns:
        None
    """
    # Convert object labels to categorical variables
    labels, targets_categorical = np.unique(targets, return_inverse=True)

    # Create a scatter plot of the t-SNE output
    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
    plt.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

    # Create a legend with the class labels and corresponding colors
    handles = [plt.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Clusters')

    # Display the plot
    plt.show()


# COMPARE LINKEAGE METHODS -------------------------------------------------------------------------------------------
def get_r2_hc(df, link_method, max_nclus, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt.

    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".

    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares

    r2 = []  # where we will store the R2 metrics for each cluster solution

    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)


        hclabels = cluster.fit_predict(df) #get cluster labels


        df_concat = pd.concat((df, pd.Series(hclabels, name='labels', index=df.index)), axis=1)  # concat df with labels


        ssw_labels = df_concat.groupby(by='labels').apply(get_ss)  # compute ssw for each cluster labels


        ssb = sst - np.sum(ssw_labels)  # remember: SST = SSW + SSB


        r2.append(ssb / sst)  # save the R2 of the given cluster solution

    return np.array(r2)


def plot_r2_linkage(df, max_nclus):
    # Prepare input
    hc_methods = ["ward", "complete", "average", "single"]
    # Call function defined above to obtain the R2 statistic for each hc_method
    max_nclus = 10
    r2_hc_methods = np.vstack(
        [
            get_r2_hc(df=df, link_method=link, max_nclus=max_nclus)
            for link in hc_methods
        ]
    ).T
    r2_hc_methods = pd.DataFrame(r2_hc_methods, index=range(1, max_nclus + 1), columns=hc_methods)

    sns.set()
    # Plot data
    fig = plt.figure(figsize=(11,5))
    sns.lineplot(data=r2_hc_methods, linewidth=2.5, markers=["o"]*4)

    # Finalize the plot
    fig.suptitle("R2 plot for various hierarchical methods", fontsize=21)
    plt.gca().invert_xaxis()  # invert x axis
    plt.legend(title="HC methods", title_fontsize=11)
    plt.xticks(range(1, max_nclus + 1))
    plt.xlabel("Number of clusters", fontsize=13)
    plt.ylabel("R2 metric", fontsize=13)

    plt.show()


# COMPARE CLUSTERING TECHNIQUES ---------------------------------------------------------------------------------------

def get_ss(df):
    """Computes the sum of squares for all variables given a dataset
    """
    ss = np.sum(df.var() * (df.count() - 1))
    return ss  # return sum of sum of squares of each df variable

def r2(df, labels):
    sst = get_ss(df)
    ssw = np.sum(df.groupby(labels).apply(get_ss))
    return 1 - ssw/sst

def get_r2_scores(df, clusterer, min_k=2, max_k=10):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        r2_clust[n] = r2(df, labels)
    return r2_clust


def get_r2_df(df, feats, kmeans_model, hierar_model):
  # Obtaining the R² scores for each cluster solution

  r2_scores = {}
  r2_scores['kmeans'] = get_r2_scores(df[feats], kmeans_model)

  for linkage in ['complete', 'average', 'single', 'ward']:
      r2_scores[linkage] = get_r2_scores(
          df[feats], hierar_model.set_params(linkage=linkage)
      )

  return pd.DataFrame(r2_scores)


def plot_r2_scores(r2_scores,
                   plot_title="Preference Variables:\nR² plot for various clustering methods\n",
                   legend_title="Cluster methods"):
  # Visualizing the R² scores for each cluster solution on demographic variables
  pd.DataFrame(r2_scores).plot.line(figsize=(10,7))

  plt.title(plot_title, fontsize=21)
  plt.legend(title=legend_title, title_fontsize=11)
  plt.xlabel("Number of clusters", fontsize=13)
  plt.ylabel("R² metric", fontsize=13)
  plt.show()


# MERGING PERSPECTIVES ---------------------------------------------------------------------------------------

def hc_merge_mapper(df, label1, label2, feats, merged_label, n_clusters=1):
  """
  merged_label  : what to call the column that should contain the merged label
  n_clusters    : how many clusters to keep
  """
  df_ = df.copy()

  # Centroids of the concatenated cluster labels
  df_centroids = df_.groupby([label1, label2])[feats].mean()

  # Running the Hierarchical clustering based on the correct number of clusters
  hclust = AgglomerativeClustering(
      linkage='ward',
      metric='euclidean',
      n_clusters=n_clusters
  )
  hclust_labels = hclust.fit_predict(df_centroids)
  df_centroids[merged_label] = hclust_labels

  cluster_mapper = df_centroids[merged_label].to_dict()

  # Mapping the clusters on the centroids to the observations
  df_[merged_label] = df_.apply(
      lambda row: cluster_mapper[
          (row[label1], row[label2])
      ], axis=1
  )

  return df_, df_centroids


def make_contingency_table(df, label1, label2):
  df_ = df.groupby([label1, label2])\
            .size()\
            .to_frame()\
            .reset_index()\
            .pivot(index=label2, columns=label1)
  df_.columns = df_.columns.droplevel()
  return df_



# PROFILING -------------------------------------------------------------------------------------------------------

def cluster_profiles(df,
                     label_columns,
                     figsize,
                     compar_titles=None,
                     colors='Set1'):
    """
    Pass df with labels columns of one or multiple clustering labels.
    Then specify this label columns to perform the cluster profile according to them.
    """
    if compar_titles == None:
        compar_titles = [""]*len(label_columns)

    fig, axes = plt.subplots(nrows=len(label_columns), ncols=2,
                             figsize=figsize, squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compar_titles):

        # Filtering df
        drop_cols = [i for i in label_columns if i!=label]
        dfax = df.drop(drop_cols, axis=1)

        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
        counts.columns = [label, "counts"]

        # Setting Data
        pd.plotting.parallel_coordinates(centroids, label,
                                         color=sns.color_palette(palette=colors), ax=ax[0])
        sns.barplot(x=label, y="counts", data=counts, ax=ax[1],
                    palette=sns.color_palette(palette=colors))

        #Setting Layout
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
        ax[0].annotate(text=titl, xy=(0.95,1.1), xycoords='axes fraction', fontsize=13, fontweight = 'heavy')
        ax[0].legend(handles, cluster_labels) # Adaptable to number of clusters
        ax[0].axhline(color="black", linestyle="--")
        ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=13)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-90)
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[1].set_xticklabels(cluster_labels)
        ax[1].set_xlabel("")
        ax[1].set_ylabel("Absolute Frequency")
        ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=13)

    plt.subplots_adjust(hspace=0.4, top=0.90)
    plt.suptitle("Cluster Simple Profilling", fontsize=23)
    plt.show()


def cluster_heatmaps(df,
                     label_columns,
                     figsize=(20,20),
                     compar_titles=None,
                     heat_colors='RdYlBu',
                     bar_colors='Set2'):
    """
    Pass df with labels columns of one or multiple clustering labels.
    Then specify this label columns to perform the cluster profile according to them.
    """
    if compar_titles == None:
        compar_titles = [""]*len(label_columns)

    fig, axes = plt.subplots(nrows=len(label_columns), ncols=2,
                             figsize=figsize, squeeze=False)
    for ax, label, titl in zip(axes, label_columns, compar_titles):

        # Filtering df
        drop_cols = [i for i in label_columns if i!=label]
        dfax = df.drop(drop_cols, axis=1)

        # Getting the cluster centroids and counts
        centroids = dfax.groupby(by=label, as_index=False).mean()
        counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
        counts.columns = [label, "counts"]


        # Setting Data
        handles, _ = ax[0].get_legend_handles_labels()
        cluster_labels = ["Cluster {}".format(i) for i in range(counts.shape[0])]

        sns.heatmap(centroids.drop(columns=label),
              square=False, cmap=heat_colors,
              ax=ax[0],
              )

        ax[0].set_title("Cluster Means Heatmap - {} Clusters".format(counts.shape[0]), fontsize=18)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-20)
        ax[0].set_yticklabels(cluster_labels, rotation=0)
        ax[1].annotate(text=titl, xy=(-0.3,1.15),
                       xycoords='axes fraction',
                       fontsize=18, fontweight = 'heavy')


        sns.barplot(y=label, x="counts", data=counts, ax=ax[1], orient='h', palette=bar_colors)
        ax[1].set_yticklabels(cluster_labels)
        ax[1].set_title("Cluster Sizes - {} Clusters".format(counts.shape[0]), fontsize=18)
        ax[1].set_ylabel("")

    plt.subplots_adjust(hspace=0.4, top=0.90)
    plt.suptitle("Cluster Simple Profilling", fontsize=23)
    plt.show()