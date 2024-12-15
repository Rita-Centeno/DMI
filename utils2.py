import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.cluster.hierarchy import dendrogram
from utils1 import set_plot_properties
from umap import UMAP


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


def plot_dendrogram(data, linkage_method, cut_line=None):
    """
    Plot a dendrogram for hierarchical clustering.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input data for clustering.
        linkage_method (str): The linkage method used for clustering.
        cut_line (float, optional): The threshold value to cut the dendrogram. Defaults to None.

    Returns:
        None
    """
    # Fit the AgglomerativeClustering model
    model = AgglomerativeClustering(linkage=linkage_method, distance_threshold=0, n_clusters=None).fit(data)

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


def groupby_mean(data, variable, n_features=13):
    """
    Group the data by a variable and calculate the mean for each group.

    Args:
        data (pandas.DataFrame): The input data.
        variable (str): The variable used for grouping.

    Returns:
        pandas.DataFrame: The transposed DataFrame containing the mean values for each group,
                          with an additional column for the overall mean.
    """
    # Group the data by the specified variable and calculate the mean for each group
    grouped_data = data.groupby(variable).mean().round(2)

    # Calculate the overall mean for all numeric columns
    overall_mean = data.mean(numeric_only=True).round(2)

    # Transpose the grouped data
    transposed_data = grouped_data.T

    # Add the overall mean as a new column
    transposed_data["data"] = overall_mean

    # Select the first n_features + 1 columns (including the variable column) and transpose the DataFrame
    result = transposed_data.iloc[:n_features, :]

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