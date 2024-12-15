import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def format_column_name(column, exclude_words=None):
    if exclude_words is None:
        exclude_words = []  # Default to no excluded words
    exclude_words = set(word.lower() for word in exclude_words)  # Normalize for case-insensitivity
    
    # Split the column into words based on '_'
    words = column.split('_')
    # Capitalize all words except those in the exclusion list
    formatted_words = [word.capitalize() if word.lower() not in exclude_words else word.lower() for word in words]
    # Rejoin the words with '_'
    return '_'.join(formatted_words)


def set_plot_properties(ax, x_label, y_label, y_lim=[]):
    """
    Set properties of a plot axis.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].

    Returns:
        None
    """
    ax.set_xlabel(x_label)  # Set the label for the x-axis
    ax.set_ylabel(y_label)  # Set the label for the y-axis
    if len(y_lim) != 0:
        ax.set_ylim(y_lim)  # Set the limits for the y-axis if provided


def plot_pie_chart(data, variable, colors, labels=None, legend=[], autopct='%1.1f%%'):
    """
    Plot a pie chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        colors (list): The colors for each pie slice.
        labels (list, optional): The labels for each pie slice. Defaults to None.
        legend (list, optional): The legend labels. Defaults to [].
        autopct (str, optional): The format for autopct labels. Defaults to '%1.1f%%'.

    Returns:
        None
    """
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable

    # Plot the pie chart with specified parameters
    plt.pie(counts, colors=colors, labels=labels, startangle=90, autopct=autopct, textprops={'fontsize': 25})
    
    if len(legend) != 0:
        plt.legend(legend, fontsize=16, bbox_to_anchor=(0.7, 0.9))  # Add a legend if provided
    
    plt.show()  # Display the pie chart


def plot_bar_chart(ax, data, variable, x_label, y_label='Count', y_lim=[], legend=[], color='cadetblue', annotate=False):
    """
    Plot a bar chart based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        y_lim (list, optional): The limits for the y-axis. Defaults to [].
        legend (list, optional): The legend labels. Defaults to [].
        color (str, optional): The color of the bars. Defaults to 'cadetblue'.
        annotate (bool, optional): Flag to annotate the bars with their values. Defaults to False.

    Returns:
        None
    """
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable
    x = counts.index
    y = counts.values

    ax.bar(x, y, color=color)  # Plot the bar chart with specified color
    ax.set_xticks(x)  # Set the x-axis tick positions
    if len(legend) != 0:
        ax.set_xticklabels(legend)  # Set the x-axis tick labels if provided

    if annotate:
        for i, v in enumerate(y):
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=12)  # Annotate the bars with their values

    set_plot_properties(ax, x_label, y_label, y_lim)  # Set plot properties using helper function


def plot_line_chart(ax, data, variable, x_label, y_label='Count', color='cadetblue', fill=False):
    """
    Plot a line chart based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the line. Defaults to 'cadetblue'.
        fill (bool, optional): Flag to fill the area under the line. Defaults to False.

    Returns:
        None
    """
    counts = data[variable].value_counts()  # Count the occurrences of each value in the variable
    counts_sorted = counts.sort_index()  # Sort the counts by index
    x = counts_sorted.index
    y = counts_sorted.values

    ax.plot(x, y, marker='o', color=color)  # Plot the line chart with specified color and marker
    if fill:
        ax.fill_between(x, y, color='cadetblue', alpha=0.25)  # Fill the area under the line if fill is True

    set_plot_properties(ax, x_label, y_label)  # Set plot properties using helper function


def plot_histogram(ax, data, variable, x_label, y_label='Count', color='cadetblue'):
    """
    Plot a histogram based on the values of a variable in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        x_label (str): The label for the x-axis.
        y_label (str, optional): The label for the y-axis. Defaults to 'Count'.
        color (str, optional): The color of the histogram bars. Defaults to 'cadetblue'.

    Returns:
        None
    """
    ax.hist(data[variable], bins=20, color=color)  # Plot the histogram using 50 bins

    set_plot_properties(ax, x_label, y_label)  # Set plot properties using helper function


def plot_correlation_matrix(data, method):
    """
    Plot a correlation matrix heatmap based on the given data.

    Args:
        data (pandas.DataFrame): The input data for calculating correlations.
        method (str): The correlation method to use.

    Returns:
        None
    """
    corr = data.corr(method=method)  # Calculate the correlation matrix using the specified method

    mask = np.tri(*corr.shape, k=0, dtype=bool)  # Create a mask to hide the upper triangle of the matrix
    corr.where(mask, np.nan, inplace=True)  # Set the upper triangle values to NaN

    plt.figure(figsize=(30, 15))  # Adjust the width and height of the heatmap as desired

    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                annot=True,
                vmin=-1, vmax=1,
                cmap=sns.diverging_palette(220, 10, n=20))  # Plot the correlation matrix heatmap


def plot_scatter(ax, data, variable1, variable2, color='cadetblue'):
    """
    Plot a scatter plot between two variables in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variables.
        variable1 (str): The name of the first variable.
        variable2 (str): The name of the second variable.
        color (str, optional): The color of the scatter plot. Defaults to 'cadetblue'.

    Returns:
        None
    """
    ax.scatter(data[variable1], data[variable2], color=color)  # Plot the scatter plot

    set_plot_properties(ax, variable1, variable2)  # Set plot properties using helper function


def plot_map(data, column, hover_col, zoom = 1, invert = False):
    px.set_mapbox_access_token('pk.eyJ1IjoiYXJjYWRldGUyMSIsImEiOiJjbGY5cXlkY3oxcnp1NDBvNHNyM3MwZm9mIn0.sN_CBzeTj04J0BRjr3DJyw')

    if not invert:
        latitude = 'latitude'
        longitude = 'longitude'
    else:
        latitude = 'longitude'
        longitude = 'latitude'

    # Create a scatter mapbox plot using the sorted DataFrame
    fig = px.scatter_mapbox(data, 
                            lat=latitude, 
                            lon=longitude, 
                            hover_name=hover_col, 
                            color=column,
                            zoom=zoom, 
                            width=400,
                            # opacity=0.1
                            )

    # Customize the layout of the figure
    fig.update_layout(mapbox_style='dark', 
                    margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                    legend_title_text='ZIP Code', 
                    legend_title_font_color='black',
                    legend_orientation='h', 
                    legend_y=0
                    )

    # Display the figure
    fig.show()


def knn_imputer_best_k(data, k_min, k_max, weights='distance'):
    """
    Find the best value of K for KNN imputation by evaluating the average silhouette score.

    Args:
        data (pandas.DataFrame): The input data for KNN imputation and clustering.
        k_min (int): The minimum value of K to evaluate.
        k_max (int): The maximum value of K to evaluate.
        weights (str, optional): The weight function used in KNN imputation. Defaults to 'distance'.

    Returns:
        None
    """
    # Define the range of K values to evaluate
    k_values = range(k_min, k_max + 1)

    # Initialize an empty list to store the average silhouette scores
    avg_silhouette_scores = []

    # Iterate over each K value
    for k in k_values:
        # Create the KNN imputer with the current K value
        knn_imputer = KNNImputer(n_neighbors=k, weights=weights)

        # Perform KNN imputation and clustering for each fold in cross-validation
        silhouette_scores = []
        for fold in range(5):  # Adjust the number of folds as needed
            # Split your data into training and test sets
            # Replace the following lines with your actual data splitting code
            X_train, X_test = train_test_split(data, train_size=0.9)

            # Perform KNN imputation on the training and test sets
            X_train_imputed = knn_imputer.fit_transform(X_train)
            X_test_imputed = knn_imputer.transform(X_test)

            # Cluster the imputed data using KMeans or other clustering algorithm
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X_train_imputed)

            # Evaluate the clustering performance using silhouette score
            labels = kmeans.predict(X_test_imputed)
            silhouette = silhouette_score(X_test_imputed, labels)
            silhouette_scores.append(silhouette)

        # Calculate the average silhouette score for the current K value
        avg_silhouette_score = np.mean(silhouette_scores)
        avg_silhouette_scores.append(avg_silhouette_score)

    # Find the index of the K value with the highest average silhouette score
    best_k_index = np.argmax(avg_silhouette_scores)
    best_k = k_values[best_k_index]

    print('Best K value:', best_k)


def knn_imputer(data, k=3, exclude_columns=[]):
    """Impute missing values using the KNN algorithm."""
    # Extract the indexes of the excluded columns
    exclude_indexes = {col: data.columns.get_loc(col) for col in exclude_columns}

    # Drop the excluded columns
    imputation_data = data.drop(columns=exclude_columns)

    # Perform KNN imputation on the remaining data
    imputer = KNNImputer(n_neighbors=k, weights='distance').fit(imputation_data)
    imputed_data = pd.DataFrame(imputer.transform(imputation_data), columns=imputation_data.columns, index=imputation_data.index
    )

    # Reinsert the excluded columns at their original positions
    for col, idx in exclude_indexes.items():
        imputed_data.insert(idx, col, data[col])

    return imputed_data


def detect_outliers_iqr_with_stats(df, cols_to_check=None):
    # Use numeric columns if cols_to_check is None
    if cols_to_check is None:
        cols_to_check = df.select_dtypes(include='number').columns

    stats = {}
    
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)  # First quartile (25%)
        Q3 = df[col].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Calculate outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
        percentage_outliers = 100 * outliers.sum() / len(df)
        
        # Store results
        stats[col] = {
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Percentage Outliers': percentage_outliers
        }
    
    return pd.DataFrame(stats).T


def remove_outliers_iqr(df, cols_to_check=None):
    # Use numeric columns if cols_to_check is None
    if cols_to_check is None:
        cols_to_check = df.select_dtypes(include='number').columns
    
    # Create a mask for rows without outliers
    no_outliers_mask = pd.Series(True, index=df.index)
    
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)  # First quartile (25%)
        Q3 = df[col].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1  # Interquartile range
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Update the mask
        no_outliers_mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
    
    # Filter the DataFrame and retain all columns
    df_no_outliers = df[no_outliers_mask]
    
    return df_no_outliers