# Week 7 Deep Learning Tutorial - Data Dimension Reduction
# Code to implement PCA (Principal Component Analysis)

# Importing the libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Importing the dataset
def import_dataset():
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Load dataset into Pandas Data Frame
    df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
    # Print the first 5 rows of the data frame.
    # print(df.head())
    return df


# Standardize the data
def standardize_data(df):
    # Separating out the features
    variables = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Separating out the target
    x = df.loc[:, variables].values
    y = df.loc[:,['target']].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # Convert to a data frame
    x= pd.DataFrame(x)

    # Print the first 5 rows of the data frame.
     # print(x.head())

    return x, y


# PCA Projection to 2D
def pca(X, y):
    pca = PCA()
    # Fit the data
    X_pca = pca.fit_transform(X)
    
    # Convert to a data frame
    X_pca = pd.DataFrame(X_pca)
    print(X_pca.head())

    # Add the target
    X_pca['target']=y
    X_pca.columns = ['PC1','PC2','PC3','PC4','target']
    X_pca.head()


    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_
    # Print the explained variance
    print(explained_variance)
    # The first principal component contains 72.77% of the variance and the second principal component contains 23.03% of the variance.
    # Together, the two components contain 95.80% of the information.
    # These can be used to reduce the dimensionality of the data from 4 to 2.
    # [0.72770452 0.23030523 0.03683832 0.00515193]

    return X_pca, explained_variance


# Plot the explained variance
def plot_PCA_variance(explained_variance):

    plt.figure(figsize=(8, 6))
    
    plt.bar(range(len(explained_variance )), explained_variance )
    
    plt.ylabel('Variance ratio')
    
    plt.xlabel('Principal components')
    
    plt.show()


# Plot the PCA 2D
def plot_PCA_2D(X_pca, explained_variance):
    fig = plt.figure()
    # Create a figure and a 3D Axes
    ax = fig.add_subplot(1,1,1) 
    
    ax.set_xlabel('Principal Component 1') 
    
    ax.set_ylabel('Principal Component 2') 
    
    ax.set_title('2 component PCA') 
    
    targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']

    # Plot the data
    for target, color in zip(targets,colors):
        indicesToKeep = X_pca['target'] == target
        ax.scatter(X_pca.loc[indicesToKeep, 'PC1']
        , X_pca.loc[indicesToKeep, 'PC2']
        , c = color
        , s = 50)
    
    # Add the legend
    ax.legend(targets)
    # Add the grid
    ax.grid()

    # Show the plot
    plt.show()



def main():
    # Import the dataset
    df = import_dataset()

    # Standardize the data
    x, y = standardize_data(df)

    # PCA Projection to 2D
    X_pca, explained_variance = pca(x, y)

    # Plot the explained variance
    plot_PCA_variance(explained_variance)

    # Plot the PCA 2D
    plot_PCA_2D(X_pca, explained_variance)



if __name__ == "__main__":
    main()