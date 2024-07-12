### Project4: MovieLens Reviews Clustering

#### Objective

The goal of Project4 is to cluster movies based on user ratings from the MovieLens dataset to uncover interesting patterns and generate insights. The project explores different clustering algorithms and their effectiveness in grouping movies according to user preferences and behaviors.

#### Project Structure

- **Folder `project4`**: Contains the main notebook and supporting files for the project.
  - **Notebook `clustering.ipynb`**: This notebook encompasses the entire clustering project and includes several key sections:
    - **Exploratory Data Analysis**: Initial exploration of the dataset to understand the distribution of ratings and movie genres.
    - **Data Preprocessing**: Cleaning and preparation of data for clustering, including normalization and handling missing values.
    - **Clustering Algorithms Evaluation**: Implementation and comparison of three clustering algorithms:
      - **DBSCAN**: Used for its density-based clustering capabilities, ideal for finding arbitrarily shaped clusters and handling noise.
      - **Agglomerative Clustering**: A hierarchical clustering method that builds a tree of clusters and is useful for understanding the hierarchical structure of data.
      - **KMeans**: A centroid-based algorithm, popular for its simplicity and efficiency in forming spherical clusters.
    - **Dimensionality Reduction**: Evaluation of clustering performance with and without dimensionality reduction techniques such as PCA (Principal Component Analysis) and SVD.
    - **Cluster Analysis**:  This section includes a discussion on the insights gained from the clustering results, such as common preferences among user groups or unique movie genres that tend to cluster together.

