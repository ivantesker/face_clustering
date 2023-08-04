# Face Clustering

## Description
This project aims to perform face clustering based on the given dataset of reference clusters. Each cluster contains several photographs of the same individual.

## Dataset
The dataset can be downloaded from the following link:
[Face Clustering Dataset](https://disk.yandex.ru/d/X7Dh7hrgF90k4g)

## Tasks
1. Cluster individuals based on their faces.
2. Evaluate the quality of clustering using metrics that satisfy the properties of homogeneity and completeness.

## Requirements
- Use an object-oriented approach for implementation.
- Apply the principles of encapsulation and dependency inversion.
- Implement a service layer pattern.

## Solution
- A baseline solution has been implemented.
- The `facenet_pytorch` library is used for face detection.
- `DBSCAN` is utilized for clustering, with dimensionality reduction using `PCA`.
- Silhouette Score: 0.31514140487799364
- Number of unique faces: 25
