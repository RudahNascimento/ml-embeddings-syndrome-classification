# ML Test - Syndrome Classification

This project implements a machine learning pipeline to classify syndromes based on embedding vectors, using K-Nearest Neighbors (KNN).

## 1) Project Structure

```
project/  
│
├── data/
│   └── mini_gm_public_v0.1.p
│
├── src/
│   ├── data_analysis.py
│   ├── data_processing.py
│   ├── data_visualization.py
│   ├── knn.py
│   ├── main.py
│   └── tables.py
│
├── results/
│   ├── plots/
│   ├── prints/
│   └── tables/
│
├── requirements.txt
└── README.md 
```

## 2) How to Run

0. Python version: 3.11

1. Install dependencies:
py -m pip install -r requirements.txt

2. Run the project:
py src/main.py

## 3) Pipeline Overview

- Load and flatten data from pickle file (generate flattened_data.csv)
- Perform data integrity checks (prints on 'prints' folder)
- Visualize embeddings using t-SNE (generate tsne.png)
- Find the optimal k and distance metric (generate k_results.png)
- Train and evaluate KNN models (generate final_metrics.png)
- Compare Euclidean and Cosine distance metrics (generate roc_curve.png)

Observations:

- The plots and tables outputs are automatically generated in the '/results' folder when running the main script.
- The data_analysis() function is commented out in main(). The outputs it generates are available in the 'prints' folder.
- The 'show' parameter in the visualization functions is set to False. To display the plots, it should be changed to True.

## 4) Outputs

The following outputs are generated automatically:

- Plots:
  - t-SNE visualization
  - ROC curve comparison

- Tables:
  - K selection results
  - Final evaluation metrics
  - Flattened dataset (CSV)

## 5) Report

A detailed analysis of the methodology, results, and conclusions is provided in the accompanying PDF report.