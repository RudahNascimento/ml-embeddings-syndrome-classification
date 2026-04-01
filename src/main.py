from data_processing import data_load, data_flatten
from data_visualization import data_preparation, tsne_visualization, roc_curves_visualization
from data_analysis import data_analysis
from tables import create_k_results_table, create_final_metrics_table, save_table_as_image
from knn import optimal_k_value, evaluate_model

def main():

    # 1) Load of the pickle file

    data = data_load("data/mini_gm_public_v0.1.p")

    # 2) Transform the file into a DataFrame (flattened_data is on 'tables' folder)

    df = data_flatten(data)

    df.to_csv("results/tables/flattened_data.csv", index=False)

    # 3) Exploratory Data Analysis and Data Integrity Verification (Outputs are on 'prints' folder)

#     data_analysis(df)

    # 4) Data Processing for Visualization and KNN

    X, y, le = data_preparation(df)

    # 5) Visualization of the Data in 2D Using t-SNE (tsne is on 'plots' folder)

    tsne_visualization(X, y, "results/plots/tsne.png", show=False)

    # 6) Finding the optimal k and distance metric (k_results is on 'tables' folder)

    k_values, results_k = optimal_k_value(X, y)
    k_table = create_k_results_table(k_values, results_k)
    save_table_as_image(k_table, "results/tables/k_results.png")
    print(k_table)

    k_optimal = 7

    # 7) Implementing KNN, Evaluating Model and Plotting ROC Curves (final_metrics is on 'tables' folder, ROC Curves plot is on 'plots' folder)

    results, mean_fpr, tprs_euclidean, tprs_cosine, aucs_euclidean, aucs_cosine = evaluate_model(X, y, k_optimal)

    final_table = create_final_metrics_table(results)
    save_table_as_image(final_table, "results/tables/final_metrics.png")
    print(final_table)

    roc_curves_visualization(mean_fpr,tprs_euclidean,tprs_cosine,aucs_euclidean,aucs_cosine,"results/plots/roc_curve.png", show=False)

if __name__ == "__main__":
    main()