import pandas as pd

### 1.2) Exploratory Data Analysis and Data Integrity

def data_analysis(df: pd.DataFrame) -> None:

    print("===== DATASET OVERVIEW =====")
    print(df.shape)
    print(df.head())

    print("\n===== BASIC STATS =====")
    print("Syndromes:", df["syndrome_id"].nunique())
    print("Subjects:", df["subject_id"].nunique())

    print("\n===== IMAGES PER SUBJECT =====")
    print(df.groupby("subject_id").size().describe())

    print("\n===== CLASS DISTRIBUTION =====")
    print(df["syndrome_id"].value_counts())

    print("\n===== DATA INTEGRITY =====")
    print("Missing:\n", df.isnull().sum())
    print("Embedding sizes:\n", df["embeddings"].apply(len).value_counts())
    print("Duplicates:\n", df[["syndrome_id", "subject_id", "image_id"]].duplicated().sum())

    print("\n===== SUBJECT-SYNDROME RELATION =====")
    print(df.groupby("subject_id")["syndrome_id"].nunique().value_counts())