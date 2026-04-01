import pandas as pd
import pickle

# 1.1) Data Processing

def data_load(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)

def data_flatten(data: dict) -> pd.DataFrame:
    rows = []

    for syndrome_id, subject in data.items():
        for subject_id, images in subject.items():
            for image_id, embeddings in images.items():
                rows.append({"syndrome_id": syndrome_id, "subject_id": subject_id, "image_id": image_id, "embeddings": embeddings})

    return pd.DataFrame(rows)