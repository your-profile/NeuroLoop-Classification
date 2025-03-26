import numpy as np
import pandas as pd
import os

def from_tensor(x):
    try:
        x = x[0].detach().cpu().numpy()
        return softmax(x)
    except:
        return x
    
def read_files(participant_list, source_folder_1, conditions):
    participant_data = {}
    # Iterate over participants and files
    for filename in os.listdir(source_folder_1):
        for participant in participant_list:
            # print(participant)
            if filename.startswith('0{}'.format(participant)):
                if filename[4:6] in conditions:
                    # Get full file path
                    # print(participant)
                    demo_path = os.path.join(source_folder_1, filename)
                    df = pd.read_csv(demo_path, index_col=0)
                    participant_data[participant] = df
    return participant_data

def apply_features(combined_df, channels):
    for channel in channels:
        combined_df[channel] = combined_df[channel].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        max_length = max(combined_df[channel].apply(len))

        feature_columns = pd.DataFrame(combined_df[channel].tolist(), columns=[f'{channel}_{i}' for i in range(max_length)])
        combined_df = pd.concat([combined_df, feature_columns], axis=1).drop(columns=[channel])
    return combined_df
    
def softmax(values):
    exp_values = np.exp(values, dtype=np.longdouble)
    exp_values_sum = np.sum(exp_values)
    vals = np.asarray(exp_values/exp_values_sum, dtype=np.float64)

    return vals

def cosine_similarity(vector1, vector2):
    if vector1 is None or vector2 is None:
        return 0
    
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def KL_regression_labels(Q, P):
    from scipy.special import rel_entr

    return sum(rel_entr(P, Q))

def CE_regression_labels(O, P):

    cross_entropy = -np.sum(O * np.log(P + 1e-9))

    return cross_entropy

def euclideanDist(vector1, vector2):
    if vector1 is None or vector2 is None:
        return 0
    
    return np.linalg.norm(vector1 - vector2)

def MSE(vector1, vector2):
    if vector1 is None or vector2 is None:
        return 0
    
    return np.mean((vector1 - vector2) ** 2)
