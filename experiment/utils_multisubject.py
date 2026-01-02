import numpy as np
import pandas as pd
import os
import scipy.stats as stats


feature_map = {0:"Mean",
                1:"Std",
                2:"Slope",
                3:"Intercept",
                4:"Kurtosis",
                5:"Skewness"
            }


def from_tensor(x):
    try:
        x = x[0].detach().cpu().numpy()
        return softmax(x)
    except:
        return x

def create_window_df(participant_df, window_size:int, step_size:int, features, labels):
    feature_map = {0:"Mean",
                1:"Std",
                2:"Slope",
                3:"Intercept",
                4:"Kurtosis",
                5:"Skewness"
                }
    
    data = {f: participant_df[f] for f in features}
    for label in labels:
        data[label] = participant_df[label]
    df = pd.DataFrame(data)

    windowed_data, windowed_labels = [], []
    start_timestamps, end_timestamps = [], []

    for start in range(0, len(df) - window_size + 1, step_size):
        label_idx = -1
        
        end = start + window_size
        window = df.iloc[start:end].copy()

        # extract labels at the end of window
        last_discrete_label = window["discrete_optimal"].iloc[label_idx]
        last_continuous_label = window["continuous_optimal"].iloc[label_idx]
        last_binary_label = window["binary_optimal"].iloc[label_idx]
        window = window.drop(columns=['continuous_optimal','binary_optimal','discrete_optimal'])

        mean_values = window.mean(axis=0).to_numpy(float)
        std_values = window.std(axis=0).to_numpy(float)
        slopes = np.array([np.polyfit(np.arange(window_size), window[f], 1)[0] for f in features], dtype=float)
        intercepts = np.array([np.polyfit(np.arange(window_size), window[f], 1)[1] for f in features], dtype=float)
        kurtosis_values = stats.kurtosis(window, axis=0, fisher=True, nan_policy='omit')
        skewness_values = stats.skew(window, axis=0, nan_policy='omit')

        data_dict = {}
        for i, f in enumerate(features):
            for idx, stat_name in feature_map.items():
                data_dict[f"{f}_{stat_name}"] = [mean_values[i], std_values[i], slopes[i], intercepts[i], kurtosis_values[i], skewness_values[i]][idx]

        windowed_data.append(data_dict)
        windowed_labels.append({
            'discrete_label': last_discrete_label,
            'continuous_label': last_continuous_label,
            'binary_label': last_binary_label
        })
        start_timestamps.append(participant_df['time'].iloc[start])
        end_timestamps.append(participant_df['time'].iloc[end - 1])

    windowed_data = pd.DataFrame(windowed_data)
    windowed_labels_df = pd.DataFrame(windowed_labels)
    windowed_data['start_timestamp'] = start_timestamps
    windowed_data['end_timestamp'] = end_timestamps
    
    return windowed_data, windowed_labels_df
    
def read_files(participant_list, source_folder_1, conditions):
    participant_data = {}
    # Iterate over participants and files
    for filename in os.listdir(source_folder_1):
        for condition in conditions:
            for participant in participant_list:
                if participant < 10:
                    participant = '00{}'.format(participant)
                else:
                    participant = '0{}'.format(participant)
                if (filename.startswith('{}'.format(participant))):
                    if filename[4:6] == condition:
                        print(filename)
                        # Get full file path
                        demo_path = os.path.join(source_folder_1, filename)
                        df = pd.read_csv(demo_path, index_col=0)
                        participant_data[str(participant)+condition] = df
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
