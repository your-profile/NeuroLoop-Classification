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
    data = {}

    # Add features
    for feature in features:
        data[feature] = participant_df[feature]
    
    # Add labels
    print(labels)
    for label in labels:
        data[label] = participant_df[label]

    df = pd.DataFrame(data)

    windowed_data = []
    windowed_labels = []
    slope_values = []
    intercept_values = []
    start_timestamps = []
    end_timestamps = []

    for start in range(0, len(df) - window_size + 1, step_size):
        data_dict = {}
        end = start + window_size
        window = df.iloc[start:end] 

        last_discrete_label = window["discrete_optimal"].iloc[-1]
        last_continuous_label = window['continuous_optimal'].iloc[-1]
        last_binary_label = window['binary_optimal'].iloc[-1]

        window = window.drop(columns=['continuous_optimal', 'binary_optimal', 'discrete_optimal'])

        # Calculate window features
        mean_values = window.mean(axis=0).to_numpy(dtype=float)
        std_values = window.std(axis=0).to_numpy(dtype=float)
        slope_values = np.array([np.polyfit(window[feature], np.arange(window_size), 1)[0] for feature in features], dtype=float)
        intercept_values = np.array([np.polyfit(window[feature], np.arange(window_size), 1)[1] for feature in features], dtype=float)
        kurtosis_values = stats.kurtosis(window, axis=0, fisher=True)
        skewdness_values = stats.skew(window, axis=0)

        # Add features
        for i, feature in enumerate(features):
            for j, stat in feature_map.items():
                data_dict[f"{feature}_{stat}"] = np.array([mean_values[i], std_values[i], slope_values[i], intercept_values[i], kurtosis_values[i], skewdness_values[i]])[j]

        # Add start and end timestamps
        start_timestamps.append(participant_df['time'].iloc[start])
        end_timestamps.append(participant_df['time'].iloc[end - 1])

        windowed_data.append(data_dict)
        windowed_labels.append({'discrete_label': last_discrete_label, 'continuous_label': last_continuous_label, 'binary_label': last_binary_label})

    windowed_data = pd.DataFrame(windowed_data)
    windowed_labels_df = pd.DataFrame(windowed_labels)

    # Add start and end timestamps to the windowed data
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
