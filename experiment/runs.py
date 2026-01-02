import numpy as np

multi_subject_experiments = {
    1: {"train_list": [2, 5, 7, 8, 9, 11, 13, 14, 15, 18, 19, 20, 21, 23], "train_conditions": ["FW"], "experiment_name": "multiSubject"}, 
    
    2: {"train_list": [3, 6, 9, 14, 16, 18, 19, 24], "train_conditions": ["FP"], "experiment_name": "multiSubject"}, 
    
    3: {"train_list": [3, 5, 6, 7, 8, 10, 12, 20, 23, 24, 25], "train_conditions": ["LW"], "experiment_name": "multiSubject"}, 

    4: {"train_list": [2, 3, 5, 10, 11, 13, 15, 16, 17, 18, 21, 23], "train_conditions": ["LP"], "experiment_name": "multiSubject"},

    5: {"train_list": [2, 5, 6, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 24], "train_conditions": ["RW"], "experiment_name": "multiSubject"},
    
    6: {"train_list": [3, 7, 8, 9, 10, 12, 14, 15, 17, 20, 21, 23, 24], "train_conditions": ["RP"], "experiment_name": "multiSubject"},

    7: {"train_list": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], "train_conditions": ["FW", "LW", "RW"], "experiment_name": "multiSubjectPassive"}, 
    
    8: {"train_list": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], "train_conditions": ["FP", "LP", "RP"], "experiment_name": "multiSubjectActive"}, 

    9: {"train_list": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], "train_conditions": ["FW", "LW", "RW", "FP", "LP", "RP"], "experiment_name": "multiSubjectAll"}, 

}

test_experiments = {
    0: {"test_list": [2], "test_condition": ["FW"]},
    1: {"test_list": [5], "test_condition": ["FW"]},
    2: {"test_list": [7], "test_condition": ["FW"]},
    3: {"test_list": [8], "test_condition": ["FW"]},
    4: {"test_list": [9], "test_condition": ["FW"]},
    5: {"test_list": [11], "test_condition": ["FW"]},
    6: {"test_list": [13], "test_condition": ["FW"]},
    7: {"test_list": [14], "test_condition": ["FW"]},
    8: {"test_list": [15], "test_condition": ["FW"]},
    9: {"test_list": [18], "test_condition": ["FW"]},
    10: {"test_list": [19], "test_condition": ["FW"]},
    11: {"test_list": [20], "test_condition": ["FW"]},
    12: {"test_list": [21], "test_condition": ["FW"]},
    13: {"test_list": [23], "test_condition": ["FW"]},
    
    14: {"test_list": [3], "test_condition": ["FP"]},
    15: {"test_list": [6], "test_condition": ["FP"]},
    16: {"test_list": [9], "test_condition": ["FP"]},
    17: {"test_list": [14], "test_condition": ["FP"]},
    18: {"test_list": [16], "test_condition": ["FP"]},
    19: {"test_list": [18], "test_condition": ["FP"]},
    20: {"test_list": [19], "test_condition": ["FP"]},
    21: {"test_list": [24], "test_condition": ["FP"]},
    
    22: {"test_list": [3], "test_condition": ["LW"]},
    23: {"test_list": [5], "test_condition": ["LW"]},
    24: {"test_list": [6], "test_condition": ["LW"]},
    25: {"test_list": [7], "test_condition": ["LW"]},
    26: {"test_list": [8], "test_condition": ["LW"]},
    27: {"test_list": [10], "test_condition": ["LW"]},
    28: {"test_list": [12], "test_condition": ["LW"]},
    29: {"test_list": [20], "test_condition": ["LW"]},
    30: {"test_list": [23], "test_condition": ["LW"]},
    31: {"test_list": [24], "test_condition": ["LW"]},
    32: {"test_list": [25], "test_condition": ["LW"]},

    33: {"test_list": [2], "test_condition": ["LP"]},
    34: {"test_list": [3], "test_condition": ["LP"]},
    35: {"test_list": [5], "test_condition": ["LP"]},
    36: {"test_list": [10], "test_condition": ["LP"]},
    37: {"test_list": [11], "test_condition": ["LP"]},
    38: {"test_list": [13], "test_condition": ["LP"]},
    39: {"test_list": [15], "test_condition": ["LP"]},
    40: {"test_list": [16], "test_condition": ["LP"]},
    41: {"test_list": [17], "test_condition": ["LP"]},
    42: {"test_list": [18], "test_condition": ["LP"]},
    43: {"test_list": [21], "test_condition": ["LP"]},
    44: {"test_list": [23], "test_condition": ["LP"]},

    45: {"test_list": [2], "test_condition": ["RW"]},
    46: {"test_list": [5], "test_condition": ["RW"]},
    47: {"test_list": [6], "test_condition": ["RW"]},
    48: {"test_list": [9], "test_condition": ["RW"]},
    49: {"test_list": [10], "test_condition": ["RW"]},
    50: {"test_list": [11], "test_condition": ["RW"]},
    51: {"test_list": [12], "test_condition": ["RW"]},
    52: {"test_list": [13], "test_condition": ["RW"]},
    53: {"test_list": [15], "test_condition": ["RW"]},
    54: {"test_list": [16], "test_condition": ["RW"]},
    55: {"test_list": [17], "test_condition": ["RW"]},
    56: {"test_list": [18], "test_condition": ["RW"]},
    57: {"test_list": [19], "test_condition": ["RW"]},
    58: {"test_list": [20], "test_condition": ["RW"]},
    59: {"test_list": [24], "test_condition": ["RW"]},
    
    60: {"test_list": [3], "test_condition": ["RP"]},
    61: {"test_list": [7], "test_condition": ["RP"]},
    62: {"test_list": [8], "test_condition": ["RP"]},
    63: {"test_list": [9], "test_condition": ["RP"]},
    64: {"test_list": [10], "test_condition": ["RP"]},
    65: {"test_list": [12], "test_condition": ["RP"]},
    66: {"test_list": [14], "test_condition": ["RP"]},
    67: {"test_list": [15], "test_condition": ["RP"]},
    68: {"test_list": [17], "test_condition": ["RP"]},
    69: {"test_list": [20], "test_condition": ["RP"]},
    70: {"test_list": [21], "test_condition": ["RP"]},
    71: {"test_list": [23], "test_condition": ["RP"]},
    72: {"test_list": [24], "test_condition": ["RP"]},
}