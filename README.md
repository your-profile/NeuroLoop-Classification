# Supplementary Material for NEURO-LOOP

This archive contains the supplementary materials for our paper submitted to IJCAI 2025.

## Contents

- `supplementary.pdf` - A document containing participant instructions, questionnaires, dataset description, participant eligibility, and model hyperparameters.
- `dataset/` - The dataset collected from participants, specifically the time classification window dataset.
- `code/` - Source code for data processing, model training, and evaluation.
  - `data_labels.ipynb` - Script for concatenating and labeling the data.
  - `data_windows.ipynb` - Script for creating windows from labeled data.
  - `singleSubject_experiment.ipynb` - Script for evaluating single subject model performance.
  - `multiSubject_experiment.ipynb` - Script for multi subject model performance.
  - `requirements.txt` - List of dependencies needed to run the code.

## Instructions

### Dataset
The dataset is stored in the `dataset/` directory. It includes time series classificaiton windows from the fNIRS data.

### Code
All procedural code that participants interacted with is labeled `procedure_[ENV].py`.

All machine learning code is labeled `data_[PREPOCESS_TASK].ipynb`. The data must be concatenated, labeled and then created into windows before training single and multi-subject models.

Machine learning experiments can be found in the `experiments/` directory of the code base.
