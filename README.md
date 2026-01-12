# NeuroLoop Classification

### NOTE: See the aaai26 branch for the repository containing source code for:

Santaniello, J., Russell, M., Jiang, B., Sassaroli, D., Jacob, R., & Sinapov, J. (2026).  
**Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance.**  
*To appear in AAAI 2026.*

**Code for AAAI-26:** [NeuroLoop-Classification GitHub repository](https://github.com/your-profile/NeuroLoop-Classification/tree/aaai)  
**Dataset link:** [fNIRS2RL Dataset GitHub repository](https://github.com/your-profile/fNIRS2RL)  

---

## Table of Contents
- [Overview](#overview)
- [Experimental Setup](#experimental-setup)
- [Directory Structure](#directory-structure)
- [Labeling](#labeling)
- [Ethics and Consent](#ethics-and-consent)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---
## Overview

Classical machine learning techniques are used as a first step towards determining if fNIRS signals can be mapped to various levels of agent optimality, or performance. This repository uses Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Random Forest (RF) and simple Multi-Layer Perceptrons (MLP) to test the feasibility of turning fNIRS data into usable feedback signals for Reinforcement Learning from Human (Neural) Feedback. 

We specifically focus on the **Neural Classification** problem, as future work will focus on applying these decoded signals to an RL loop.

---

## Experimental Setup

- **PID:** Participant ID (1-25)
- **Domain Type:** Domain (e.g. Robot, Lunar Lander or Flappy Bird)
- **Task Type:** Interaction Type (e.g., passive observation (W - watch), active teleoperation (P - play))
- **Condition Type:** Domain-Task (e.g., LP, RW)
- **Agent:** RL policy (autonomous or teleoperated)
- **Human Role:** Implicit evaluator (neural feedback only, no explicit labels) or Explicit Evaluator (neural feedback and teleoperation/explicit play)

Participants interact with some **domain** through some **task**. Domains are one of three environments: Robot, Lunar Lander or Flappy Bird. Tasks are one of two types: Passive Observation (Watch Task) or Active Teleoperation (Play Task). A condition is what the participant is asked to complete (e.g. teleoperation of the robot or observing the robot).

**Sampling Rate:** 5.2 Hz
**fNIRS Device:** ISS Imagent fNIRS Device
**Task Duration:** 2-5 minutes

---

## Directory Structure

```text
.
├── neuroloop-classification/
│    ├── experiment/
│    │   ├── results/
│    │   │   ├── data/
│    │   │   │   ├── multiSubject_binary.csv
│    │   │   │   ├── multiSubject_discrete.csv
│    │   │   │   └── ...
│    │   │   └── models/
│    │   │       ├── Participant[2, 5, 6]_ConditionRW_MLP_binary.csv
│    │   │       └── ...
│    │   ├── NASATLX_averages.csv
│    ├── EXP_classical_models.ipynb
│    ├── runs.py
│    └── utils_multisubject.py
├── games/
│   ├── flappy_bird.py
│   └── ...
├── networks/
│   ├── ddpg_fetchrobot.py
│   └── ...
├── policies/
│   ├── FlappyBirdPolicies
│   └── ...
├── FIGURES_{figure_type}.ipynb
├── procedure_{domain}.py
├── utils.py
└── README.md

```
---

## Labeling

To be documented on 01/05/26.

## Ethics and Consent

This study was approved by the Institutional Review Board (IRB) of Tufts University's Social, Behavioral, and Educational Research Office (SBER) under protocol IRB-00005080, and all participants provided informed consent.

## Citation 

Santaniello, J., Russell, M., Jiang, B., Sassaroli, D., Jacob, R., & Sinapov, J. (2026).  
**Towards Reinforcement Learning from Neural Feedback: Mapping fNIRS Signals to Agent Performance.**  
*To appear in AAAI 2026.*


## Contact
Contact Julia Santaniello for inquiries: [julia.santaniello@tufts.edu](mailto:julia.santaniello@tufts.edu)


