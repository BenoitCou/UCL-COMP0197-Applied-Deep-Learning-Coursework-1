# UCL COMP0197 Coursework 1 (2024-25) – Applied Deep Learning  

This repository contains the **Assessed Component 1 (CW 1)** of the UCL module **COMP0197 – Applied Deep Learning**.  
It implements and evaluates the two required tasks:

1. **Task 1 – Optimising Logistic Binary Regression Models**  
2. **Task 2 – Regularising Extreme Learning Machines (ELMs)**  

---

## Repository Layout

```text
cw1-pt/ or cw1-tf/        # root folder specifies chosen framework
├── task1/
│   ├── task.py           # main script
│   ├── task1a.py         # extension: learnable polynomial order
│   └── *.png             # saved visualisations
├── task2/
│   ├── task.py           # ELM with optional MixUp & Ensemble
│   ├── task2a.py         # extension: least-squares solver + search
│   ├── saved_models/     # pre-trained checkpoints (< 100 MB total)
│   └── *.png             # result, mixup & new_result montages
├── Assessed-Component-1 COMP0197 2024-25.pdf   # document containing the questions of the coursework
└── README.md
```

## Code Setup

```bash
# 1. Clone the repo
git clone https://github.com/BenoitCou/UCL-COMP0197-Applied-Deep-Learning-Coursework-1
cd UCL-COMP0197-Applied-Deep-Learning-Coursework-1/cw1-pt

# 2. Activate the environment
conda create -n comp0197-cw1-pt python=3.12 pip
conda activate comp0197-cw1-pt
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Run the code
python task1\task.py
python task1\task1a.py
python task2\task.py
python task2\task2a.py

```
## Running the Tasks

All scripts are self-contained; run them from inside their own folder so that relative paths resolve.

| Purpose                          | Command                  | Outputs                         |
| -------------------------------- | ------------------------ | ------------------------------- |
| Train & evaluate Task 1          | `python task1/task.py`   | console log + `*.png`           |
| Extension 1 – learnable *M*      | `python task1/task1a.py` | console log                     |
| Train & evaluate Task 2          | `python task2/task.py`   | log + `mixup.png`, `result.png` |
| Extension 2 – LS solver & search | `python task2/task2a.py` | log + `new_result.png`          |

By default each script:
- seeds NumPy and the backend framework for reproducibility,
- prints loss values at least ten times during optimisation,
- writes required montages to PNG in its working directory,
- loads pre-trained weights from `saved_models/` (Task 2) to avoid long retraining.

## What the Code Does

**Task 1 – Logistic Regression**
- Implements a fully-enumerated polynomial logistic model logistic_fun with order M.
- Provides custom losses MyCrossEntropy and MyRootMeanSquare.
- Optimises with stochastic mini-batch gradient descent (fit_logistic_sgd).
- Reports both loss values and an additional classification metric (Accuracy) for train/test splits at M ∈ {1, 2, 3}.
- **task1a.py** treats M as a learnable integer via straight-through estimation and SGD.

**Task 2 – Extreme Learning Machines**
- Defines `MyExtremeLearningMachine` – one fixed convolutional layer (Gaussian-initialised) feeding a trainable FC layer.
- Adds two regularisers:
  -   `MyMixUp` – on-the-fly mixup augmentation (seeded).
  -   `MyEnsembleELM` – averages logits from several independently-initialised ELMs.
- `fit_elm_sgd` trains with mini-batch SGD; metrics (Accuracy & Macro-F1) are logged per epoch.
- **task2a.py** compares least-squares training (`fit_elm_ls`) vs. SGD, then performs random hyper-parameter search to exceed the previous best model.

## Coursework Mark & Feedback

| Task                                                                     |     Max | Awarded |
| ------------------------------------------------------------------------ | ------: | ------: |
| Task 1 – SGD for Logistic Regression                                     |      50 |      41 |
| Task 2 – Extreme Learning Machines <br>(includes Task 2a ablation study) |      50 |      44 |
| **Overall – Coding component**                                           | **100** |  **85** |


