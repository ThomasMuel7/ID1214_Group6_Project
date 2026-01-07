# 🏈 NFL 2025 Quantum & Classical AI Predictor

## Project Overview

This project is a **machine learning pipeline** designed to predict the outcomes of NFL games, specifically **Home Team wins**.  
It compares the performance of three different neural network architectures:

- **Classical Neural Network** (TensorFlow / Keras)
- **Quantum Neural Network** (PennyLane / PyTorch)
- **Hybrid Quantum–Classical Neural Network** (PennyLane / PyTorch)

The system scrapes real-time NFL data from **footballdb.com**, processes offensive and defensive team statistics, and predicts future game outcomes.

---

## 1. Prerequisites & Installation

### System Requirements
- Python **3.9+**
- Stable internet connection (required for data scraping)

### Dependencies

Install all required libraries using `pip`:

```bash
pip install pandas numpy requests beautifulsoup4 openpyxl scikit-learn tensorflow torch pennylane seaborn matplotlib
```

---

## 2. Project Workflow

⚠️ **Important:** The pipeline must be run in the order described below.

---

### Step 1: Data Collection — `get_data.py`

This script scrapes schedules, scores, and detailed team statistics.

#### Usage

```bash
python get_data.py --week 14 --mode all
```

#### Arguments

- `--week`  
  The last completed week of the NFL season (used to separate training/testing/validation data from upcoming games).  
  **Default:** `14`

- `--mode`
  - `stats` → Scrapes only team statistics
  - `scores` → Scrapes only game schedules and scores
  - `all` → Scrapes both stats and scores (**Recommended**)

- `--type` *(optional, used with `scores` mode)*  
  `training`, `upcoming`, or `all`

#### Output
Files are saved to the `prep_data/` directory.


#### Note
You don't need to run this part of the code if you want to try the architectures of our neural network that got those results.

---

### Step 2: Data Processing — `data_processing.py`

This script merges raw statistics with game schedules and computes **differential statistics**  
(e.g., Home Rushing Yards − Visitor Rushing Yards) to generate model features.

#### Usage

```bash
python data_processing.py
```

#### Output
- `processed_data/2025_processed_scores.xlsx` *(Training Data)*
- `processed_data/2025_processed_upcoming_games.xlsx` *(Prediction Data)*

#### Note
You don't need to run this part of the code if you want to try the architectures of our neural network that got those results.

---

### Step 3: Model Training and Predictions Making

You may train one or all of the following models.

---

#### A. Classical Neural Network — `neural_network.py`

Trains a standard deep neural network using TensorFlow/Keras. 

**Configuration**
- Toggle `use_pca = True/False`
- Adjust `n_components` at the bottom of the script

```bash
python neural_network.py
```

**Output**
- Model: `model/classical.keras`
- Stats: `model_stats/classical/`

---

#### B. Quantum Neural Network — `quantum_neural_network.py`

Trains a pure quantum neural network using PennyLane.

> Uses a quantum simulator (`lightning.qubit`).  
> Training time may be significantly longer than classical models.

**Configuration**
- Number of qubits: `n_qubits = 7` (default)

```bash
python quantum_neural_network.py
```

**Output**
- Model: `model/quantum.pth`
- Stats: `model_stats/quantum/`

---

#### C. Hybrid Quantum–Classical Neural Network — `hybrid_neural_network.py`

Combines classical feature extraction with a quantum layer.

**Configuration**
- Toggle `use_pca`
- Adjust `n_components` (default: 32)

```bash
python hybrid_neural_network.py
```

**Output**
- Model: `model/hybrid.pth`
- Stats: `model_stats/hybrid/`

---

### Step 4: Evaluation — `evaluate_models.py`

Evaluates prediction accuracy once actual results are available.

```bash
python evaluate_models.py
```

**Output**
- Prints accuracy for each model found in the `predictions/` directory.

---

## 3. Directory Structure

```
├── get_data.py
├── data_processing.py
├── neural_network.py
├── quantum_neural_network.py
├── hybrid_neural_network.py
├── evaluate_models.py
│
├── prep_data/
│   ├── 2025_scores.xlsx
│   ├── 2025_passing_offense.xlsx
│   └── ...
│
├── processed_data/
│   ├── 2025_processed_scores.xlsx
│   └── 2025_processed_upcoming_games.xlsx
│
├── model/
│   ├── classical.keras
│   ├── hybrid.pth
│   ├── quantum.pth
│   └── ...
│
├── model_stats/
│   ├── classical/
│   ├── hybrid/
│   └── quantum/
│       ├── accuracy.png
│       ├── loss.png
│       ├── report.txt
│       └── ...
│
└── predictions/
    ├── classical_predictions.xlsx
    ├── hybrid_predictions.xlsx
    ├── quantum_predictions.xlsx
    └── ...
```

There are other files and directories that are not mentioned above but that have helped us with the creation of the model, the writing of the report and the showing part of the presentation. We kept them in this git repo but they do not interfere with the experiment.

---

## 4. Troubleshooting

### Scraping Errors
- Ensure **footballdb.com** is accessible
- Avoid aggressive scraping (may trigger rate limits)

### Memory Issues
- Quantum simulations use state vectors
- Reduce `n_qubits` if you encounter RAM limitations

### Missing Columns
- Make sure `get_data.py --mode all` ran successfully
- All stat categories must exist in `prep_data/`

## 5. Further Reading

We haven't hardcoded every model since some of them have different architectures and some also use PCA. Depending on the model you want to train and try out, you might have to toggle the use_pca variable to `True`. You might also have to change the architecture of the neural network itself since we only kept 3 of the 5 models in the files. They are hardcoded.

We could have changed this in order chose directly which model we want to try. But we felt like it was better to allow you to change the model architecture as you want and make it modulable. 

We also haven't created any files/code that retrieves one of the models to make further predictions. We always have to train it first in order to get the predictions. But feel free to implement a file like that.