# 🏈 NFL 2025 Quantum & Classical AI Predictor

## Project Overview

This project is a machine learning pipeline designed to predict the outcomes of NFL games (specifically **Home Team wins**).
It compares the performance of three different neural network architectures:

- Classical Neural Network (TensorFlow / Keras)
- Quantum Neural Network (PennyLane / PyTorch)
- Hybrid Quantum–Classical Neural Network (PennyLane / PyTorch)

The system scrapes real NFL data from **footballdb.com**, processes team offensive and defensive statistics, and predicts future game outcomes.

---

## 1. Prerequisites & Installation

### System Requirements
- Python 3.9+

### Dependencies

```bash
pip install pandas numpy requests beautifulsoup4 openpyxl scikit-learn tensorflow torch pennylane seaborn matplotlib
```

> If no GPU is configured, TensorFlow and PyTorch will default to CPU.  
> Quantum simulations may run slower without sufficient RAM.

---

## 2. Project Workflow

### Step 1: Data Collection (get_data.py)

Scrapes NFL schedules, scores, and team statistics.

```bash
python get_data.py --week 16 --mode all
```

**Arguments**
- `--week`: Last completed NFL week (default: 14)
- `--mode`: stats | scores | all (recommended)
- `--type` (optional with scores): training | upcoming | all

**Output**
- Files saved in `prep_data/`

---

### Step 2: Data Processing (data_processing.py)

Merges statistics with schedules and computes differential features.

```bash
python data_processing.py
```

**Output**
- `processed_data/2025_processed_scores.xlsx`
- `processed_data/2025_processed_upcoming_games.xlsx`

---

## 3. Model Training

### A. Classical Neural Network (neural_network.py)

TensorFlow/Keras deep neural network.

```bash
python neural_network.py
```

**Output**
- `model/classical.keras`
- `model_stats/classical/`

---

### B. Quantum Neural Network (quantum_neural_network.py)

Pure quantum neural network using PennyLane.

```bash
python quantum_neural_network.py
```

**Configuration**
- `n_qubits` (default: 7)

**Output**
- `model/quantum.pth`
- `model_stats/quantum/`

---

### C. Hybrid Quantum–Classical Network (hybrid_neural_network.py)

Classical preprocessing + quantum layer.

```bash
python hybrid_neural_network.py
```

**Output**
- `model/hybrid.pth`
- `model_stats/hybrid/`

---

You can run one of the 3 files to get directly the predictions with the model we found to be the best performing for each of the architectures.

## 4. Evaluation (evaluate_models.py)

Evaluates prediction accuracy once game results are available.

```bash
python evaluate_models.py
```

---

## 5. Directory Structure

```
├── get_data.py
├── data_processing.py
├── neural_network.py
├── quantum_neural_network.py
├── hybrid_neural_network.py
├── evaluate_models.py
│
├── prep_data/
├── processed_data/
├── model/
├── model_stats/
└── predictions/
```

There are also other files and directories that are not really useful for you to run the experiments but were useful for us for the creation of the models and for the plots and the writing of the report.

---

## 6. Troubleshooting

**Scraping Errors**
- Check access to footballdb.com
- Avoid frequent repeated requests

**Memory Issues**
- Reduce `n_qubits` for quantum models

**Missing Columns**
- Ensure `get_data.py --mode all` was run first
- Don't forget to run `data_processing.py` afterwards

---

## Disclaimer

This project is for research and educational purposes only.
Predictions should not be used for betting or financial decisions.
