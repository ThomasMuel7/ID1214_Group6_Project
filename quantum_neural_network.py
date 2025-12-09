
import pandas as pd
import random
import numpy as np
import seaborn as sns
import pennylane as qml
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader


n_qubits = 8
dev = qml.device('default.qubit', wires=n_qubits)
reps = 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def process_data():
    # expect CSV in repository data folder with .csv extension
    processed_scores = pd.read_csv('data/2025_processed_scores.csv')
    upcoming_games = pd.read_csv('data/2025_processed_upcoming_games.csv')

    scores_clean = processed_scores.dropna()
    upcoming_games = upcoming_games.dropna()

    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
    X_upcoming = upcoming_games[feature_cols].values
    results = upcoming_games[['Home', 'Visitor']].copy()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_upcoming = scaler.transform(X_upcoming)

    pca = PCA(n_components=n_qubits)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    X_upcoming = pca.transform(X_upcoming)
    
    return X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results

def transform_data(batch_size=20):
    X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results = process_data()

    Xtr_t = torch.from_numpy(X_train).double()
    ytr_t = torch.from_numpy(y_train.astype(np.float64)).double().reshape(-1, 1)
    Xval_t = torch.from_numpy(X_val).double()
    yval_t = torch.from_numpy(y_val.astype(np.float64)).double().reshape(-1, 1)
    Xte_t  = torch.from_numpy(X_test).double()
    yte_t  = torch.from_numpy(y_test.astype(np.float64)).double().reshape(-1, 1)
    X_upcoming_t = torch.from_numpy(X_upcoming).double()
        
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval_t, yval_t), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results

@qml.qnode(dev, interface='torch', diff_method='best')
def qnode(x, theta):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(theta, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class BatchedQNodeLayer(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super().__init__()
        # ensure torch interface
        if getattr(qnode, "interface", None) not in ("torch",):
            qnode = qml.QNode(qnode.func, qnode.device, interface="torch", diff_method="best")
        self.qnode = qnode
        # register parameters
        self.params = nn.ParameterDict()
        for name, shape in weight_shapes.items():
            if isinstance(shape, int):
                shape = (shape,)
            elif isinstance(shape, (list, tuple)):
                shape = tuple(shape)
            else:
                raise ValueError(f"Invalid shape for {name}: {shape}")
            p = nn.Parameter(torch.randn(*shape, dtype=torch.float64) * 0.1)
            self.params[name] = p

    def forward(self, x):
        x = x.to(dtype=torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        outs = []
        for i in range(x.shape[0]):
            ev = self.qnode(x[i], **{k: v for k, v in self.params.items()})
            if not isinstance(ev, torch.Tensor):
                ev = torch.tensor(ev, dtype=torch.float64)
            p = (ev + 1.0) / 2.0
            outs.append(p.reshape(()))
        return torch.stack(outs, dim=0).unsqueeze(1)

def train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3):
    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def test_model(Xte_t, y_test, model):
    model.eval()
    with torch.no_grad():
        probs = model(Xte_t).cpu().numpy()
    
    y_pred = (probs >= 0.5).astype(np.int64)
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.cpu().numpy()
    y_test = y_test.astype(np.int64)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print("="*60)
    print(report)
    
    plt.figure()
    sns.set(font_scale=1.0)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
def predictions(X_upcoming_t, results, model):
    with torch.no_grad():
        probs = model(X_upcoming_t).squeeze().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    results['Model_Home_win'] = preds.flatten()
    results['Model_home_win_probability'] = probs
    results.to_excel('./predictions/quantum_predictions.xlsx', index=False)
    print(f"Predictions saved to ./predictions/quantum_predictions.xlsx")
    print(results.head())

def plot_history(history):
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
#------- Main code execution -------
weight_shapes = {"theta": (reps, n_qubits, 3)}
model = nn.Sequential(BatchedQNodeLayer(qnode, weight_shapes)).double()
train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results = transform_data(batch_size=20)
model, history = train_model(model, train_loader, val_loader, epochs=15, patience=5, lr=1e-3)
test_model(Xte_t, yte_t, model)
predictions(X_upcoming_t, results, model)
plot_history(history)