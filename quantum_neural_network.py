
import pandas as pd
import random
import numpy as np
import pennylane as qml
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

reps = 2
n_qubits = 7
dev = qml.device('lightning.qubit', wires=n_qubits)

SEED = 47
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def process_data():
    processed_scores = pd.read_excel('processed_data/2025_processed_scores.xlsx')
    upcoming_games = pd.read_excel('processed_data/2025_processed_upcoming_games.xlsx')

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

    Xtr_t = torch.from_numpy(X_train).float()
    ytr_t = torch.from_numpy(y_train.astype(np.float64)).float().reshape(-1, 1)
    Xval_t = torch.from_numpy(X_val).float()
    yval_t = torch.from_numpy(y_val.astype(np.float64)).float().reshape(-1, 1)
    Xte_t  = torch.from_numpy(X_test).float()
    yte_t  = torch.from_numpy(y_test.astype(np.float64)).float().reshape(-1, 1)
    X_upcoming_t = torch.from_numpy(X_upcoming).float()
        
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
        if getattr(qnode, "interface", None) not in ("torch",):
            qnode = qml.QNode(qnode.func, qnode.device, interface="torch", diff_method="best")
        self.qnode = qnode
        self.params = nn.ParameterDict()
        for name, shape in weight_shapes.items():
            if isinstance(shape, int):
                shape = (shape,)
            elif isinstance(shape, (list, tuple)):
                shape = tuple(shape)
            else:
                raise ValueError(f"Invalid shape for {name}: {shape}")
            p = nn.Parameter(torch.randn(*shape, dtype=torch.float32) * 0.1)
            self.params[name] = p

    def forward(self, x):
        x = x.to(dtype=torch.float32)
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
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_time': [], 'val_time': []}
    for epoch in range(1, epochs+1):
        train_start = time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            predicted = (preds >= 0.5).double()
            train_correct += (predicted == yb).sum().item()
        train_end = time()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_start = time()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                predicted = (preds >= 0.5).double()
                val_correct += (predicted == yb).sum().item()
        val_end = time()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        history['train_time'].append(train_end - train_start)
        history['val_time'].append(val_end - val_start)

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
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch:02d}/{epochs} -     loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")
        
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
    return cm, report
    
def predictions(X_upcoming_t, results, model):
    with torch.no_grad():
        probs = model(X_upcoming_t).squeeze().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    results['Model_Home_win'] = preds.flatten()
    results['Model_home_win_probability'] = probs
    results.to_excel('./predictions/quantum_predictions.xlsx', index=False)
    print(f"Predictions saved to ./predictions/quantum_predictions.xlsx")

def save_stats(history, cm, report, seed=None, n_components=None):
    out_dir = "./model_stats/quantum"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'report.txt')
    loss_path = os.path.join(out_dir, 'loss.png')
    acc_path = os.path.join(out_dir, 'accuracy.png')
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    time_path = os.path.join(out_dir, 'training_time.png')

    # write report
    train_losses = history.get('train_loss', [])
    val_losses = history.get('val_loss', [])
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])
    train_time = history.get('train_time', [])
    val_time = history.get('val_time', [])

    with open(report_path, 'w') as f:
        f.write(str(report) + '\n')
        f.write(f"Total Time = {sum(train_time) + sum(val_time)} seconds\n")
        f.write(f"Random Seed = {seed}\n")
        f.write(f"Number of PCA components = {n_components}\n")

    # Loss plot
    if train_losses or val_losses:
        plt.figure(figsize=(8, 5))
        if train_losses:
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()

    # Accuracy plot
    if train_acc or val_acc:
        plt.figure(figsize=(8, 5))
        if train_acc:
            plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train Acc')
        if val_acc:
            plt.plot(range(1, len(val_acc) + 1), val_acc, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(acc_path)
        plt.close()

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    # Time plot
    if train_time or val_time:
        max_epochs = max(len(train_time), len(val_time))
        tt = train_time + [0] * (max_epochs - len(train_time))
        vt = val_time + [0] * (max_epochs - len(val_time))
        total = [a + b for a, b in zip(tt, vt)]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_epochs + 1), tt, label='Train time (s)')
        plt.plot(range(1, max_epochs + 1), vt, label='Val time (s)')
        plt.plot(range(1, max_epochs + 1), total, label='Train+Val time (s)')
        plt.xlabel('Epoch')
        plt.ylabel('Seconds')
        plt.title('Per-epoch Training and Validation Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(time_path)
        plt.close()

    print(f"Stats saved to: {out_dir}")
    
def save_model(model):    
    filename = './model/quantum.pth'
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
    
#------- Main code execution -------
weight_shapes = {"theta": (reps, n_qubits, 3)}

model = nn.Sequential(BatchedQNodeLayer(qnode, weight_shapes)).float()
train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results = transform_data(batch_size=20)
model, history = train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3)
cm, report = test_model(Xte_t, yte_t, model)
save_stats(history=history, cm=cm, report=report, seed=SEED, n_components=n_qubits)
save_model(model=model)
predictions(X_upcoming_t, results, model)
