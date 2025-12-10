import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Reproducibility
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

n_qubits = 16
reps = 1
dev = qml.device("lightning.qubit", wires=n_qubits)

def process_data(use_pca=False, n_components=30):
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
    
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)  
        X_test = pca.transform(X_test)
        X_upcoming = pca.transform(X_upcoming)
    
    input_dim = X_train.shape[1]
    
    return X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results, input_dim

def transform_data(use_pca=False, n_components=30, batch_size=20):
    data = process_data(use_pca=use_pca, n_components=n_components)

    X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results, input_dim = data

    Xtr_t = torch.from_numpy(X_train).double()
    ytr_t = torch.from_numpy(y_train.astype(np.float64)).double().reshape(-1, 1)
    Xval_t = torch.from_numpy(X_val).double()
    yval_t = torch.from_numpy(y_val.astype(np.float64)).double().reshape(-1, 1)
    Xte_t  = torch.from_numpy(X_test).double()
    yte_t  = torch.from_numpy(y_test.astype(np.float64)).double().reshape(-1, 1)
    X_upcoming_t = torch.from_numpy(X_upcoming).double()
        
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval_t, yval_t), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results, input_dim

@qml.qnode(dev, interface='torch', diff_method='best')
def qnode(inputs, theta):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(theta, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

class HybridBinaryClassifier(nn.Module):
    def __init__(self, num_dim, n_qubits, weight_shapes):
        super().__init__()
        self.fc1 = nn.Linear(num_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_qubits) 
        self.act1 = nn.ReLU()
        self.act_out = nn.Tanh() 
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act_out(self.fc3(x)) 
        x = x * np.pi 
        x = self.qlayer(x) 
        x = (x + 1) / 2
        if x.shape[-1] != 1:
            x = x.reshape(-1, 1)
        return x

def train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3):
    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.001)
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

def predictions(use_pca, X_upcoming_t, results, model):
    model.eval()
    with torch.no_grad():
        probs = model(X_upcoming_t).cpu().numpy()
        
    preds = (probs >= 0.5).astype(np.int64)
    results['Model_Home_win'] = preds.flatten()
    results['Model_home_win_probability'] = probs.flatten()
    
    if use_pca: 
        filename = './predictions/hybrid_pca_predictions.xlsx'
    else :
        filename = './predictions/hybrid_predictions.xlsx'
        
    results.to_excel(filename, index=False)
    print(f"Predictions saved to {filename}")
    print(results.head())

def plot_history(history):
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    
#------- Main code execution -------
use_pca = False
n_components = 30
weight_shapes = {"theta": (reps, n_qubits, 3)}

train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results, input_dim = transform_data(batch_size=20, use_pca=use_pca, n_components=n_components)
model = HybridBinaryClassifier(num_dim=input_dim, n_qubits=n_qubits, weight_shapes=weight_shapes)
model, history = train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=1e-3)
test_model(Xte_t, yte_t, model)
predictions(use_pca=use_pca, X_upcoming_t=X_upcoming_t, results=results, model=model)
plot_history(history)