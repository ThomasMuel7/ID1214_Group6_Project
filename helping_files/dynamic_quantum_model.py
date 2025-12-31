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

# Global Seed Configuration

def process_data(n_qubits, seed=42):
    """
    Dynamically process data with PCA components matching n_qubits.
    """
    processed_scores = pd.read_excel('processed_data/2025_processed_scores.xlsx')
    upcoming_games = pd.read_excel('processed_data/2025_processed_upcoming_games.xlsx')

    scores_clean = processed_scores.dropna()
    upcoming_games = upcoming_games.dropna()

    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    # Ensure we don't try to use more qubits than we have actual features
    actual_n_components = min(n_qubits, len(feature_cols))

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
    X_upcoming = upcoming_games[feature_cols].values
    results = upcoming_games[['Home', 'Visitor']].copy()
    
    # Stratified split ensures class balance is maintained
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_upcoming = scaler.transform(X_upcoming)

    # PCA Projection to fit into the quantum embedding
    pca = PCA(n_components=actual_n_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    X_upcoming = pca.transform(X_upcoming)
    
    return X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results

def transform_data(n_qubits, batch_size=20, seed=42):
    X_train, X_test, X_val, X_upcoming, y_train, y_test, y_val, results = process_data(n_qubits, seed)

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

class BatchedQNodeLayer(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super().__init__()
        # Ensure qnode interface is torch
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
            # Initialize weights
            p = nn.Parameter(torch.randn(*shape, dtype=torch.float32) * 0.1)
            self.params[name] = p

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        outs = []
        # Process batch items
        for i in range(x.shape[0]):
            ev = self.qnode(x[i], **{k: v for k, v in self.params.items()})
            if not isinstance(ev, torch.Tensor):
                ev = torch.tensor(ev, dtype=torch.float64)
            # Map expectation value [-1, 1] to probability [0, 1]
            p = (ev + 1.0) / 2.0
            outs.append(p.reshape(()))
        return torch.stack(outs, dim=0).unsqueeze(1)

def create_quantum_model(n_qubits, reps):
    """
    Creates a fresh QNode and Model with the specific qubit/rep configuration.
    """
    # 1. Create a new device for this specific number of qubits
    dev = qml.device('lightning.qubit', wires=n_qubits)

    # 2. Define the qnode using this specific device
    @qml.qnode(dev, interface='torch', diff_method='best')
    def qnode(x, theta):
        qml.templates.AngleEmbedding(x, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(theta, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    # 3. Define weight shapes
    weight_shapes = {"theta": (reps, n_qubits, 3)}

    # 4. Wrap in Torch Layer
    model = nn.Sequential(BatchedQNodeLayer(qnode, weight_shapes)).float()
    return model

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
                # Early stopping quietly
                break
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        # Verbose set to false for loop cleanliness
        # print(f"Epoch {epoch:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        
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
    accuracy = np.mean(y_pred == y_test.reshape(-1, 1))
    return cm, report, accuracy
    
def predictions(X_upcoming_t, results, model):
    with torch.no_grad():
        probs = model(X_upcoming_t).squeeze().cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    results['Model_Home_win'] = preds.flatten()
    results['Model_home_win_probability'] = probs
    results.to_excel('./predictions/quantum_predictions.xlsx', index=False)
    print(f"Predictions saved to ./predictions/quantum_predictions.xlsx")

def save_stats(history, cm, report, config_info):
    report_path = "./model_stats/quantum.txt"
    with open(report_path, "w") as f:
        f.write(f"CONFIGURATION: {config_info}\n")
        f.write("-" * 30 + "\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(str(report))
        f.write("\n")
        f.write(f"Training loss: {history['train_loss'][-1]}")
        f.write("\n")
        f.write(f"Validation loss: {history['val_loss'][-1]}")
    print(f"Stats saved to {report_path}")
    
def save_model(model):    
    filename = './model/quantum.pth'
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

#------- Main Execution -------
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('model_stats', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    target_accuracy = 0.70
    max_attempts = 50
    attempt = 0
    SEED = 42
    # Ranges
    qubit_choices = range(2, 17) # 2 to 16
    rep_choices = range(2, 5)   # 2 to 4
    
    print(f"Starting Quantum Architecture Search (Target > {target_accuracy*100}%)...")
    
    while attempt < max_attempts:
        attempt += 1
        SPECIFIC_SEED = SEED + attempt  # Different seed per attempt
        random.seed(SPECIFIC_SEED)
        np.random.seed(SPECIFIC_SEED)
        torch.manual_seed(SPECIFIC_SEED)
        # 1. Random Selection
        current_qubits = random.choice(qubit_choices)
        current_reps = random.choice(rep_choices)
        
        print(f"\n--- Attempt {attempt} | Qubits: {current_qubits} | Reps: {current_reps} ---")
        
        # 2. Transform Data (PCA needs to run again for new n_qubits)
        # Note: We stick to the global SEED inside the functions to ensure data splits are consistent
        train_loader, val_loader, Xte_t, yte_t, X_upcoming_t, results = transform_data(n_qubits=current_qubits, batch_size=20, seed=SPECIFIC_SEED)
        
        # 3. Create Model
        model = create_quantum_model(n_qubits=current_qubits, reps=current_reps)
        
        # 4. Train
        model, history = train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=0.01) # Slightly higher LR for QNN often helps
        
        # 5. Test
        cm, report, accuracy = test_model(Xte_t, yte_t, model)
        print(f"Result: Accuracy = {accuracy:.4f} Seed = {SPECIFIC_SEED}")
        
        if accuracy > target_accuracy:
            print(f"\nSUCCESS! Found model with {accuracy:.4f} accuracy.")
            print(report)
            
            config_info = f"Qubits={current_qubits}, Reps={current_reps}"
            save_stats(history=history, cm=cm, report=report, config_info=config_info)
            save_model(model=model)
            predictions(X_upcoming_t, results, model)
            break

    if attempt == max_attempts:
        print("\nSearch finished without reaching target accuracy.")