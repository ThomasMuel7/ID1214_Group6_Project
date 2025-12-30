import pandas as pd
import numpy as np
import random
import io 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

#Reproducibility
SEED = 54
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED) 

class EpochTimer(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epoch_start = None
            self.val_start = None
            self.train_time = []
            self.val_time = []

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time()
            self.val_start = None

        def on_test_begin(self, logs=None):
            # validation (test) begins
            self.val_start = time()

        def on_test_end(self, logs=None):
            # validation (test) ends
            if self.val_start is not None:
                self.val_time.append(time() - self.val_start)
            else:
                self.val_time.append(0.0)

        def on_epoch_end(self, epoch, logs=None):
            epoch_end = time()
            total_epoch = epoch_end - (self.epoch_start or epoch_end)
            val_dur = self.val_time[-1] if len(self.val_time) > 0 else 0.0
            train_dur = max(0.0, total_epoch - val_dur)
            self.train_time.append(train_dur)

def load_data(use_pca=False, n_components=30):   
    scores = pd.read_excel('processed_data/2025_processed_scores.xlsx')
    upcoming_scores = pd.read_excel('processed_data/2025_processed_upcoming_games.xlsx')
    
    upcoming_scores_clean = upcoming_scores.dropna()
    scores_clean = scores.dropna()
    
    results = upcoming_scores_clean[['Home', 'Visitor']].copy()

    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
    
    X_upcoming = upcoming_scores_clean[feature_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_upcoming_scaled = scaler.transform(X_upcoming)
    
    if use_pca:
        pca = PCA(n_components=n_components)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        X_upcoming_scaled = pca.transform(X_upcoming_scaled)
    
    return X_train_scaled, X_test_scaled, X_upcoming_scaled, y_train, y_test, results, len(feature_cols) if not use_pca else n_components

def create_model(input_dim):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train_scaled, y_train, epoch=500, batch_size=32):

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )

    timer = EpochTimer()
    callbacks = [early_stop, timer]

    history = model.fit(
        X_train_scaled, y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # attach per-epoch times to history for downstream saving
    history.history['train_time'] = timer.train_time
    history.history['val_time'] = timer.val_time
    return model, history

def test_model(model, X_test_scaled, y_test):
    preds = model.predict(X_test_scaled, verbose=0)
    
    y_pred = (preds > 0.5).astype(int).flatten()
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    return cm, report

def predictions(use_pca, X_predict, results, model):
    predictions_prob = model.predict(X_predict, verbose=0)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    results['Model_Home_win'] = predictions
    results['Model_home_win_probability'] = predictions_prob.flatten()
    
    if use_pca: 
        filename = './predictions/classical_pca_predictions.xlsx'
    else : 
        filename = './predictions/classical_predictions.xlsx'
    
    results.to_excel(filename, index=False)
    print(f"Predictions saved to {filename}")

def save_stats(history, cm, report, use_pca, seed=None, n_components=None, model=None):
    if use_pca:
        out_dir = "./model_stats/classical_pca"
    else:
        out_dir = "./model_stats/classical"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'report.txt')
    loss_path = os.path.join(out_dir, 'loss.png')
    acc_path = os.path.join(out_dir, 'accuracy.png')
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    time_path = os.path.join(out_dir, 'training_time.png')
    f_time = history.history.get('train_time', [])
    v_time = history.history.get('val_time', [])
    losses = history.history.get('loss', [])
    val_losses = history.history.get('val_loss', [])
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    
    # Write textual report
    with open(report_path, 'w') as f:
        f.write(str(report))
        f.write('\n')
        f.write(f'\n Total Time = {sum(f_time) + sum(v_time)} seconds\n')
        f.write(f' Random Seed = {seed}\n')
        f.write(f' Number of PCA components = {n_components if use_pca else "N/A"}\n')

    # Loss plot
    if losses or val_losses:
        plt.figure(figsize=(8, 5))
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, label='Train Loss')
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
    if acc or val_acc:
        plt.figure(figsize=(8, 5))
        if acc:
            plt.plot(range(1, len(acc) + 1), acc, label='Train Acc')
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
    t_train = history.history.get('train_time', [])
    t_val = history.history.get('val_time', [])
    if t_train or t_val:
        max_epochs = max(len(t_train), len(t_val))
        tt = t_train + [0] * (max_epochs - len(t_train))
        vt = t_val + [0] * (max_epochs - len(t_val))
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

def save_model(model, use_pca):    
    if use_pca:
        filename = './model/classical_pca.keras'
    else:
        filename = './model/classical.keras'
    model.save(filename)
    print(f"Model saved to {filename}")
    
# --- Main Execution ---
use_pca = True
n_components = 32

X_train_scaled, X_test_scaled, X_upcoming_scaled, y_train, y_test, results, number_feature_cols = load_data(use_pca=use_pca, n_components=n_components)
model = create_model(input_dim=number_feature_cols)
model, history = train_model(model, X_train_scaled, y_train, epoch=500)
cm, report = test_model(model, X_test_scaled, y_test)
save_stats(history=history, cm=cm, report=report, use_pca=use_pca, seed=SEED, n_components=n_components, model=model)
save_model(model=model, use_pca=use_pca)
predictions(use_pca, X_upcoming_scaled, results, model)
