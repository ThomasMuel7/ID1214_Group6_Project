import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

#Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_data(use_pca=False, n_components=30):   
    # Load processed scores with team statistics
    scores = pd.read_excel('processed_data/2025_processed_scores.xlsx')
    upcoming_scores = pd.read_excel('processed_data/2025_processed_upcoming_games.xlsx')
    
    upcoming_scores_clean = upcoming_scores.dropna()
    scores_clean = scores.dropna()
    
    # Create results dataframe specifically from the CLEAN data
    results = upcoming_scores_clean[['Home', 'Visitor']].copy()

    # Extract features (all _diff columns) and target (Home_win)
    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
    
    X_upcoming = upcoming_scores_clean[feature_cols].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_upcoming_scaled = scaler.transform(X_upcoming)
    
    # Apply PCA if requested
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
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
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
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epoch,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history

def test_model(model, X_test_scaled, y_test):
    preds = model.predict(X_test_scaled, verbose=0)
    
    y_pred = (preds > 0.5).astype(int).flatten()
    
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

def predictions(use_pca, X_predict, results, model):
    predictions_prob = model.predict(X_predict, verbose=0)
    predictions = (predictions_prob > 0.5).astype(int).flatten()

    results['Model_Home_win'] = predictions
    results['Model_home_win_probability'] = predictions_prob.flatten()
    
    if use_pca: 
        filename = './predictions/classical_pca_predictions.xlsx'
    else : 
        filename = './predictions/classical_predictions.xlsx'
    
    # Save predictions
    results.to_excel(filename, index=False)
    print(f"Predictions saved to {filename}")
    print(results.head())

def plot_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    
# --- Main Execution ---
use_pca = False
n_components = 64

X_train_scaled, X_test_scaled, X_upcoming_scaled, y_train, y_test, results, number_feature_cols = load_data(use_pca=use_pca, n_components=n_components) 
model = create_model(input_dim=number_feature_cols)
model, history = train_model(model, X_train_scaled, y_train, epoch=500)
test_model(model, X_test_scaled, y_test)
predictions(use_pca, X_upcoming_scaled, results, model)
plot_history(history)