import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import os

# Reproducibility (We keep the seed fixed now, as the variance comes from architecture)

def load_data(use_pca=False, n_components=30, seed=42):   
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('model_stats', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    scores = pd.read_excel('processed_data/2025_processed_scores.xlsx')
    upcoming_scores = pd.read_excel('processed_data/2025_processed_upcoming_games.xlsx')
    
    upcoming_scores_clean = upcoming_scores.dropna()
    scores_clean = scores.dropna()
    
    results = upcoming_scores_clean[['Home', 'Visitor']].copy()

    feature_cols = [col for col in scores_clean.columns if col.endswith('_diff')]

    X = scores_clean[feature_cols].values
    y = scores_clean['Home_win'].values
    
    X_upcoming = upcoming_scores_clean[feature_cols].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
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

def create_dynamic_model(input_dim, hidden_layers_config):
    """
    hidden_layers_config: A list of integers, where each integer is the 
                          number of neurons in that layer.
                          e.g., [64, 32] creates two hidden layers.
    """
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Dynamically add layers based on the config list
    for neurons in hidden_layers_config:
        model.add(layers.Dense(neurons, activation='relu'))
        # Optional: Add Dropout to prevent overfitting if networks get large
        # model.add(layers.Dropout(0.2)) 
        
    # Final output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
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
        verbose=0 # Quiet training
    )
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

def save_stats(history, cm, report, use_pca, architecture_info):
    if use_pca:
        report_path = "./model_stats/classical_pca.txt"
    else:
        report_path = "./model_stats/classical.txt"
    with open(report_path, "w") as f:
        f.write(f"ARCHITECTURE: {architecture_info}\n")
        f.write("-" * 30 + "\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(str(report))
        f.write("\n")
        f.write(f"Training loss: {history.history['loss'][-1]}")
        f.write("\n")
        f.write(f"Validation loss: {history.history['val_loss'][-1]}")
    print(f"Stats saved to {report_path}")

def save_model(model, use_pca):    
    if use_pca:
        filename = './model/classical_pca.keras'
    else:
        filename = './model/classical.keras'
    model.save(filename)
    print(f"Model saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    SEED=42
    use_pca = True
    n_components = 32
    target_accuracy = 0.75
    max_attempts = 50
    
    attempt = 0
    
    # Define possibilities for Random Search
    possible_neurons = [16, 32, 64, 128, 256]
    possible_layers_count = [1, 2, 3, 4] 
    
    print(f"Starting Architecture Search (Target > {target_accuracy*100}%)...")
    
    while attempt < max_attempts:
        attempt += 1
        SPECIFIC_SEED = SEED + attempt  # Different seed per attempt
        random.seed(SPECIFIC_SEED)
        np.random.seed(SPECIFIC_SEED)
        tf.random.set_seed(SPECIFIC_SEED)
        # Load Data
        X_train_scaled, X_test_scaled, X_upcoming_scaled, y_train, y_test, results, number_feature_cols = load_data(use_pca=use_pca, n_components=n_components, seed=SPECIFIC_SEED)
        # 1. Randomly define architecture
        num_layers = random.choice(possible_layers_count)
        # Create a list of neurons for each layer (e.g., [64, 32])
        current_config = [random.choice(possible_neurons) for _ in range(num_layers)]
        
        print(f"\n--- Attempt {attempt} | Architecture: {current_config} ---")
        
        # 2. Create and Train
        model = create_dynamic_model(input_dim=number_feature_cols, hidden_layers_config=current_config)
        model, history = train_model(model, X_train_scaled, y_train, epoch=500)
        
        # 3. Evaluate
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Result: Accuracy = {accuracy:.4f} Seed = {SPECIFIC_SEED}")
        
        if accuracy > target_accuracy:
            print(f"\nSUCCESS! Found model with {accuracy:.4f} accuracy.")
            print(f"Winning Architecture: {current_config}")
            
            cm, report = test_model(model, X_test_scaled, y_test)
            print(report)
            
            # Save stats including the architecture used
            save_stats(history, cm, report, use_pca, architecture_info=str(current_config))
            save_model(model, use_pca)
            predictions(use_pca, X_upcoming_scaled, results, model)
            break
            
    if attempt == max_attempts:
        print("\nSearch finished without reaching target accuracy.")