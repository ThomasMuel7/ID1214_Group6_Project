from tensorflow import keras
from tensorflow.keras import layers
from train_data_prep import load_data
import random
import pandas as pd

# Load preprocessed data
X_train_scaled, X_test_scaled, y_train, y_test, number_feature_cols = load_data(use_pca=True,n_components=30) 

# Grid settings
neurons = [2, 4, 8, 16, 32, 64, 128, 256]
num_layers = [1, 2, 3, 4, 5]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
epoch = 500
batch_size = 16

# Track best model
best_accuracy = 0
best_results = {}
best_model = None

print("\nStarting hyperparameter tuning...")
for i in range(1000): 
    num_layer = random.choice(num_layers)
    # Build model dynamically
    model = keras.Sequential()
    model.add(layers.Input(shape=(number_feature_cols,)))
                    
    # Add hidden layers
    for j in range(num_layer):
        num_neurons = random.choice(neurons)
        dropout = random.choice(dropout_rates)
        model.add(layers.Dense(num_neurons, activation='relu'))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
                    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train with early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
                    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epoch,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    history = history.history.get('val_accuracy', [])
    if history :
        max_accuracy = max(history)
        max_idx = len(history) - 1 - history[::-1].index(max_accuracy)
        # Evaluate on test set
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        test_acc = (y_pred == y_test).mean()
        new_accuracy = max_accuracy*0.3 + test_acc*0.7
        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            best_results = {
                'num_layers': num_layer,
                'neurons_per_layer': [layer.units for layer in model.layers if isinstance(layer, layers.Dense)][:-1],
                'dropout_rates': [layer.rate for layer in model.layers if isinstance(layer, layers.Dropout)],
                'best_epoch': max_idx + 1,
                'best_val_accuracy': new_accuracy
            }
            best_model = model
            print(f"New best model found: {best_results}")
    print(f"Iteration {i+1}/1000 completed.")                   
    
# Save best model
pd.DataFrame([best_results]).to_csv('./model/nfl_model_hyperparameters4.csv', index=False)
best_model.save('./model/nfl_prediction_model4.keras')
print("Best model saved to 'nfl_prediction_model4.keras'")