import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Tuple, Optional 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, PReLU, ReLU , Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber, LogCosh


class DataLoader:
    def __init__(self, data_folder: str = "csv"):
        self.data_folder = data_folder
        self.df = None
    
    def load_csv_file(self, file_path: str) -> pd.DataFrame:

        if not os.path.isabs(file_path) and not os.path.exists(file_path):
            csv_path = os.path.join(self.data_folder, file_path)
            if os.path.exists(csv_path):
                file_path = csv_path
            else:
                for ext in ['.csv', '.xlsx', '.xls']:
                    test_path = os.path.join(self.data_folder, file_path + ext)
                    if os.path.exists(test_path):
                        file_path = test_path
                        break
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File {file_path} not found!")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, index_col=0)
            else:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', index_col=0)
                except:
                    df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            raise FileNotFoundError(f"❌ Error reading file {file_path}: {str(e)}")
        
        print(f"📊 Number of rows: {len(df)}")
        print(f"📊 Number of columns: {len(df.columns)}\n")
        
        self.df = df
        return df
    
    def create_dataset(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 32,
                      shuffle: bool = False,
                      prefetch: bool = True,
                      repeat: bool = False) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=300, seed=7)
        
        dataset = dataset.batch(batch_size)
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        if repeat:
            dataset = dataset.repeat()
        
        return dataset


class SequenceGenerator:
    @staticmethod
    def create_sequences(data: np.ndarray, targets: np.ndarray, 
        sequence_length: int = 120) -> Tuple[np.ndarray, np.ndarray]:
        if sequence_length is None:
            sequence_length = 120 

        if len(data.shape) != 2:
            raise ValueError(f"❌ data should be two-dimensional (samples, features), but current shape: {data.shape}")
        
        if len(data) < sequence_length:
            raise ValueError(f"❌ Number of samples ({len(data)}) is less than the sequence length ({sequence_length})")
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
            if len(data.shape) == 1: 
                data = data.reshape(-1, 1)
        
        X, y = [], []
        n_samples = len(data)
        n_features = data.shape[1]
        
        for i in range(sequence_length, n_samples):
            sequence = data[i-sequence_length:i] 
            
            if sequence.shape != (sequence_length, n_features):
                sequence = sequence.reshape(sequence_length, n_features)
            
            X.append(sequence)
            y.append(targets[i])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X.shape) != 3:
            raise ValueError(f"❌ Output shape should be (samples, timesteps, features), but current shape: {X.shape}")
        
        expected_shape = (len(X), sequence_length, n_features)
        if X.shape != expected_shape:
            print(f"⚠️ Warning: shape of X ({X.shape}) is not equal to ({expected_shape}). Reshaping...")
            X = X.reshape(expected_shape)
        
        return X, y




class LSTMRegressor:
    def __init__(self, sequence_length: int = 120):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int], 
                   lstm_units: list = [88, 48],
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001) -> Sequential:
        if len(lstm_units) != 2:
            raise ValueError(f"❌ We should have exactly 2 LSTM layers, but {len(lstm_units)} layers are defined")
        
        for units in lstm_units:
            if units < 16:
                raise ValueError(f"❌ Each layer should have at least 16 neurons, but {units} neurons are defined")
        
        model = Sequential()

        model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Dropout(dropout_rate))

        model.add(Bidirectional(LSTM(88, return_sequences=True)))
        model.add(Bidirectional(LSTM(48, return_sequences=False)))

        model.add(Dense(1, activation='linear'))

        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss=LogCosh(),
            metrics=['mse', 'mae']
        )   

        
        model.add(Dense(units=1, activation='linear'))
        
        model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss=LogCosh(),
            metrics=['mse', 'mae']
        )
        



        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 20) -> dict:

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_regressor.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        data_loader = DataLoader()
        
        train_dataset = data_loader.create_dataset(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=False,
            prefetch=True,
            repeat=False
        )
        
        val_dataset = data_loader.create_dataset(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False,
            prefetch=True,
            repeat=False
        )
        
        print("🚀 Starting training model LSTM...\n")
        print(f"📦 Batch size: {batch_size}")
        print(f"🔄 Train shuffle: enabled (buffer: 300)")
        print(f"🔄 Val shuffle: disabled")
        print(f"⚡ Prefetch: enabled")
        print(f"🔁 Repeat: disabled\n")
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training model successfully\n")
        return self.history.history
    
    def predict(self, X: np.ndarray, inverse_transform: bool = True) -> np.ndarray:

        y_pred_normalized = self.model.predict(X, verbose=0)
        
        if inverse_transform:
            y_pred = self.scaler_y.inverse_transform(y_pred_normalized).flatten()
        else:
            y_pred = y_pred_normalized.flatten()
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, inverse_transform: bool = True) -> dict:

        y_pred_normalized = self.model.predict(X, verbose=0)
        evaluation_results = self.model.evaluate(X, y, verbose=0)
        
        if len(evaluation_results) == 3:
            loss, mse_normalized, mae_normalized = evaluation_results
        elif len(evaluation_results) == 2:
            loss, mae_normalized = evaluation_results
            mse_normalized = loss 
        else:
            loss = evaluation_results[0]
            mse_normalized = loss
            mae_normalized = evaluation_results[-1] if len(evaluation_results) > 1 else loss
        
        if inverse_transform:
            y_pred = self.scaler_y.inverse_transform(y_pred_normalized).flatten()
            y_actual = self.scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
            
            mse = mean_squared_error(y_actual, y_pred)
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = math.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)
        else:
            y_pred = y_pred_normalized.flatten()
            y_actual = y.flatten()
            
            mse = mse_normalized
            mae = mae_normalized
            rmse = math.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)
        
        return {
            'loss': loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': y_pred,
            'actual': y_actual
        }


class Visualizer:
    @staticmethod
    def plot_training_history(history: dict, save_path: str = 'result/training_history.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history['loss'], label='Train Loss (MSE)', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss (MSE)', linewidth=2)
        axes[0].set_title('Model Loss (MSE)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE (Mean Absolute Error)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved in '{save_path}'\n")
        plt.close()
    
    @staticmethod
    def plot_predictions(actual: np.ndarray, predicted: np.ndarray,
                        n_samples: int = 200,
                        save_path: str = 'result/predictions_vs_actual.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        n_plot = min(n_samples, len(actual))
        indices = np.arange(n_plot)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        axes[0].plot(indices, actual[:n_plot], label='Actual Price', 
                   alpha=0.7, linewidth=2, color='blue', marker='o', markersize=3)
        axes[0].plot(indices, predicted[:n_plot], label='Predicted Price', 
                   alpha=0.7, linewidth=2, color='red', marker='s', markersize=3)
        axes[0].set_title('Price Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sample', fontsize=12)
        axes[0].set_ylabel('Price', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        error = predicted[:n_plot] - actual[:n_plot]
        axes[1].plot(indices, error, label='Prediction Error', 
                    alpha=0.7, linewidth=2, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
        axes[1].set_title('Prediction Error (Predicted - Actual)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sample', fontsize=12)
        axes[1].set_ylabel('Error', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Predictions plot saved in '{save_path}'\n")
        plt.close()


def create_price_target(df: pd.DataFrame, target_col: str = 'Close') -> pd.Series:

    if target_col not in df.columns:
        raise KeyError(f"❌ Column '{target_col}' not found!")
    
    price_target = df[target_col].copy()
    
    return price_target


def main():
    print("=" * 60)
    print("🏦 Price Prediction with LSTM Neural Network (Regression)")
    print("=" * 60)
    print()
    
    
    csv_file_path = input("📁 Enter the CSV file path: ").strip()
    
    if not csv_file_path:
        print("❌ CSV file path not entered!")
        sys.exit(1)
    
    SEQUENCE_LENGTH = 120
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    BATCH_SIZE = 32
    
    print("\n📁 Step 1: Load data")
    print("-" * 60)
    data_loader = DataLoader()
    df = data_loader.load_csv_file(csv_file_path)
    print()
    
    print("🔧 Step 2: Prepare features and target")
    print("-" * 60)
    exclude_cols = ['time', 'date']
    
    if 'Close' in df.columns:
        target_col = 'Close'
    elif 'close' in df.columns:
        target_col = 'close'
    elif 'Price' in df.columns:
        target_col = 'Price'
    elif 'price' in df.columns:
        target_col = 'price'
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            print(f"⚠️ Column 'Close' not found. Using column '{target_col}'")
        else:
            raise ValueError("❌ No numeric column found for target creation!")
    
    feature_cols = [col for col in df.columns 
                   if col.lower() not in exclude_cols and col != target_col]
    
    y = create_price_target(df, target_col)
    
    X = df[feature_cols].values
    
    print(f"📊 shape of features: {X.shape}")
    print(f"📊 shape of target: {y.shape}")
    print(f"📋 number of features: {len(feature_cols)}")
    print(f"📋 target column: '{target_col}'")
    print(f"📋 price range: [{y.min():.2f}, {y.max():.2f}]\n")
    
    print("📏 Step 3: Normalize data")
    print("-" * 60)
    regressor = LSTMRegressor(sequence_length=SEQUENCE_LENGTH)
    
    X_scaled = regressor.scaler_X.fit_transform(X)
    
    y_values = y.values.reshape(-1, 1)
    y_scaled = regressor.scaler_y.fit_transform(y_values)
    y_scaled = y_scaled.flatten()
    
    print(f"✅ Normalization done")
    print(f"📊 shape of normalized X: {X_scaled.shape}")
    print(f"📊 shape of normalized y: {y_scaled.shape}")
    print(f"📊 normalized y range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]\n")
    
    print("🔄 Step 4: Convert to 3D sequences")
    print("-" * 60)
    seq_gen = SequenceGenerator()
    
    X_seq, y_seq = seq_gen.create_sequences(
        X_scaled, y_scaled, SEQUENCE_LENGTH
    )
    
    print(f"\n📊 final shape of sequences:")
    print(f"   - X_seq: {X_seq.shape} (samples, timesteps, features)")
    print(f"   - y_seq: {y_seq.shape} (samples,)\n")
    
    print("✂️ Step 5: Split data into train, val, test")
    print("-" * 60)
    n_total_seq = len(X_seq)
    n_train = int(n_total_seq * TRAIN_SIZE)
    n_val = int(n_total_seq * VAL_SIZE)
    n_test = n_total_seq - n_train - n_val
    
    X_train_seq = X_seq[:n_train]
    y_train_seq = y_seq[:n_train]
    
    X_val_seq = X_seq[n_train:n_train+n_val]
    y_val_seq = y_seq[n_train:n_train+n_val]
    
    X_test_seq = X_seq[n_train+n_val:]
    y_test_seq = y_seq[n_train+n_val:]
    
    print(f"📊 training sequences: {X_train_seq.shape} ({len(X_train_seq)/n_total_seq*100:.1f}%)")
    print(f"📊 validation sequences: {X_val_seq.shape} ({len(X_val_seq)/n_total_seq*100:.1f}%)")
    print(f"📊 test sequences: {X_test_seq.shape} ({len(X_test_seq)/n_total_seq*100:.1f}%)\n")
    
    print("🏗️ Step 6: Build LSTM model")
    print("-" * 60)
    input_shape = (SEQUENCE_LENGTH, X_train_seq.shape[2])
    
    model = regressor.build_model(
        input_shape=input_shape,
        lstm_units=[88, 48],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    print("📋 Model architecture:")
    model.summary()
    print()
    
    print("🎓 Step 7: Train model")
    print("-" * 60)
    history = regressor.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        epochs=100,
        batch_size=BATCH_SIZE,
        patience=20
    )
    
    print("🧪 Step 8: Evaluate model")
    print("-" * 60)
    test_results = regressor.evaluate(X_test_seq, y_test_seq, inverse_transform=True)
    
    print("📈 Evaluation results on test data (in original price scale):")
    print(f"   Loss (MSE): {test_results['mse']:.6f}")
    print(f"   MAE: {test_results['mae']:.4f}")
    print(f"   RMSE: {test_results['rmse']:.4f}")
    print(f"   R2-Score: {test_results['r2_score']:.4f}\n")
    
    print("📊 Step 9: Plot results")
    print("-" * 60)
    visualizer = Visualizer()
    
    visualizer.plot_training_history(history, 'result/training_history.png')
    
    visualizer.plot_predictions(
        test_results['actual'], 
        test_results['predictions'],
        n_samples=200,
        save_path='result/predictions_vs_actual.png'
    )
    
    print("\n" + "=" * 60)
    print("✅ All steps completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

