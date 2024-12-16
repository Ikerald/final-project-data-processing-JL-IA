import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

def check_gpu():
    """Check GPU availability and print information"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nGPU Information:")
        print(f"- GPU Available: Yes")
        print(f"- GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"- Number of GPUs: {torch.cuda.device_count()}")
        print(f"- CUDA Version: {torch.version.cuda}")
        
        # Print memory information
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
        memory_cached = torch.cuda.memory_reserved(0) / 1024**2
        print(f"- GPU Memory Allocated: {memory_allocated:.2f} MB")
        print(f"- GPU Memory Cached: {memory_cached:.2f} MB")
    else:
        device = torch.device("cpu")
        print("\nGPU not available, using CPU instead")
    
    return device

class RecipeDataset(Dataset):
    """Custom Dataset for recipe data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class RecipeRatingPredictor(nn.Module):
    """Neural network for predicting recipe ratings"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super(RecipeRatingPredictor, self).__init__()
        
        # Create layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], learning_rate=0.001):
        self.device = check_gpu()
        self.model = RecipeRatingPredictor(input_size, hidden_sizes).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
    
    def prepare_data(self, features, targets, test_size=0.2, batch_size=32):
        try:
            features_scaled = self.scaler.fit_transform(features)
            
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, targets, test_size=test_size, random_state=42
            )
            
            train_dataset = RecipeDataset(X_train, y_train)
            test_dataset = RecipeDataset(X_test, y_test)
            
            num_workers = 4 if self.device.type == 'cuda' else 0
            pin_memory = True if self.device.type == 'cuda' else False
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for features, targets in self.train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for features, targets in loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), targets)
                total_loss += loss.item()
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        return total_loss / len(loader), np.array(predictions), np.array(actuals)

    def plot_prediction_analysis(self, predictions, actuals, title="Prediction Analysis"):
        plt.figure(figsize=(15, 10))
        
        # Predicted vs Actual
        plt.subplot(2, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Predicted vs Actual')
        
        # Residuals
        residuals = predictions - actuals
        plt.subplot(2, 2, 2)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Ratings')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Error Distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=50)
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        
        # Metrics
        plt.subplot(2, 2, 4)
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        metrics_text = f'MSE: {mse:.4f}\nR2 Score: {r2:.4f}'
        plt.text(0.5, 0.5, metrics_text, ha='center', va='center')
        plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def train(self, features, targets, epochs=50, test_size=0.2, batch_size=32):
        try:
            print(f"\nUsing device: {self.device}")
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
            
            X_train, X_test, y_train, y_test = self.prepare_data(
                features, targets, test_size, batch_size
            )
            
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            print("\nStarting training...")
            for epoch in range(epochs):
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                train_loss = self.train_epoch()
                val_loss, predictions, actuals = self.evaluate(self.test_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    print(f"Train Loss: {train_loss:.4f}")
                    print(f"Val Loss: {val_loss:.4f}")
                    
                    mse = mean_squared_error(actuals, predictions)
                    r2 = r2_score(actuals, predictions)
                    print(f"MSE: {mse:.4f}")
                    print(f"R2 Score: {r2:.4f}")
                    
                    if self.device.type == 'cuda':
                        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                        print(f"GPU Memory Used: {memory_allocated:.2f} MB")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f'best_model_{epoch}.pth')
            
            # Final evaluation and visualization
            _, final_predictions, final_actuals = self.evaluate(self.test_loader)
            self.plot_prediction_analysis(final_predictions, final_actuals)
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.show()
            
            return train_losses, val_losses
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            raise

def train_models(processor, vector_type='tfidf'):
    try:
        if torch.cuda.is_available():
            print("\nInitial GPU Memory Status:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        
        if vector_type == 'tfidf':
            vectors = processor.create_tfidf_vectors()
        elif vector_type == 'word2vec':
            vectors = processor.create_word2vec_vectors()
        elif vector_type == 'bert':
            vectors = processor.create_bert_vectors()
        else:
            raise ValueError(f"Unknown vector type: {vector_type}")
        
        if hasattr(vectors, 'toarray'):
            vectors = vectors.toarray()
        
        trainer = ModelTrainer(
            input_size=vectors.shape[1],
            hidden_sizes=[256, 128, 64],
            learning_rate=0.001
        )
        
        train_losses, val_losses = trainer.train(
            features=vectors,
            targets=processor.df['rating'].values,
            epochs=50,
            batch_size=64
        )
        
        print(f"\nTraining completed for {vector_type} vectors!")
        return trainer
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise