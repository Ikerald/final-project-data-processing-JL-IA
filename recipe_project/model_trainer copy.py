import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    """Class to handle model training and evaluation"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RecipeRatingPredictor(input_size, hidden_sizes).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        
    def prepare_data(self, features, targets, test_size=0.2, batch_size=32):
        """Prepare data for training"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, targets, test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = RecipeDataset(X_train, y_train)
        test_dataset = RecipeDataset(X_test, y_test)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return X_train, X_test, y_train, y_test
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for features, targets in self.train_loader:
            features, targets = features.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        """Evaluate the model"""
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
        
        return (total_loss / len(loader), 
                np.array(predictions), 
                np.array(actuals))
    
    def train(self, features, targets, epochs=50, test_size=0.2, batch_size=32):
        """Full training process"""
        print(f"Using device: {self.device}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(
            features, targets, test_size, batch_size
        )
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_loss, predictions, actuals = self.evaluate(self.test_loader)
            
            # Save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                
                # Calculate additional metrics
                mse = np.mean((predictions - actuals) ** 2)
                mae = np.mean(np.abs(predictions - actuals))
                r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
                
                print(f"MSE: {mse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R2 Score: {r2:.4f}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
        
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

def train_models(processor, vector_type='tfidf'):
    """Train models with different vector representations"""
    try:
        if vector_type == 'tfidf':
            vectors = processor.create_tfidf_vectors()
        elif vector_type == 'word2vec':
            vectors = processor.create_word2vec_vectors()
        elif vector_type == 'bert':
            vectors = processor.create_bert_vectors()
        else:
            raise ValueError(f"Unknown vector type: {vector_type}")
        
        # Convert sparse matrix to dense if needed
        if hasattr(vectors, 'toarray'):
            vectors = vectors.toarray()
        
        # Initialize trainer
        trainer = ModelTrainer(
            input_size=vectors.shape[1],
            hidden_sizes=[256, 128, 64],
            learning_rate=0.001
        )
        
        # Train model
        train_losses, val_losses = trainer.train(
            features=vectors,
            targets=processor.df['rating'].values,
            epochs=50,
            batch_size=32
        )
        
        print(f"Training completed for {vector_type} vectors!")
        return trainer
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise