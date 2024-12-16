import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RecipeVisualizer:
    def __init__(self, processor):
        """Initialize visualizer with RecipeProcessor instance"""
        self.processor = processor
        self.df = processor.df
        # Set style for matplotlib
        plt.style.use('seaborn')
        
    def plot_rating_distribution(self):
        """Plot the distribution of recipe ratings"""
        plt.figure(figsize=(12, 6))
        
        # Create histogram with KDE
        sns.histplot(data=self.df, x='rating', bins=50, kde=True)
        plt.title('Distribution of Recipe Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        
        # Add mean and median lines
        plt.axvline(self.df['rating'].mean(), color='r', linestyle='--', label=f'Mean: {self.df["rating"].mean():.2f}')
        plt.axvline(self.df['rating'].median(), color='g', linestyle='--', label=f'Median: {self.df["rating"].median():.2f}')
        plt.legend()
        plt.show()

    def plot_wordcloud(self, column='processed_text'):
        """Generate wordcloud from specified text column"""
        print(f"Generating wordcloud from {column}...")
        
        # Combine all text
        text = ' '.join(self.df[column].astype(str))
        
        # Create and generate wordcloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100).generate(text)
        
        # Display wordcloud
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud from {column}')
        plt.show()

    def plot_category_analysis(self, top_n=20):
        """Plot detailed category analysis"""
        # Explode categories and calculate statistics
        categories_df = self.df.explode('categories')
        category_stats = categories_df.groupby('categories').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        
        category_stats.columns = ['count', 'mean_rating', 'std_rating']
        category_stats = category_stats.sort_values('count', ascending=False)
        
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Top Categories by Recipe Count',
                                         'Average Rating by Category'),
                           vertical_spacing=0.12)
        
        # Plot 1: Top categories by count
        fig.add_trace(
            go.Bar(x=category_stats.head(top_n).index,
                  y=category_stats.head(top_n)['count'],
                  name='Recipe Count'),
            row=1, col=1
        )
        
        # Plot 2: Average rating by category
        fig.add_trace(
            go.Bar(x=category_stats.head(top_n).index,
                  y=category_stats.head(top_n)['mean_rating'],
                  error_y=dict(type='data',
                              array=category_stats.head(top_n)['std_rating']),
                  name='Average Rating'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(height=800, width=1200,
                         title_text='Category Analysis',
                         showlegend=False)
        fig.show()

    def plot_prediction_analysis(self, predictions, actuals, model_name='Model'):
        """Plot analysis of model predictions"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Scatter plot of predicted vs actual values
        plt.subplot(2, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Predicted vs Actual Ratings')
        
        # 2. Residual plot
        residuals = predictions - actuals
        plt.subplot(2, 2, 2)
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Ratings')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 3. Residual distribution
        plt.subplot(2, 2, 3)
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual Value')
        plt.ylabel('Count')
        plt.title('Residual Distribution')
        
        # 4. Error metrics
        plt.subplot(2, 2, 4)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        mae = np.mean(np.abs(residuals))
        
        metrics_text = f'Model Performance Metrics:\n\n' \
                      f'MSE: {mse:.4f}\n' \
                      f'RMSE: {rmse:.4f}\n' \
                      f'MAE: {mae:.4f}\n' \
                      f'R²: {r2:.4f}'
        
        plt.text(0.5, 0.5, metrics_text, ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        
        # Adjust layout and display
        plt.suptitle(f'{model_name} Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_training_history(self, train_losses, val_losses, model_name='Model'):
        """Plot training and validation loss history"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        plt.title(f'{model_name} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance for applicable models"""
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 6))
            plt.title('Top Feature Importances')
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("This model doesn't provide feature importances.")

    def plot_interactive_predictions(self, predictions, actuals):
        """Create interactive scatter plot of predictions"""
        df_plot = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'Error': np.abs(predictions - actuals)
        })
        
        fig = px.scatter(df_plot, x='Actual', y='Predicted',
                        color='Error', title='Interactive Prediction Analysis',
                        labels={'Actual': 'Actual Ratings',
                               'Predicted': 'Predicted Ratings',
                               'Error': 'Absolute Error'},
                        color_continuous_scale='Viridis')
        
        # Add perfect prediction line
        fig.add_trace(
            go.Scatter(x=[df_plot['Actual'].min(), df_plot['Actual'].max()],
                      y=[df_plot['Actual'].min(), df_plot['Actual'].max()],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash'))
        )
        
        fig.show()

def update_training_plot(fig, train_losses, val_losses):
    """Update training plot for real-time visualization"""
    plt.clf()
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)