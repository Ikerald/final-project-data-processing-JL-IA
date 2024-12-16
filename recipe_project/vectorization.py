"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


def bert_vectorization(device, df):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model = model.to(device)

    bert_vectors = []
    batch_size = 32

    # Process in batches to handle memory constraints
    for i in range(0, len(df), batch_size):
        batch_texts = df["directions_post"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embeddings as document representation
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            bert_vectors.extend(batch_embeddings)
    vectors = np.array(bert_vectors)
    return vectors


def TF_vectorization(df):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_vectors = tfidf.fit_transform(df["directions_post"])
    print(f"Created TF-IDF vectors with shape: {tfidf_vectors.shape}")
    print(tfidf_vectors)
    return tfidf_vectors.toarray()


def Word2Vec_vectorization(df):
    sentences = [text.split() for text in df["directions_post"]]
    vector_size = 100

    w2v_model = Word2Vec(
        sentences, vector_size=vector_size, window=5, min_count=1, workers=4
    )

    # Create document vectors by averaging word vectors
    doc_vectors = []
    for sentence in sentences:
        word_vectors = [w2v_model.wv[word] for word in sentence if word in w2v_model.wv]
        if word_vectors:
            doc_vectors.append(np.mean(word_vectors, axis=0))
        else:
            doc_vectors.append(np.zeros(vector_size))

    W2V_vectors = np.array(doc_vectors)
    print(f"Created Word2Vec vectors with shape: {W2V_vectors.shape}")
    print(W2V_vectors)
    return W2V_vectors


def NeuralNetwork(vectors, df):
    features_scaled = StandardScaler().fit_transform(vectors)
    targets = df["rating_pre"]
    print("Scaled features shape:", features_scaled.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=42
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
    y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)
    print("X_train tensor shape:", X_train_t.shape)
    print("y_train tensor shape:", y_train_t.shape)

    # Now create the model with the correct input size
    input_size = X_train.shape[1]  # This will get the correct dimension
    print("Input size:", input_size)

    model_ML = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),  # Add dropout
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),  # Add dropout
        nn.Linear(64, 1),
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model_ML.parameters(), lr=0.001, weight_decay=0.01
    )  # Add weight_decay

    # Lists to store metrics
    train_losses = []
    epochs_list = []
    train_r2_scores = []
    test_r2_scores = []  # Added test R² tracking

    # Training loop
    epochs = 100
    batch_size = 32

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i : i + batch_size]
            batch_y = y_train_t[i : i + batch_size]

            outputs = model_ML(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches

        # Calculate R² scores for both train and test
        model_ML.eval()
        with torch.no_grad():
            train_predictions = model_ML(X_train_t)
            test_predictions = model_ML(X_test_t)
            train_r2 = r2_score(y_train_t.numpy(), train_predictions.numpy())
            test_r2 = r2_score(y_test_t.numpy(), test_predictions.numpy())
        model_ML.train()

        # Store metrics
        train_losses.append(avg_loss)
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)  # Store test R²
        epochs_list.append(epoch + 1)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}"
            )

    # Final evaluation
    model_ML.eval()
    with torch.no_grad():
        # Get predictions for training set
        train_predictions = model_ML(X_train_t)
        train_mse = mean_squared_error(y_train_t.numpy(), train_predictions.numpy())
        train_r2 = r2_score(y_train_t.numpy(), train_predictions.numpy())

        # Get predictions for test set
        test_predictions = model_ML(X_test_t)
        test_mse = mean_squared_error(y_test_t.numpy(), test_predictions.numpy())
        test_r2 = r2_score(y_test_t.numpy(), test_predictions.numpy())

    print("\nTraining Set Metrics:")
    print(f"MSE: {train_mse:.4f}")
    print(f"R²: {train_r2:.4f}")

    print("\nTest Set Metrics:")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")

    # Visualization code
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Training Loss Over Time
    ax1.plot(epochs_list, train_losses)
    ax1.set_title("Training Loss Over Time")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot 2: R² Score Over Time
    ax2.plot(epochs_list, train_r2_scores, label="Train R²")
    ax2.plot(epochs_list, test_r2_scores, label="Test R²")
    ax2.set_title("R² Score Over Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R² Score")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Predictions vs Actuals (Scatter)
    ax3.scatter(y_test_t.numpy(), test_predictions.numpy(), alpha=0.5)
    ax3.plot(
        [y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], "r--", lw=2
    )
    ax3.set_title("Predictions vs Actual Values")
    ax3.set_xlabel("Actual Values")
    ax3.set_ylabel("Predicted Values")
    ax3.grid(True)

    # Plot 4: Distribution Plot
    ax4.hist(
        y_test_t.numpy().flatten(), bins=30, alpha=0.5, label="Actual", density=True
    )
    ax4.hist(
        test_predictions.numpy().flatten(),
        bins=30,
        alpha=0.5,
        label="Predicted",
        density=True,
    )
    ax4.set_title("Distribution of Actual vs Predicted Values")
    ax4.set_xlabel("Values")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_train_r2 = lr.score(X_train, y_train)
    lr_test_r2 = lr.score(X_test, y_test)
    print(f"Linear Regression R² - Train: {lr_train_r2:.4f}, Test: {lr_test_r2:.4f}")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get predictions
    rf_train_predictions = rf_model.predict(X_train)
    rf_test_predictions = rf_model.predict(X_test)

    # Calculate metrics
    rf_train_r2 = r2_score(y_train, rf_train_predictions)
    rf_test_r2 = r2_score(y_test, rf_test_predictions)
    rf_train_mse = mean_squared_error(y_train, rf_train_predictions)
    rf_test_mse = mean_squared_error(y_test, rf_test_predictions)

    print("\nRandom Forest Metrics:")
    print("Training Set:")
    print(f"MSE: {rf_train_mse:.4f}")
    print(f"R²: {rf_train_r2:.4f}")
    print("\nTest Set:")
    print(f"MSE: {rf_test_mse:.4f}")
    print(f"R²: {rf_test_r2:.4f}")

    # Single plot for Random Forest
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_test_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.title("Random Forest: Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()
