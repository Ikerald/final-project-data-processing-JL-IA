"""
Authors: Iker Aldasoro
         Jon Lejardi

Date: 16/12/2024
"""

# vectorization.py

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import torch
import pandas
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def bert_vectorization(device: torch.device, df: pandas.core.frame.DataFrame):
    """Creates the BERT vectors needed for the Regression models.

    Args:
        device (torch.device): Device where the model will be runned.
        df (pandas.core.frame.DataFrame): Dataframe to be analyzed.

    Returns:
        numpy.ndarray: Final vectors te be used in the regression models.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model = model.to(device)

    bert_vectors = []
    batch_size = 32

    # Process in batches to handle memory constraints
    for i in range(0, len(df), batch_size):
        batch_texts = df["directions_pre"].iloc[i : i + batch_size].tolist()
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
    print("Created BERT vectors:")
    print(vectors)
    print("")
    return vectors


def TF_vectorization(df: pandas.core.frame.DataFrame):
    """Creates the TF-IDF vectors needed for the regression models.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe to be analyzed.

    Returns:
        numpy.ndarray: Final vectors te be used in the regression models.
    """
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_vectors = tfidf.fit_transform(df["directions_post"])
    print(f"Created TF-IDF vectors with shape: {tfidf_vectors.shape}")
    print(tfidf_vectors)
    print("")
    return tfidf_vectors.toarray()


def Word2Vec_vectorization(df: pandas.core.frame.DataFrame):
    """Creates the Word2Vec vectors needed for the regression models.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe to be analyzed.

    Returns:
        numpy.ndarray: Final vectors te be used in the regression models.
    """
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


def roberta_vectorization(device: torch.device, df: pandas.core.frame.DataFrame):
    """Creates the BERT vectors needed for the Regression models.

    Args:
        device (torch.device): Device where the model will be runned.
        df (pandas.core.frame.DataFrame): Dataframe to be analyzed.

    Returns:
        numpy.ndarray: Final vectors te be used in the regression models.
    """
    batch_size = 32

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=1
    )
    model = model.to(device)

    robert_vectors = []  # Initialize predictions list

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_texts = df["directions_pre"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Get regression predictions
            batch_predictions = outputs.logits.squeeze().cpu().numpy()
            robert_vectors.extend(batch_predictions)

    # Convert to numpy array if needed
    robert_vectors = np.array(robert_vectors)
    print(f"Created BERT vectors with shape: {robert_vectors.shape}")

    return robert_vectors.reshape(-1, 1)


def NeuralNetwork(vectors, df: pandas.core.frame.DataFrame):
    """Creates the Neural network and visualized the result. Also creates a linear
    regression and random forest model and visualizes it. Includes cross-validation
    evaluation.

    Args:
        vectors (_type_): Vectors (Word2Vec, TF-IDF or BERT) created after the vectorization step.
        df (pandas.core.frame.DataFrame): Dataframe to be analyzed.
    """
    features_scaled = StandardScaler().fit_transform(vectors)
    targets = df["rating_pre"]
    print("Scaled features shape:", features_scaled.shape)

    # Part 1: Cross-validation evaluation
    print("\n=== Cross-validation Evaluation ===")
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    nn_cv_scores = []
    lr_cv_scores = []
    rf_cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(features_scaled)):
        print(f"\nFold {fold + 1}/{k_folds}")

        # Split data for this fold
        X_train_cv = features_scaled[train_idx]
        X_val_cv = features_scaled[val_idx]
        y_train_cv = targets.iloc[train_idx]
        y_val_cv = targets.iloc[val_idx]

        # Convert to tensors for neural network
        X_train_t_cv = torch.FloatTensor(X_train_cv)
        X_val_t_cv = torch.FloatTensor(X_val_cv)
        y_train_t_cv = torch.FloatTensor(y_train_cv.values).reshape(-1, 1)
        y_val_t_cv = torch.FloatTensor(y_val_cv.values).reshape(-1, 1)

        # Neural Network for this fold
        input_size = X_train_cv.shape[1]
        model_cv = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_cv.parameters(), lr=0.001, weight_decay=0.01)

        # Training loop for cross-validation
        for epoch in range(100):
            model_cv.train()
            for i in range(0, len(X_train_t_cv), 32):
                batch_X = X_train_t_cv[i : i + 32]
                batch_y = y_train_t_cv[i : i + 32]

                outputs = model_cv(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model_cv.eval()
        with torch.no_grad():
            val_predictions = model_cv(X_val_t_cv)
            nn_fold_score = r2_score(y_val_t_cv.numpy(), val_predictions.numpy())
        nn_cv_scores.append(nn_fold_score)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_cv, y_train_cv)
        lr_fold_score = r2_score(y_val_cv, lr.predict(X_val_cv))
        lr_cv_scores.append(lr_fold_score)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=20, random_state=42)
        rf.fit(X_train_cv, y_train_cv)
        rf_fold_score = r2_score(y_val_cv, rf.predict(X_val_cv))
        rf_cv_scores.append(rf_fold_score)

        print(f"Fold {fold + 1} R² scores:")
        print(f"Neural Network: {nn_fold_score:.4f}")
        print(f"Linear Regression: {lr_fold_score:.4f}")
        print(f"Random Forest: {rf_fold_score:.4f}")

    # Print cross-validation results
    print("\nCross-validation Results (mean ± std):")
    print(
        f"Neural Network R²: {np.mean(nn_cv_scores):.4f} ± {np.std(nn_cv_scores):.4f}"
    )
    print(
        f"Linear Regression R²: {np.mean(lr_cv_scores):.4f} ± {np.std(lr_cv_scores):.4f}"
    )
    print(f"Random Forest R²: {np.mean(rf_cv_scores):.4f} ± {np.std(rf_cv_scores):.4f}")

    # Visualize cross-validation results
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        [nn_cv_scores, lr_cv_scores, rf_cv_scores],
        labels=["Neural Network", "Linear Regression", "Random Forest"],
    )
    plt.title("Cross-validation R² Scores Across Models")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.show()

    # Part 2: Original training and visualization
    print("\n=== Detailed Model Training and Visualization ===")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
    y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Create the model
    input_size = X_train.shape[1]
    model_ML = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_ML.parameters(), lr=0.001, weight_decay=0.01)

    # Lists to store metrics
    train_losses = []
    epochs_list = []
    train_r2_scores = []
    test_r2_scores = []

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

        avg_loss = total_loss / num_batches

        # Calculate R² scores
        model_ML.eval()
        with torch.no_grad():
            train_predictions = model_ML(X_train_t)
            test_predictions = model_ML(X_test_t)
            train_r2 = r2_score(y_train_t.numpy(), train_predictions.numpy())
            test_r2 = r2_score(y_test_t.numpy(), test_predictions.numpy())
        model_ML.train()

        train_losses.append(avg_loss)
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)
        epochs_list.append(epoch + 1)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}"
            )

    # Final evaluation
    model_ML.eval()
    with torch.no_grad():
        train_predictions = model_ML(X_train_t)
        train_mse = mean_squared_error(y_train_t.numpy(), train_predictions.numpy())
        train_r2 = r2_score(y_train_t.numpy(), train_predictions.numpy())

        test_predictions = model_ML(X_test_t)
        test_mse = mean_squared_error(y_test_t.numpy(), test_predictions.numpy())
        test_r2 = r2_score(y_test_t.numpy(), test_predictions.numpy())

    print("\nFinal Training Set Metrics:")
    print(f"MSE: {train_mse:.4f}")
    print(f"R²: {train_r2:.4f}")

    print("\nFinal Test Set Metrics:")
    print(f"MSE: {test_mse:.4f}")
    print(f"R²: {test_r2:.4f}")

    # Visualization of training results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Training Loss
    ax1.plot(epochs_list, train_losses)
    ax1.set_title("Training Loss Over Time")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # R² Score
    ax2.plot(epochs_list, train_r2_scores, label="Train R²")
    ax2.plot(epochs_list, test_r2_scores, label="Test R²")
    ax2.set_title("R² Score Over Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R² Score")
    ax2.legend()
    ax2.grid(True)

    # Predictions vs Actuals
    ax3.scatter(y_test_t.numpy(), test_predictions.numpy(), alpha=0.5)
    ax3.plot(
        [y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], "r--", lw=2
    )
    ax3.set_title("Predictions vs Actual Values")
    ax3.set_xlabel("Actual Values")
    ax3.set_ylabel("Predicted Values")
    ax3.grid(True)

    # Distribution Plot
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

    # Additional models evaluation
    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_train_r2 = lr.score(X_train, y_train)
    lr_test_r2 = lr.score(X_test, y_test)
    print(f"\nLinear Regression R² - Train: {lr_train_r2:.4f}, Test: {lr_test_r2:.4f}")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_train_predictions = rf_model.predict(X_train)
    rf_test_predictions = rf_model.predict(X_test)

    rf_train_r2 = r2_score(y_train, rf_train_predictions)
    rf_test_r2 = r2_score(y_test, rf_test_predictions)
    rf_train_mse = mean_squared_error(y_train, rf_train_predictions)
    rf_test_mse = mean_squared_error(y_test, rf_test_predictions)

    print("\nRandom Forest Final Metrics:")
    print("Training Set:")
    print(f"MSE: {rf_train_mse:.4f}")
    print(f"R²: {rf_train_r2:.4f}")
    print("\nTest Set:")
    print(f"MSE: {rf_test_mse:.4f}")
    print(f"R²: {rf_test_r2:.4f}")

    # Random Forest visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_test_predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.title("Random Forest: Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()
