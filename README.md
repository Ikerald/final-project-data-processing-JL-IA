# Data Processing Final proyect GitHub Jon Lejardi Iker Aldasoro - Universidad Carlos III [![python version](https://img.shields.io/badge/python-3.12.6+-blue.svg)](https://www.python.org/downloads/)

This repository holds the files created by Iker Aldasoro Marculeta and Jon Lejardi Jericó for the final proyect of the subject Data Processing.

Here we will explain the steps followed for the completion of the final project and the results obtained. 
Basic Project
For the basic project we had to solve a regression task, comparing the performance of different vectorizations and machine learning strategies.

**1.	Input Variable Analysis**

Firstly, we are asked to visualize the relationship between the output variable, “rating”, and the input variable “categories”. Before starting with it, we imported all the libraries and packets needed for the project and did some preprocessing by deleting the null values from both variables.

After separating and counting all the existing categories, they were too many to do a proper analysis, so we had to limit the range. That is why we picked the 15 most common categories and do box plot to see the mean rating and standard deviation of those categories.

![image](https://github.com/user-attachments/assets/91c347e0-7b68-4a64-b49d-f7f7f9e64b19)

At a first glance, we can tell that all the categories studied have a very high mean rating. Otherwise, there are no conclusions to be drawn from this graph. Another, more exhaustive analysis would be necessary.
We also performed a correlation matrix of the integer input variables as “sodium”, “fat”, “calories”, “protein” and the output variable. But we can see there is no correlation at all between them. Bellow they are represented in a plot separately, but we reach to the same conclusion.

![image](https://github.com/user-attachments/assets/aec09f72-3d57-475b-b3a5-d86943011000)

![image](https://github.com/user-attachments/assets/609f1b2e-1c92-4cfa-a45b-a9f68ab35670)

To complete the comparation of input variables with the output one, on the one hand we tried to see if there is any correlation between the text length and the recipe's rating, but it doesn't seem significant.

![image](https://github.com/user-attachments/assets/e1124ea3-e710-4507-b96f-16ed96d83d0d)

For the missing input variable "years", it is interesting to see the difference of rating variation. The recipes between 2004 and 2011 keep a high mean rating with a reasonable deviateino, while in 2003 and from 2012 in advance, the ratings are much more distributed.

![image](https://github.com/user-attachments/assets/5557bbd5-1934-49f0-aed7-cc88232ba7a9)

To complete this first analysis we counted the ratings and represented their distribution and outliers.x 

![image](https://github.com/user-attachments/assets/26c26047-9f4f-4029-a27f-21e6656af874)

 ![image](https://github.com/user-attachments/assets/6697e53b-0e35-4a63-9c84-9fd60686386b)

Here it is seen that the majority of the ratings are between 3 and 5, having the highes concentration in the range of 3.5 - 4.5. On the other side, recipes with bad ratings (under 2.5) are rare, but among the failed ones a rating of 0. is more common.

**2. Text Preprocessing**

In order to prepare the text data so it can be processed by the regression models it has to go through a multi stage processing composed by tokenization, homogeneization, cleaning and voectorization. In this second stage we perform the first three stages of the processing to keep the text variables ready to be transformed to numerical. 
At first the phrases are divided into tokens (words). Then starts the homogeneization, where all the verbs are lemmatized (conjugated verbs are changed to the basis) and everything is in lowercase. Finally the cleaning is done, here special characters, extra white spaces, stop words and other meaningless elements are removed.

**3. Vector Representation**

After the text preprocessing, so the machine learning models can work with the information in text variables, these variables are transformed into word vectors. This means, they will have a numerical representation where their value is related to the meaning they have.
For this type of transformation we are asked to use three different methods:

- TF-IDF: Gives a high value for a given term in a given document if that terms occurs often in that particular document and very rarely anywhere else. I just focuses on word importance, doesn't take into account the context. The size of the results obtained from this method is (20130, 1000), since we stablished a maximum of 1000 features.
  
- Word2Vec: It has a shallow word relationship, but lacks full contextual understanding. The size of the results is (20130, 100), since we stablished a maximum of 100 features for the vectorization. Along this parameters we used a window of 5 to create the mean vector.
  
- BERT: We were free to choose a transform-based model, so we chose BERT. It is context-aware and provides deep contextual embedings. The result size is (20130, 768), since 768 is the maximum size of the hidden layer, which is the length of the output embeddings.

**4. Training and Evaluation of Regression Models**





















# Recipe Rating Prediction Project

## Introduction
This project implements a machine learning solution for predicting recipe ratings based on textual and numerical features. It utilizes various NLP techniques and machine learning models to analyze recipe data from epicurious.com and predict user ratings.

## Authors
- Iker Aldasoro
- Jon Lejardi

## Project Structure
```
recipe_project/
├── __init__.py
├── auxiliar.py
├── processing.py
├── vectorization.py
├── visualization.py
main.py
```

## Features
- Text preprocessing and homogenization
- Multiple vector representation techniques:
  - TF-IDF
  - Word2Vec
  - BERT (contextual embeddings)
- Regression models:
  - Neural Networks (PyTorch)
  - Linear Regression
  - Random Forest
- Cross-validation evaluation
- Comprehensive visualization tools

## Dependencies
- PyTorch
- Transformers (Hugging Face)
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gensim

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install torch transformers nltk pandas numpy scikit-learn matplotlib seaborn gensim
```
3. Download required NLTK data:
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")
```

## Module Description

### auxiliar.py
- Handles CUDA device detection and configuration
- Provides GPU information and memory statistics

### processing.py
- Implements text preprocessing pipeline
- Features:
  - String conversion and cleaning
  - NLTK-based text processing
  - Stopword removal
  - Lemmatization
  - Special character removal

### vectorization.py
- Implements three vectorization methods:
  - BERT embeddings
  - TF-IDF vectorization
  - Word2Vec embeddings
- Includes neural network implementation and training
- Implements cross-validation evaluation
- Provides comprehensive model evaluation metrics and visualizations

### visualization.py
- Generates various analytical visualizations:
  - Category-rating relationships
  - Feature correlations
  - Rating distributions
  - Text length analysis
  - Temporal analysis
  - Category distribution

## Methodology

### 1. Data Preprocessing
- Text cleaning and normalization
- Handling missing values
- Feature extraction from text data
- Data type conversion

### 2. Vectorization
Three different approaches are implemented:
1. **BERT Vectorization**
   - Uses pretrained BERT model
   - Generates contextual embeddings
   - Handles batching for memory efficiency

2. **TF-IDF Vectorization**
   - Implements term frequency-inverse document frequency
   - Uses scikit-learn's TfidfVectorizer
   - Limited to 1000 features for efficiency

3. **Word2Vec Vectorization**
   - Implements word embeddings
   - Uses Gensim's Word2Vec model
   - Creates document vectors by averaging word vectors

### 3. Model Implementation
- **Neural Network**
  - Three-layer architecture with dropout
  - ReLU activation
  - Adam optimizer with weight decay
  - Batch processing

- **Additional Models**
  - Linear Regression
  - Random Forest Regressor

### 4. Evaluation
- Cross-validation with k=5 folds
- Metrics:
  - Mean Squared Error (MSE)
  - R² Score
  - Training and validation loss curves
  - Prediction vs actual value plots
  - Distribution analysis

## Results Visualization
The project includes comprehensive visualization of:
- Training metrics over time
- Model performance comparisons
- Prediction accuracy
- Error distribution
- Feature importance
- Cross-validation results

## Usage
Run the main script to execute the complete pipeline:
```bash
python main.py
```

## Project Requirements Met
1. ✓ Text processing and homogenization
2. ✓ Vector representation using multiple techniques
3. ✓ Regression task implementation
4. ✓ Cross-validation evaluation
5. ✓ Comprehensive documentation
6. ✓ Well-structured code organization
7. ✓ Results analysis and visualization
   

# Recipe Rating Prediction Project - Technical Documentation

## Technical Overview
This project implements a machine learning pipeline to predict recipe ratings using a combination of NLP techniques and regression models. The system processes recipe data from epicurious.com, which includes textual descriptions, ingredients, and nutritional information.

## Detailed Code Structure Analysis

### 1. Data Pipeline (`main.py`)
```python
def main():
    # NLTK Setup
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

    # Data Loading
    df = pd.read_json("./data/full_format_recipes.json")
```
The pipeline begins by downloading required NLTK resources and loading the JSON dataset. The data contains 20,130 recipes with multiple features including directions, categories, descriptions, and ratings.

### 2. CUDA Configuration (`auxiliar.py`)
```python
def get_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # GPU information logging
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
```
The auxiliary module handles GPU detection and configuration, providing detailed hardware information and memory statistics for optimization.

### 3. Text Processing Pipeline (`processing.py`)

#### Data Preprocessing
```python
def preprocessing(df):
    df["directions_pre"] = df["directions"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )
    df["rating_pre"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_pre"] = df["rating_pre"].fillna(df["rating_pre"].mean())
```
Key preprocessing steps:
- Converts list-type data to strings
- Handles missing values in ratings
- Normalizes text data formats

#### NLTK Processing
```python
def NTLK_clean(text: str):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    # Processing steps
    text = text.lower()
    text = re.sub(r"[^a-zA-Zs]", "", text)
    text = lemmatizer.lemmatize(text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
```
The NLP pipeline includes:
- Lowercasing
- Special character removal
- Lemmatization
- Tokenization
- Stopword removal

### 4. Vectorization Methods (`vectorization.py`)

#### BERT Vectorization
```python
def bert_vectorization(device: torch.device, df: pandas.DataFrame):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    # Batch processing for memory efficiency
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch_texts = df["directions_pre"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512)
```
Features:
- Uses BERT base uncased model
- Implements batch processing
- Handles memory efficiently
- Maximum sequence length of 512 tokens

#### TF-IDF Vectorization
```python
def TF_vectorization(df: pandas.DataFrame):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_vectors = tfidf.fit_transform(df["directions_post"])
```
Characteristics:
- Limited to 1000 features for dimensionality control
- Sparse matrix representation
- Based on preprocessed text

#### Word2Vec Vectorization
```python
def Word2Vec_vectorization(df: pandas.DataFrame):
    sentences = [text.split() for text in df["directions_post"]]
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
```
Configuration:
- 100-dimensional vectors
- Context window of 5 words
- Minimum word count of 1
- Document vectors created by averaging word vectors

### 5. Neural Network Implementation

#### Architecture
```python
model_ML = nn.Sequential(
    nn.Linear(input_size, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
)
```
Network features:
- Three-layer architecture
- ReLU activation functions
- Dropout layers (0.3 rate) for regularization
- Output layer for regression

#### Training Configuration
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model_ML.parameters(), lr=0.001, weight_decay=0.01)
```
Training parameters:
- Mean Squared Error loss
- Adam optimizer
- Learning rate: 0.001
- Weight decay: 0.01
- Batch size: 32
- Epochs: 100

### 6. Cross-Validation and Evaluation
```python
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(features_scaled)):
    # Training and evaluation for each fold
```
Evaluation metrics:
- 5-fold cross-validation
- MSE (Mean Squared Error)
- R² Score
- Training/validation loss curves
- Prediction vs actual plots
- Distribution analysis

### 7. Visualization Component (`visualization.py`)
```python
def visualize(df: pd.DataFrame):
    # Category analysis
    category_counts = df["categories"].explode().value_counts()[:15]
    
    # Correlation analysis
    numerical_cols = ["rating", "fat", "protein", "calories", "sodium"]
    sns.heatmap(df[numerical_cols].corr(), annot=True)
```
Visualization features:
- Category-rating relationships
- Feature correlations
- Rating distributions
- Text length analysis
- Temporal trends
- Category distributions

## Performance Optimization

### Memory Management
- Batch processing for BERT vectorization
- TF-IDF feature limitation
- Float32 conversion for reduced memory footprint
```python
vectors_TF = vectors_TF.astype(np.float32)
```

### Computational Efficiency
- GPU acceleration when available
- Efficient text preprocessing pipeline
- Batch processing in neural network training
- Optimized cross-validation implementation

## Extension Possibilities
1. Summarization implementation
2. Recipe generation using transformers
3. Advanced NLP techniques integration
4. Alternative embedding approaches
5. Graph-based analysis implementation













