import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_json("data/full_format_recipes.json")
#df = pd.DataFrame(df)
nlp = spacy.load('en_core_web_md')

def process_category_text(category):
    """Process a single category text"""
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase
    category = category.lower()
    
    # Remove special characters
    category = re.sub(r'[^a-zA-Z\s]', ' ', category)

    category = [w for w in category if w not in set(stopwords.words('english'))]
    
    # Remove extra whitespace
    category = ' '.join(category.split())
    
    # Lemmatize
    category = lemmatizer.lemmatize(category)
    
    return category

def process_category_list(text):
    """Process a list of categories with NaN handling"""
    # Handle NaN/None cases
    if isinstance(text, (float, type(None))):
        return []
    
    try:
        # Process each category in the list
        return [process_category_text(cat) for cat in text]
    except (TypeError, AttributeError):
        return []
    
df['processed_categories'] = df['categories'].apply(process_category_list)
df['processed_directions'] = df["directions"].apply(process_category_list)

print(df[['categories', 'processed_categories']])