import os
from recipe_processor import RecipeProcessor
from model_trainer import train_models
import torch
import matplotlib.pyplot as plt

def main():
    try:
        # Get the directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the path to the JSON file
        json_path = os.path.join(script_dir, "full_format_recipes.json")
        
        # Print the path being used
        print(f"Looking for JSON file at: {json_path}")
        
        if not os.path.exists(json_path):
            print(f"Error: Could not find file at {json_path}")
            print("\nCurrent directory contains:")
            for file in os.listdir(script_dir):
                print(f"- {file}")
            return
        
        # Step 1: Initialize and process data
        print("\n1. Initializing RecipeProcessor...")
        processor = RecipeProcessor(json_path)
        
        # Step 2: Prepare and analyze data
        print("\n2. Preparing and analyzing data...")
        df = processor.prepare_data()
        
        # Step 3: Analyze categories
        print("\n3. Analyzing categories...")
        category_analysis = processor.analyze_categories()
        
        # Step 4: Train models with different vector representations
        print("\n4. Training models...")
        
        print("\nTraining with TF-IDF vectors...")
        tfidf_trainer = train_models(processor, 'tfidf')
        
        print("\nTraining with Word2Vec vectors...")
        w2v_trainer = train_models(processor, 'word2vec')
        
        print("\nTraining with BERT vectors...")
        bert_trainer = train_models(processor, 'bert')
        
        print("\nAll processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred in main execution: {str(e)}")

if __name__ == "__main__":
    main()