import os
import time
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 10
DELAY_SECONDS = 1
VECTOR_DIM = 384

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

def validate_recipe(row):
    """Enhanced recipe validation"""
    required_fields = ['name', 'ingredients', 'instructions', 'calories', 'protein', 'carbs', 'fats', 'tags']
    for field in required_fields:
        if pd.isna(row.get(field)):
            raise ValueError(f"Missing required field: {field}")
    
    name_lower = str(row['name']).lower()
    ingredients_lower = str(row['ingredients']).lower()
    tags_lower = str(row['tags']).lower()
    
    # Validate name-ingredient consistency
    if 'curry' in name_lower and 'curry' not in ingredients_lower:
        raise ValueError(f"Recipe {row['name']} claims to be curry but missing curry ingredients")
    
    if 'chocolate' in name_lower and ('cocoa' not in ingredients_lower and 'chocolate' not in ingredients_lower):
        raise ValueError(f"Recipe {row['name']} claims to be chocolate but missing cocoa/chocolate")
    
    # Validate dietary claims
    if 'vegan' in tags_lower:
        non_vegan = ['milk', 'cheese', 'butter', 'egg', 'honey', 'yogurt', 'cream']
        if any(ing in ingredients_lower for ing in non_vegan):
            raise ValueError(f"Recipe {row['name']} claims to be vegan but contains animal products")
    
    if 'gluten-free' in tags_lower:
        gluten_sources = ['wheat', 'barley', 'rye', 'bread', 'pasta', 'flour']
        if any(source in ingredients_lower for source in gluten_sources):
            raise ValueError(f"Recipe {row['name']} claims to be gluten-free but contains gluten")

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME in pc.list_indexes().names():
        index_info = pc.describe_index(INDEX_NAME)
        if index_info.dimension != VECTOR_DIM:
            print(f"⚠️ Recreating index with correct dimensions...")
            pc.delete_index(INDEX_NAME)
            pc.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    else:
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(INDEX_NAME)

def create_recipe_text(row):
    """Create enhanced text for embedding"""
    validate_recipe(row)
    
    nutrition_info = f"{row['calories']} calories, {row['protein']}g protein, {row['carbs']}g carbs, {row['fats']}g fats"
    return f"""
    Recipe: {row['name']}
    Cuisine: {row.get('cuisine', 'unknown')}
    Ingredients: {row['ingredients']}
    Instructions: {row['instructions']}
    Nutrition: {nutrition_info}
    Dietary Tags: {row['tags']}
    """

def main():
    embeddings = SentenceTransformer(EMBEDDING_MODEL)
    index = initialize_pinecone()
    
    df = pd.read_csv("recipes.csv")
    print(f"Loaded {len(df)} recipes")
    
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        vectors = []
        
        for idx, row in batch.iterrows():
            try:
                text = create_recipe_text(row)
                vector = embeddings.encode(text).tolist()
                
                metadata = {
                    "name": row["name"],
                    "ingredients": row["ingredients"],
                    "instructions": row["instructions"],
                    "cuisine": row.get("cuisine", "").lower(),
                    "calories": float(row["calories"]),
                    "protein": float(row["protein"]),
                    "carbs": float(row["carbs"]),
                    "fats": float(row["fats"]),
                    "tags": row["tags"].lower()
                }
                
                vectors.append((str(idx), vector, metadata))
                
            except Exception as e:
                print(f"⛔ Skipping recipe {idx}: {str(e)}")
                continue
        
        if vectors:
            index.upsert(vectors=vectors)
            print(f"✅ Processed batch {i//BATCH_SIZE + 1}/{(len(df)//BATCH_SIZE)+1}")
            time.sleep(DELAY_SECONDS)
    
    print("🎉 Data ingestion complete!")

if __name__ == "__main__":
    main()