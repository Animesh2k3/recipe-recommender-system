import pandas as pd
from typing import List, Dict, Optional, Union
import re
import random

# Health condition mappings
HEALTH_CONDITION_RECOMMENDATIONS = {
    "diabetes": {
        "nutrient_limits": {"carbs": 30, "sugar": 10},
        "preferred_tags": ["low-carb", "diabetic-friendly"],
        "avoid_tags": ["high-sugar"]
    },
    "heart health": {
        "nutrient_limits": {"fats": 15, "sodium": 500},
        "preferred_tags": ["low-fat", "heart-healthy"],
        "avoid_tags": ["high-fat"]
    },
    "weight loss": {
        "nutrient_limits": {"calories": 400},
        "preferred_tags": ["low-calorie"],
        "avoid_tags": ["high-calorie"]
    }
}

# Cultural cuisine mappings
CUISINE_ADAPTATIONS = {
    "italian": ["tomato", "basil", "olive oil", "garlic"],
    "mexican": ["beans", "corn", "avocado", "chili"],
    "indian": ["curry", "spices", "lentils", "yogurt"],
    "asian": ["soy sauce", "ginger", "sesame", "tofu"]
}

# Enhanced substitutions
SUBSTITUTIONS = {
    "milk": ["almond milk", "soy milk", "oat milk", "coconut milk"],
    "butter": ["coconut oil", "olive oil", "avocado"],
    "cheese": ["nutritional yeast", "cashew cheese", "tofu"],
    "egg": ["flaxseed meal + water", "chia seeds + water", "applesauce"],
    "meat": ["tofu", "tempeh", "jackfruit", "mushrooms"]
}

def filter_by_diet(df: pd.DataFrame, restrictions: List[str], allergies: List[str], health_condition: Optional[str] = None) -> pd.DataFrame:
    """Strict filtering with comprehensive checks"""
    filtered = df.copy()
    restrictions = [r.lower().strip() for r in restrictions if r.strip()]
    allergies = [a.lower().strip() for a in allergies if a.strip()]
    
    # Vegan/Vegetarian filter
    if 'vegan' in restrictions or 'vegetarian' in restrictions:
        non_veg = ['beef', 'chicken', 'pork', 'fish', 'shrimp', 'meat', 'turkey', 'bacon']
        pattern = '|'.join([f'(^|\\W){term}(\\W|$)' for term in non_veg])
        filtered = filtered[~filtered['ingredients'].str.lower().str.contains(pattern)]
    
    # Gluten-free filter
    if 'gluten-free' in restrictions:
        gluten_sources = ['wheat', 'barley', 'rye', 'bread', 'pasta', 'flour']
        pattern = '|'.join([f'(^|\\W){term}(\\W|$)' for term in gluten_sources])
        filtered = filtered[~filtered['ingredients'].str.lower().str.contains(pattern)]
    
    # Strict allergy filtering
    for allergen in allergies:
        if allergen:
            allergen_terms = allergen.split()
            pattern = '|'.join([f'(^|\\W){term}(\\W|$)' for term in allergen_terms])
            filtered = filtered[~filtered['ingredients'].str.lower().str.contains(pattern)]
    
    # Health condition filters
    if health_condition and health_condition.lower() in HEALTH_CONDITION_RECOMMENDATIONS:
        condition = HEALTH_CONDITION_RECOMMENDATIONS[health_condition.lower()]
        
        for nutrient, limit in condition["nutrient_limits"].items():
            if nutrient in ['calories', 'protein', 'carbs', 'fats']:
                filtered = filtered[filtered['nutrition'].apply(lambda x: x[nutrient] <= limit)]
        
        for tag in condition.get("avoid_tags", []):
            filtered = filtered[~filtered['tags'].str.contains(tag)]
    
    return filtered

def suggest_substitutions(ingredient: str, allergies: List[str] = None, cuisine: Optional[str] = None) -> List[str]:
    """Smart substitutions with allergy awareness"""
    ingredient = ingredient.lower().strip()
    subs = []
    allergies = [a.lower().strip() for a in allergies or [] if a.strip()]
    
    # Find direct substitutions
    for key, options in SUBSTITUTIONS.items():
        if key in ingredient:
            subs.extend([s for s in options if not any(a in s.lower() for a in allergies)])
    
    # Add cultural adaptations
    if cuisine and cuisine.lower() in CUISINE_ADAPTATIONS:
        cultural_ingredients = [ing for ing in CUISINE_ADAPTATIONS[cuisine.lower()] 
                              if not any(a in ing.lower() for a in allergies)]
        subs.extend(cultural_ingredients)
    
    return list(set(subs))[:3]  # Return max 3 unique options

def calculate_nutrition_score(recipe: Dict, goals: Dict) -> float:
    """Calculate nutrition score based on health goals"""
    if not goals:
        return 1.0
    
    score = 0.0
    nutrition = recipe['nutrition']
    total_weight = 0
    
    for nutrient, target in goals.items():
        if nutrient in nutrition:
            actual = nutrition[nutrient]
            if nutrient == 'calories':
                # For calories, prefer being under target
                score += max(0, 1 - max(0, actual - target) / (target + 0.1))
            else:
                # For macros, aim close to target
                score += max(0, 1 - abs(actual - target) / (target + 0.1))
            total_weight += 1
    
    return score / total_weight if total_weight > 0 else 1.0

def add_diversity_to_results(df: pd.DataFrame, diversity: int) -> pd.DataFrame:
    """Add controlled randomness to results based on diversity setting"""
    if diversity == 1 or len(df) <= 1:
        return df
    
    # Keep top 3 stable, shuffle the rest based on diversity level
    stable = df.iloc[:3]
    to_shuffle = df.iloc[3:]
    
    shuffle_factor = min(0.9, (diversity-1)/4)  # 0-0.9 range
    to_shuffle = to_shuffle.sample(frac=1-shuffle_factor, random_state=diversity)
    
    return pd.concat([stable, to_shuffle]).reset_index(drop=True)