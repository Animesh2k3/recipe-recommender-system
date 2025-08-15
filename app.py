import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from utils import (
    filter_by_diet,
    suggest_substitutions,
    HEALTH_CONDITION_RECOMMENDATIONS,
    CUISINE_ADAPTATIONS,
    calculate_nutrition_score,
    add_diversity_to_results
)

load_dotenv()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RETRIES = 3
INITIAL_RESULTS = 30
DISPLAY_COUNT = 5

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_services():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embeddings = SentenceTransformer(EMBEDDING_MODEL)
    return pc.Index(os.getenv("INDEX_NAME")), embeddings

try:
    index, embeddings = init_services()
except Exception as e:
    st.error(f"🔴 Failed to initialize services: {str(e)}")
    st.stop()

# Streamlit UI
st.title("🍲 Advanced Recipe & Nutrition Recommender")

# User inputs
col1, col2 = st.columns(2)
with col1:
    query = st.text_input("What kind of recipe are you looking for?", placeholder="e.g. vegetarian pasta")
    restrictions = st.text_input("Dietary restrictions (comma separated)", placeholder="vegetarian, gluten-free")
    allergies = st.text_input("Allergies (comma separated)", placeholder="nuts, dairy")
    
with col2:
    health_condition = st.selectbox("Health condition (optional)", 
                                  ["None"] + list(HEALTH_CONDITION_RECOMMENDATIONS.keys()))
    cuisine = st.selectbox("Preferred cuisine style", 
                         ["Any"] + list(CUISINE_ADAPTATIONS.keys()))
    diversity = st.slider("Result diversity", 1, 5, 3, 
                         help="1 = Most similar, 5 = Most diverse")

if st.button("Find Recipes"):
    with st.spinner("🔍 Finding the best recipes..."):
        try:
            # Enhanced query processing
            search_query = f"{query} {cuisine if cuisine != 'Any' else ''}"
            vector = embeddings.encode(search_query).tolist()
            
            # Query Pinecone
            results = index.query(
                vector=vector,
                top_k=INITIAL_RESULTS,
                include_metadata=True
            )

            # Process results
            recipes = []
            for match in results["matches"]:
                try:
                    metadata = match["metadata"]
                    tags = metadata.get("tags", "")
                    
                    # Skip if doesn't match cuisine filter
                    if cuisine != "Any" and cuisine.lower() not in metadata.get("cuisine", ""):
                        continue
                        
                    recipes.append({
                        "id": match["id"],
                        "name": metadata["name"],
                        "ingredients": metadata["ingredients"],
                        "instructions": metadata["instructions"],
                        "tags": tags,
                        "nutrition": {
                            "calories": metadata["calories"],
                            "protein": metadata["protein"],
                            "carbs": metadata["carbs"],
                            "fats": metadata["fats"]
                        },
                        "score": match["score"]
                    })
                except KeyError as e:
                    continue
            
            if not recipes:
                st.warning("No recipes found. Try different filters.")
                st.stop()
                
            df = pd.DataFrame(recipes)
            
            # Apply filters
            df_filtered = filter_by_diet(
                df,
                [r.strip() for r in restrictions.split(",") if r.strip()],
                [a.strip() for a in allergies.split(",") if a.strip()],
                health_condition if health_condition != "None" else None
            )
            
            # Enhanced ranking and diversity
            if not df_filtered.empty:
                # Calculate nutrition scores
                if health_condition != "None":
                    condition = HEALTH_CONDITION_RECOMMENDATIONS[health_condition.lower()]
                    df_filtered['nutrition_score'] = df_filtered.apply(
                        lambda x: calculate_nutrition_score(x, condition.get("nutrient_limits", {})), 
                        axis=1
                    )
                else:
                    df_filtered['nutrition_score'] = 1.0
                
                # Combine scores
                df_filtered['combined_score'] = (
                    0.6 * df_filtered['score'] + 
                    0.3 * df_filtered['nutrition_score'] + 
                    0.1 * diversity
                )
                
                # Apply diversity
                df_filtered = add_diversity_to_results(df_filtered.sort_values('combined_score', ascending=False), diversity)
                
                # Display results
                st.subheader(f"🍽️ Recommended Recipes ({len(df_filtered)} found)")
                for _, row in df_filtered.head(DISPLAY_COUNT).iterrows():
                    with st.expander(f"🍴 {row['name']} (Score: {row['combined_score']:.2f})"):
                        st.markdown(f"**Ingredients:**\n{row['ingredients']}")
                        st.markdown(f"**Instructions:**\n{row['instructions']}")
                        st.markdown(f"**Nutrition:**\nCalories: {row['nutrition']['calories']} | Protein: {row['nutrition']['protein']}g | Carbs: {row['nutrition']['carbs']}g | Fats: {row['nutrition']['fats']}g")
                        st.markdown(f"**Tags:** {row['tags']}")
                        
                        # Show substitutions if needed
                        if allergies or health_condition != "None":
                            subs = []
                            for ing in row['ingredients'].split(", "):
                                ing_subs = suggest_substitutions(
                                    ing, 
                                    allergies.split(",") if allergies else [],
                                    cuisine if cuisine != "Any" else None
                                )
                                if ing_subs:
                                    subs.append(f"{ing} → {', '.join(ing_subs)}")
                            
                            if subs:
                                st.markdown("**Suggested Substitutions:**")
                                st.write("\n".join(f"🔀 {s}" for s in subs))
            else:
                st.warning("No recipes match all your criteria. Try relaxing some filters.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")