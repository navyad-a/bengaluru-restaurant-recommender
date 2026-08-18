# -*- coding: utf-8 -*-
"""
Synthetic Collaborative Filtering Benchmark Generator
=====================================================
IMPORTANT NOTICE:
The authentic Zomato Bengaluru dataset is restaurant-centric and does not contain 
individual user_id -> restaurant_id -> rating interactions.
This module creates a reproducible, statistically grounded SYNTHETIC INTERACTION BENCHMARK
strictly for evaluating Collaborative Filtering (Surprise SVD) and Hybrid recommendation algorithms.

The benchmark does NOT alter or fabricate authentic restaurant metadata.
All generated files are explicitly labeled with "synthetic_" prefixes.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Define 7 distinct Indian dining personas
INDIAN_DINING_PERSONAS = [
    {
        "persona_id": "south_indian_traditionalist",
        "name_prefix": "South Indian Tiffin Lover",
        "preferred_cuisines": ["South Indian", "Karnataka", "Udupi", "Andhra", "Tamil", "Kerala", "Chettinad"],
        "preferred_budget": "Budget",
        "max_budget_inr": 500,
        "preferred_localities": ["Jayanagar", "Basavanagudi", "Malleshwaram", "Banashankari", "Rajajinagar", "BTM"],
        "base_bias": 0.2
    },
    {
        "persona_id": "biryani_mughlai_enthusiast",
        "name_prefix": "Biryani & Kebabs Lover",
        "preferred_cuisines": ["Biryani", "Mughlai", "North Indian", "Hyderabadi", "Kebab"],
        "preferred_budget": "Moderate",
        "max_budget_inr": 1000,
        "preferred_localities": ["Frazer Town", "Shivajinagar", "Koramangala 5th Block", "BTM", "HSR", "Marathahalli"],
        "base_bias": 0.1
    },
    {
        "persona_id": "budget_tech_student",
        "name_prefix": "Budget Techie / Student",
        "preferred_cuisines": ["Fast Food", "Chinese", "Street Food", "Rolls", "Quick Bites", "North Indian"],
        "preferred_budget": "Budget",
        "max_budget_inr": 350,
        "preferred_localities": ["BTM", "HSR", "Electronic City", "Koramangala 1st Block", "Marathahalli"],
        "base_bias": -0.1
    },
    {
        "persona_id": "koramangala_cafe_pub_goer",
        "name_prefix": "Indiranagar & Koramangala Cafe Hopper",
        "preferred_cuisines": ["Cafe", "Continental", "Italian", "Desserts", "Beverages", "Pizza", "Burger"],
        "preferred_budget": "Premium",
        "max_budget_inr": 1800,
        "preferred_localities": ["Koramangala 5th Block", "Indiranagar", "Church Street", "Lavelle Road", "MG Road"],
        "base_bias": 0.15
    },
    {
        "persona_id": "north_indian_family_diner",
        "name_prefix": "North Indian Family Diner",
        "preferred_cuisines": ["North Indian", "Punjabi", "Mughlai", "Desserts", "Chinese"],
        "preferred_budget": "Moderate",
        "max_budget_inr": 900,
        "preferred_localities": ["JP Nagar", "Whitefield", "Bannerghatta Road", "Bellandur", "HSR"],
        "base_bias": 0.05
    },
    {
        "persona_id": "indo_chinese_asian_seeker",
        "name_prefix": "Pan-Asian & Indo-Chinese Fan",
        "preferred_cuisines": ["Chinese", "Momos", "Asian", "Thai", "Fast Food"],
        "preferred_budget": "Budget",
        "max_budget_inr": 600,
        "preferred_localities": ["Koramangala 7th Block", "Indiranagar", "BTM", "Kammanahalli", "Kalyan Nagar"],
        "base_bias": 0.0
    },
    {
        "persona_id": "luxury_fine_dine_gourmet",
        "name_prefix": "Fine Dining Connoisseur",
        "preferred_cuisines": ["Continental", "Italian", "North Indian", "European", "Seafood"],
        "preferred_budget": "Luxury",
        "max_budget_inr": 4000,
        "preferred_localities": ["Lavelle Road", "Residency Road", "MG Road", "Indiranagar", "Sadashiv Nagar"],
        "base_bias": 0.25
    }
]

INDIAN_FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Aniruddh", "Arjun", "Deepak", "Divya",
    "Gautam", "Harish", "Ishaan", "Kavya", "Manish", "Meera", "Neha", "Naveen",
    "Pooja", "Pranav", "Priya", "Rahul", "Rakesh", "Riya", "Rohan", "Sanjay",
    "Shreya", "Sneha", "Suresh", "Tanvi", "Varun", "Vikas", "Vikram", "Yash"
]

INDIAN_LAST_NAMES = [
    "Sharma", "Rao", "Patel", "Reddy", "Iyer", "Nair", "Kulkarni", "Hegde",
    "Gowda", "Shetty", "Bhat", "Deshmukh", "Verma", "Gupta", "Menon", "Singh",
    "Murthy", "Kumar", "Chopra", "Joshi", "Prasad", "Das", "Banerjee", "Bose"
]


def generate_synthetic_benchmark(
    clean_restaurants_csv_path: str,
    output_dir: str,
    num_users: int = 600,
    ratings_per_user_mean: int = 20,
    test_split_ratio: float = 0.20
) -> Dict[str, Any]:
    """
    Generates synthetic user profiles and ratings interactions with deterministic seed.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    
    df_rest = pd.read_csv(clean_restaurants_csv_path)
    restaurants_pool = df_rest.to_dict('records')
    
    users = []
    ratings = []
    
    # 1. Generate Synthetic Users
    for u_id in range(1, num_users + 1):
        persona = random.choice(INDIAN_DINING_PERSONAS)
        f_name = random.choice(INDIAN_FIRST_NAMES)
        l_name = random.choice(INDIAN_LAST_NAMES)
        home_loc = random.choice(persona["preferred_localities"])
        
        user_record = {
            "user_id": u_id,
            "name": f"{f_name} {l_name}",
            "email": f"{f_name.lower()}.{l_name.lower()}{u_id}@example.in",
            "persona_id": persona["persona_id"],
            "persona_description": persona["name_prefix"],
            "home_locality": home_loc,
            "preferred_budget_tier": persona["preferred_budget"],
            "max_budget_inr": persona["max_budget_inr"],
            "preferred_cuisines": ", ".join(persona["preferred_cuisines"][:3]),
            "base_bias": persona["base_bias"]
        }
        users.append(user_record)
        
        # 2. Sample interaction restaurants for this user
        # Number of ratings for this user follows Poisson distribution (min 5, max 60)
        num_ratings = max(5, int(np.random.poisson(ratings_per_user_mean)))
        num_ratings = min(num_ratings, 60)
        
        # Compute preference weights for all candidate restaurants
        candidate_weights = []
        for r in restaurants_pool:
            w = 1.0
            r_cuisines = str(r.get('cuisines', '')).lower()
            
            # Cuisine preference alignment
            matches_cuisine = any(pc.lower() in r_cuisines for pc in persona["preferred_cuisines"])
            if matches_cuisine:
                w += 8.0
                
            # Budget alignment
            cost = r.get('cost_for_two_inr', 500)
            if cost <= persona["max_budget_inr"]:
                w += 4.0
            elif cost > persona["max_budget_inr"] * 1.5:
                w *= 0.2
                
            # Locality affinity
            if r.get('area') == home_loc or r.get('area') in persona["preferred_localities"]:
                w += 5.0
                
            # Restaurant popularity / rating prior
            r_rate = r.get('rating')
            if pd.notna(r_rate) and float(r_rate) >= 4.0:
                w += 3.0
                
            candidate_weights.append(w)
            
        prob_dist = np.array(candidate_weights) / sum(candidate_weights)
        
        # Sample restaurants without replacement for this user
        sampled_indices = np.random.choice(
            len(restaurants_pool),
            size=min(num_ratings, len(restaurants_pool)),
            replace=False,
            p=prob_dist
        )
        
        for idx in sampled_indices:
            r = restaurants_pool[idx]
            r_id = r['restaurant_id']
            r_rate = r.get('rating')
            r_base = float(r_rate) if pd.notna(r_rate) else 3.7
            
            # Calculate rating with persona taste match + noise
            r_cuisines = str(r.get('cuisines', '')).lower()
            cuisine_boost = 0.5 if any(pc.lower() in r_cuisines for pc in persona["preferred_cuisines"]) else -0.3
            budget_boost = 0.3 if r.get('cost_for_two_inr', 500) <= persona["max_budget_inr"] else -0.4
            noise = np.random.normal(0.0, 0.4)
            
            raw_rating = r_base + persona["base_bias"] + cuisine_boost + budget_boost + noise
            clean_rating = round(min(5.0, max(1.0, raw_rating)) * 2) / 2 # Round to nearest 0.5 or 1.0
            clean_rating = min(5.0, max(1.0, clean_rating))
            
            ratings.append({
                "user_id": u_id,
                "restaurant_id": r_id,
                "rating": float(clean_rating),
                "review_text": f"Simulated review for {r['name']} based on {persona['persona_id']} persona."
            })
            
    df_users = pd.DataFrame(users)
    df_ratings = pd.DataFrame(ratings)
    
    # 3. Perform Leakage-Free Per-User Stratified Holdout Split
    train_ratings = []
    test_ratings = []
    
    for u_id, group in df_ratings.groupby('user_id'):
        n_items = len(group)
        n_test = max(1, int(n_items * test_split_ratio))
        # Shuffle group deterministically
        shuffled = group.sample(frac=1.0, random_state=SEED + u_id)
        test_part = shuffled.iloc[:n_test]
        train_part = shuffled.iloc[n_test:]
        test_ratings.append(test_part)
        train_ratings.append(train_part)
        
    df_train = pd.concat(train_ratings, ignore_index=True)
    df_test = pd.concat(test_ratings, ignore_index=True)
    
    # 4. Save processed benchmark files
    os.makedirs(output_dir, exist_ok=True)
    df_users.to_csv(os.path.join(output_dir, 'synthetic_users.csv'), index=False, encoding='utf-8')
    df_ratings.to_csv(os.path.join(output_dir, 'synthetic_ratings.csv'), index=False, encoding='utf-8')
    df_train.to_csv(os.path.join(output_dir, 'synthetic_train_ratings.csv'), index=False, encoding='utf-8')
    df_test.to_csv(os.path.join(output_dir, 'synthetic_test_ratings.csv'), index=False, encoding='utf-8')
    
    metadata = {
        "benchmark_name": "Synthetic Collaborative Filtering Benchmark",
        "description": "Statistically simulated user ratings matrix generated deterministically for testing Surprise SVD on authentic Bengaluru restaurants.",
        "random_seed": SEED,
        "total_simulated_users": len(df_users),
        "total_simulated_ratings": len(df_ratings),
        "train_ratings_count": len(df_train),
        "test_ratings_count": len(df_test),
        "ratings_scale": "1.0 to 5.0",
        "avg_ratings_per_user": round(len(df_ratings) / len(df_users), 2),
        "avg_ratings_per_restaurant": round(len(df_ratings) / df_rest['restaurant_id'].nunique(), 2),
        "sparsity_percentage": f"{round((1 - len(df_ratings) / (len(df_users) * len(df_rest))) * 100, 4)}%",
        "warning": "This interaction dataset is explicitly SYNTHETIC. Model evaluation metrics on this benchmark validate recommendation architecture mechanics and must not be cited as real customer behavior."
    }
    
    with open(os.path.join(output_dir, 'benchmark_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    return metadata
