# -*- coding: utf-8 -*-
"""
Database Seeding Script
Populates PostgreSQL tables from processed CSV datasets:
- 12,481 Authentic Bengaluru Restaurant Outlets
- 600 Synthetic Benchmark Users (is_synthetic_benchmark=True)
- 11,920 Synthetic Benchmark Ratings (is_synthetic_benchmark=True)
"""

import os
import sys
import asyncio
import pandas as pd
from sqlalchemy import insert, select, func

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import settings
from app.database.session import AsyncSessionLocal, engine
from app.models.restaurant import Restaurant
from app.models.user import User, UserPreferences
from app.models.rating import Rating


async def seed_database(batch_size: int = 1000):
    print("=" * 80)
    print("DATABASE SEEDING — POPULATING POSTGRESQL CATALOG & BENCHMARK DATA")
    print("=" * 80)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rest_csv = os.path.join(base_dir, 'data', 'processed', 'restaurants_clean.csv')
    users_csv = os.path.join(base_dir, 'data', 'processed', 'synthetic_users.csv')
    ratings_csv = os.path.join(base_dir, 'data', 'processed', 'synthetic_ratings.csv')
    
    if not os.path.exists(rest_csv):
        print(f"[ERROR] Processed restaurants CSV not found at: {rest_csv}")
        return
        
    try:
        async with AsyncSessionLocal() as session:
            # Check if already seeded
            count_res = await session.execute(select(func.count()).select_from(Restaurant))
            existing_count = count_res.scalar() or 0
            if existing_count > 0:
                print(f"\n[INFO] Database already contains {existing_count:,} restaurants. Seeding is already complete (Idempotent skip).")
                print("=" * 80)
                return

            # 1. Seed Authentic Restaurants
            print("\n[*] Step 1: Seeding Authentic Bengaluru Restaurants...")
            df_rest = pd.read_csv(rest_csv)
            # Clean NaNs in optional fields for SQL insert
            rest_records = df_rest.where(pd.notna(df_rest), None).to_dict('records')
            
            # Insert in batches
            total_inserted_rest = 0
            for i in range(0, len(rest_records), batch_size):
                batch = rest_records[i:i + batch_size]
                stmt = insert(Restaurant).values(batch)
                await session.execute(stmt)
                total_inserted_rest += len(batch)
                
            await session.commit()
            print(f"  [SUCCESS] Inserted {total_inserted_rest:,} authentic restaurant outlets.")
            
            # 2. Seed Synthetic Benchmark Users
            print("\n[*] Step 2: Seeding Synthetic Benchmark Users...")
            if os.path.exists(users_csv):
                df_users = pd.read_csv(users_csv)
                user_records = []
                pref_records = []
                for u in df_users.to_dict('records'):
                    user_records.append({
                        "id": u["user_id"],
                        "name": u["name"],
                        "email": u["email"],
                        "city": "Bengaluru",
                        "is_synthetic_benchmark": True
                    })
                    pref_records.append({
                        "user_id": u["user_id"],
                        "preferred_cuisines": u["preferred_cuisines"],
                        "preferred_price_tier": u["preferred_budget_tier"],
                        "max_cost_for_two": u["max_budget_inr"],
                        "maximum_distance_km": 10.0,
                        "online_order_only": False,
                        "book_table_only": False
                    })
                    
                await session.execute(insert(User).values(user_records))
                await session.execute(insert(UserPreferences).values(pref_records))
                await session.commit()
                print(f"  [SUCCESS] Inserted {len(user_records):,} synthetic benchmark users & preferences.")
                
            # 3. Seed Synthetic Benchmark Ratings
            print("\n[*] Step 3: Seeding Synthetic Benchmark Ratings...")
            if os.path.exists(ratings_csv):
                df_ratings = pd.read_csv(ratings_csv)
                rating_records = []
                for r in df_ratings.to_dict('records'):
                    rating_records.append({
                        "user_id": r["user_id"],
                        "restaurant_id": r["restaurant_id"],
                        "rating": float(r["rating"]),
                        "review_text": r["review_text"],
                        "is_synthetic_benchmark": True
                    })
                    
                total_inserted_ratings = 0
                for i in range(0, len(rating_records), batch_size):
                    batch = rating_records[i:i + batch_size]
                    await session.execute(insert(Rating).values(batch))
                    total_inserted_ratings += len(batch)
                    
                await session.commit()
                print(f"  [SUCCESS] Inserted {total_inserted_ratings:,} synthetic benchmark rating records (is_synthetic_benchmark=True).")
                
        print("\n" + "=" * 80)
        print("DATABASE SEEDING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    except Exception as e:
        print(f"\n[NOTE] Database connection status: {e}")
        print("  Ensure your PostgreSQL database service is running on port 5432 and initialized via scripts/init_db.py.")


if __name__ == "__main__":
    asyncio.run(seed_database())
