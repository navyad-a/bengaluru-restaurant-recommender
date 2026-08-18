# -*- coding: utf-8 -*-
"""
Database & ORM Model Test Suite
Tests table definitions, constraints, indexes, relationships, and PostgreSQL DDL compilation.
"""

import pytest
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from app.database.base import Base
from app.models.user import User, UserPreferences
from app.models.restaurant import Restaurant
from app.models.rating import Rating


def test_metadata_tables_registered():
    """Verify all 4 core tables are registered in SQLAlchemy Base.metadata."""
    table_names = set(Base.metadata.tables.keys())
    expected = {"users", "user_preferences", "restaurants", "ratings"}
    assert expected.issubset(table_names), f"Missing tables in metadata: {expected - table_names}"


def test_postgresql_ddl_compilation():
    """Verify all models compile cleanly to valid PostgreSQL DDL syntax."""
    dialect = postgresql.dialect()
    for table_name, table in Base.metadata.tables.items():
        ddl = str(CreateTable(table).compile(dialect=dialect))
        assert "CREATE TABLE" in ddl
        assert table_name in ddl


def test_restaurant_model_constraints():
    """Verify CheckConstraints for ratings, costs, reviews, and coordinates."""
    rest_table = Base.metadata.tables["restaurants"]
    constraint_names = {c.name for c in rest_table.constraints if hasattr(c, "name")}
    
    expected_constraints = {
        "chk_rest_rating",
        "chk_rest_cost",
        "chk_rest_reviews",
        "chk_rest_lat",
        "chk_rest_lon"
    }
    assert expected_constraints.issubset(constraint_names)


def test_restaurant_indexes():
    """Verify all single and composite indexes exist on the restaurants table."""
    rest_table = Base.metadata.tables["restaurants"]
    index_names = {idx.name for idx in rest_table.indexes}
    
    expected_indexes = {
        "ix_restaurants_name",
        "ix_restaurants_city",
        "ix_restaurants_area",
        "ix_restaurants_cuisines",
        "ix_restaurants_restaurant_type",
        "ix_restaurants_price_tier",
        "ix_restaurants_rating",
        "ix_restaurants_review_count",
        "idx_rest_area_cuisine",
        "idx_rest_city_rating",
        "idx_rest_price_cost",
        "idx_rest_coords"
    }
    assert expected_indexes.issubset(index_names)


def test_rating_model_uniqueness_and_synthetic_flag():
    """Verify user-restaurant uniqueness and synthetic benchmark isolation on ratings."""
    rating_table = Base.metadata.tables["ratings"]
    
    # Check unique constraint on (user_id, restaurant_id)
    uq_names = {c.name for c in rating_table.constraints if hasattr(c, "name")}
    assert "uq_user_restaurant_rating" in uq_names
    
    # Check columns and synthetic benchmark column
    col_names = {col.name for col in rating_table.columns}
    assert "is_synthetic_benchmark" in col_names
    assert "rating" in col_names
    assert "user_id" in col_names
    assert "restaurant_id" in col_names


def test_user_preferences_foreign_key():
    """Verify UserPreferences has a cascade foreign key to users.id."""
    pref_table = Base.metadata.tables["user_preferences"]
    fk_targets = {fk.target_fullname for fk in pref_table.foreign_keys}
    assert "users.id" in fk_targets


def test_orm_model_instantiation():
    """Verify ORM model objects instantiate with default values and types."""
    rest = Restaurant(
        name="MTR 1924",
        city="Bengaluru",
        area="Basavanagudi",
        address="Lalbagh Road, Basavanagudi, Bangalore",
        cuisines="South Indian",
        restaurant_type="Quick Bites",
        cost_for_two_inr=250,
        price_tier="Budget",
        rating=4.5,
        review_count=1200,
        latitude=12.9416,
        longitude=77.5753
    )
    assert rest.name == "MTR 1924"
    assert rest.location_source == "Bengaluru locality centroid"
    assert rest.location_precision == "locality-level"
    assert rest.restaurant_id_source == "generated_surrogate_key"
    assert rest.online_order is False
    assert rest.book_table is False

    user = User(
        name="Aarav Sharma",
        email="aarav.sharma@example.in",
        city="Bengaluru",
        is_synthetic_benchmark=True
    )
    assert user.is_synthetic_benchmark is True

    rating = Rating(
        user_id=1,
        restaurant_id=1,
        rating=4.5,
        is_synthetic_benchmark=True
    )
    assert rating.is_synthetic_benchmark is True
