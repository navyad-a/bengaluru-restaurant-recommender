# -*- coding: utf-8 -*-
"""
User and UserPreferences SQLAlchemy 2.0 ORM Models
"""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, CheckConstraint, Index
)
from sqlalchemy.orm import relationship
from app.database.base import Base


class User(Base):
    """
    User entity in the recommendation system.
    Represents both real and synthetic benchmark user profiles.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    city = Column(String(50), nullable=False, default="Bengaluru")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_synthetic_benchmark = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    preferences = relationship(
        "UserPreferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    ratings = relationship(
        "Rating",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("latitude >= -90.0 AND latitude <= 90.0", name="chk_user_lat"),
        CheckConstraint("longitude >= -180.0 AND longitude <= 180.0", name="chk_user_lon"),
        Index("idx_users_city", "city"),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("city", "Bengaluru")
        kwargs.setdefault("is_synthetic_benchmark", False)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}', synthetic={self.is_synthetic_benchmark})>"


class UserPreferences(Base):
    """
    User dietary, cuisine, budget, and spatial preferences.
    Used for cold-start profiling and content-based matching.
    """
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    preferred_cuisines = Column(String(255), nullable=True)
    preferred_price_tier = Column(String(50), nullable=True)
    max_cost_for_two = Column(Integer, nullable=True)
    maximum_distance_km = Column(Float, nullable=False, default=10.0)
    online_order_only = Column(Boolean, default=False, nullable=False)
    book_table_only = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="preferences")

    __table_args__ = (
        CheckConstraint("max_cost_for_two >= 0", name="chk_user_pref_cost"),
        CheckConstraint("maximum_distance_km > 0", name="chk_user_pref_dist"),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("maximum_distance_km", 10.0)
        kwargs.setdefault("online_order_only", False)
        kwargs.setdefault("book_table_only", False)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<UserPreferences(user_id={self.user_id}, cuisines='{self.preferred_cuisines}', max_dist={self.maximum_distance_km}km)>"
