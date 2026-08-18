# -*- coding: utf-8 -*-
"""
Restaurant SQLAlchemy 2.0 ORM Model
Represents physical restaurant branches with authentic attributes and locality geocoding.
"""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, CheckConstraint, Index
)
from sqlalchemy.orm import relationship
from app.database.base import Base


class Restaurant(Base):
    """
    Physical restaurant outlet in Bengaluru.
    All attributes reflect authentic scraped catalog data with locality centroid coordinates.
    """
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(150), nullable=False, index=True)
    city = Column(String(50), nullable=False, default="Bengaluru", index=True)
    area = Column(String(100), nullable=False, index=True)
    address = Column(Text, nullable=False)
    cuisines = Column(String(255), nullable=False, index=True)
    restaurant_type = Column(String(100), nullable=False, index=True)
    cost_for_two_inr = Column(Integer, nullable=False, default=400)
    price_tier = Column(String(50), nullable=False, default="Moderate", index=True)
    rating = Column(Float, nullable=True, index=True)
    review_count = Column(Integer, nullable=False, default=0, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_source = Column(String(100), nullable=False, default="Bengaluru locality centroid")
    location_precision = Column(String(50), nullable=False, default="locality-level")
    restaurant_id_source = Column(String(50), nullable=False, default="generated_surrogate_key")
    online_order = Column(Boolean, nullable=False, default=False, index=True)
    book_table = Column(Boolean, nullable=False, default=False, index=True)
    dish_liked = Column(Text, nullable=True)
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
    ratings = relationship(
        "Rating",
        back_populates="restaurant",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("rating >= 1.0 AND rating <= 5.0", name="chk_rest_rating"),
        CheckConstraint("cost_for_two_inr >= 0", name="chk_rest_cost"),
        CheckConstraint("review_count >= 0", name="chk_rest_reviews"),
        CheckConstraint("latitude >= -90.0 AND latitude <= 90.0", name="chk_rest_lat"),
        CheckConstraint("longitude >= -180.0 AND longitude <= 180.0", name="chk_rest_lon"),
        Index("idx_rest_area_cuisine", "area", "cuisines"),
        Index("idx_rest_city_rating", "city", "rating"),
        Index("idx_rest_price_cost", "price_tier", "cost_for_two_inr"),
        Index("idx_rest_coords", "latitude", "longitude"),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("city", "Bengaluru")
        kwargs.setdefault("cost_for_two_inr", 400)
        kwargs.setdefault("price_tier", "Moderate")
        kwargs.setdefault("review_count", 0)
        kwargs.setdefault("location_source", "Bengaluru locality centroid")
        kwargs.setdefault("location_precision", "locality-level")
        kwargs.setdefault("restaurant_id_source", "generated_surrogate_key")
        kwargs.setdefault("online_order", False)
        kwargs.setdefault("book_table", False)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return f"<Restaurant(id={self.id}, name='{self.name}', area='{self.area}', rating={self.rating}, cost_for_two=Rs.{self.cost_for_two_inr})>"
