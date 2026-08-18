# -*- coding: utf-8 -*-
"""
Rating SQLAlchemy 2.0 ORM Model
Explicit user-restaurant interaction record.
All benchmark ratings are explicitly flagged with is_synthetic_benchmark=True.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, Boolean, DateTime,
    Text, ForeignKey, CheckConstraint, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from app.database.base import Base


class Rating(Base):
    """
    Explicit rating record linking a user and a restaurant.
    Contains is_synthetic_benchmark flag to ensure strict isolation of simulated benchmark records.
    """
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    restaurant_id = Column(
        Integer,
        ForeignKey("restaurants.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    rating = Column(Float, nullable=False)
    review_text = Column(Text, nullable=True)
    is_synthetic_benchmark = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="ratings")
    restaurant = relationship("Restaurant", back_populates="ratings")

    __table_args__ = (
        CheckConstraint("rating >= 1.0 AND rating <= 5.0", name="chk_rating_range"),
        UniqueConstraint("user_id", "restaurant_id", name="uq_user_restaurant_rating"),
        Index("idx_ratings_user_synthetic", "user_id", "is_synthetic_benchmark"),
        Index("idx_ratings_rest_synthetic", "restaurant_id", "is_synthetic_benchmark"),
    )

    def __repr__(self) -> str:
        return f"<Rating(user_id={self.user_id}, restaurant_id={self.restaurant_id}, rating={self.rating}, synthetic={self.is_synthetic_benchmark})>"
