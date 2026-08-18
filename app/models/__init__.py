# -*- coding: utf-8 -*-
"""
SQLAlchemy 2.0 ORM Models Export Package
"""

from app.models.user import User, UserPreferences
from app.models.restaurant import Restaurant
from app.models.rating import Rating

__all__ = ["User", "UserPreferences", "Restaurant", "Rating"]
