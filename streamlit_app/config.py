# -*- coding: utf-8 -*-
"""
Streamlit App Configuration & Catalog Constants
"""

import os
from typing import List

# API Connection
API_BASE_URL: str = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# App Branding
APP_TITLE: str = "Bengaluru Restaurant Intelligence"
APP_SUBTITLE: str = "AI-powered restaurant recommendations personalized to your taste, location, budget, and dining preferences."
APP_ICON: str = "🍽️"
APP_VERSION: str = "1.0.0"

# Currency
CURRENCY_SYMBOL: str = "₹"

# Authentic Catalog Localities (Top 30 Bengaluru Areas)
BENGALURU_LOCALITIES: List[str] = [
    "Indiranagar",
    "Koramangala 5th Block",
    "HSR",
    "BTM",
    "Whitefield",
    "JP Nagar",
    "Jayanagar",
    "Bellandur",
    "Marathahalli",
    "Electronic City",
    "Bannerghatta Road",
    "Sarjapur Road",
    "New BEL Road",
    "Banashankari",
    "Kalyan Nagar",
    "Malleshwaram",
    "Rajajinagar",
    "Basavanagudi",
    "Banaswadi",
    "Frazer Town",
    "Church Street",
    "MG Road",
    "Brigade Road",
    "Lavelle Road",
    "Residency Road",
    "Cunningham Road",
    "Ulsoor",
    "Richmond Town",
    "Domlur",
    "Old Airport Road"
]

# Prominent Authentic Cuisines
POPULAR_CUISINES: List[str] = [
    "Biryani",
    "North Indian",
    "South Indian",
    "Chinese",
    "Fast Food",
    "Continental",
    "Cafe",
    "Italian",
    "Andhra",
    "Mughlai",
    "Kerala",
    "Desserts",
    "Street Food",
    "Bakery",
    "Pizza",
    "Burger",
    "Seafood",
    "Arabian",
    "Momos",
    "Rolls",
    "Asian",
    "Thai"
]

# Price Tiers
PRICE_TIERS: List[str] = [
    "Budget",
    "Moderate",
    "Premium",
    "Luxury"
]

# Dining Formats
DINING_TYPES: List[str] = [
    "Casual Dining",
    "Quick Bites",
    "Cafe",
    "Delivery",
    "Dessert Parlor",
    "Bakery",
    "Food Court",
    "Bar",
    "Fine Dining"
]

# Budget Preset Filter Options (in INR ₹ for two)
BUDGET_OPTIONS: List[int] = [300, 500, 700, 1000, 1500, 2000, 3000]

# Default MMR & Retrieval Parameters
DEFAULT_TOP_K: int = 10
DEFAULT_MMR_ENABLED: bool = True
DEFAULT_MMR_LAMBDA: float = 0.75
DEFAULT_SEARCH_RADIUS_KM: float = 3.0

# Coordinate Defaults (Bengaluru City Center: Vidhana Soudha / MG Road)
DEFAULT_LATITUDE: float = 12.9716
DEFAULT_LONGITUDE: float = 77.5946

