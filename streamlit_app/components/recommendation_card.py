# -*- coding: utf-8 -*-
"""
Recommendation Card Component: Google Play Store Style Restaurant Presentation
"""

import streamlit as st
from typing import Dict, Any, Union
from streamlit_app.config import CURRENCY_SYMBOL
from streamlit_app.components.explanation_card import render_explanation_card


def _get_cuisine_icon(cuisines_str: str, rest_type: str) -> str:
    """Returns a matching Google Play style food emoji icon."""
    combined = f"{cuisines_str} {rest_type}".lower()
    if any(k in combined for k in ["cafe", "coffee", "bakery", "dessert", "ice cream"]):
        return "☕"
    elif any(k in combined for k in ["biryani", "mughlai", "kebab", "north indian"]):
        return "🍲"
    elif any(k in combined for k in ["south indian", "dosa", "idli", "andhra", "chettinad"]):
        return "🍛"
    elif any(k in combined for k in ["pizza", "italian", "pasta"]):
        return "🍕"
    elif any(k in combined for k in ["burger", "fast food", "sandwich"]):
        return "🍔"
    elif any(k in combined for k in ["chinese", "asian", "momos", "thai", "japanese"]):
        return "🥢"
    elif any(k in combined for k in ["pub", "bar", "brewery", "lounge"]):
        return "🍺"
    elif any(k in combined for k in ["seafood", "fish"]):
        return "🐟"
    return "🍽️"


def render_recommendation_card(item: Dict[str, Any], rank: int = 1):
    """
    Renders a single recommended restaurant item as a Google Play Store style discovery card.
    Uses clean, unindented HTML to prevent Markdown parser from converting tags to code blocks.
    """
    name = item.get("name", "Unknown Restaurant")
    rating = item.get("rating")
    reviews = item.get("review_count", 0)
    
    # Rating badge styling (Google Play green for good, amber for moderate, gray for unrated)
    if rating and rating > 0:
        rating_val_str = f"★ {rating:.1f}"
        if rating >= 3.8:
            rating_badge_class = "play-rating-badge"
        elif rating >= 3.0:
            rating_badge_class = "play-rating-badge-amber"
        else:
            rating_badge_class = "play-rating-badge-gray"
    else:
        rating_val_str = "★ New"
        rating_badge_class = "play-rating-badge-gray"

    # Cuisines parsing
    raw_cuisines = item.get("cuisines", "Multi-Cuisine")
    if isinstance(raw_cuisines, list):
        cuisines_list = raw_cuisines
        cuisines_display = ", ".join(raw_cuisines)
    else:
        cuisines_display = str(raw_cuisines)
        cuisines_list = [c.strip() for c in cuisines_display.split(",") if c.strip()]

    area = item.get("area", "Bengaluru")
    rest_type = item.get("restaurant_type") or item.get("rest_type", "Restaurant")
    cost_for_two = item.get("cost_for_two_inr", 0)
    price_tier = item.get("price_tier", "Moderate")
    online_order = item.get("online_order", False)
    book_table = item.get("book_table", False)
    distance_km = item.get("distance_km")
    
    icon_emoji = _get_cuisine_icon(cuisines_display, rest_type)

    # Build Play Store chips
    chips_html = []
    for c in cuisines_list[:3]:
        chips_html.append(f'<span class="play-chip">{c}</span>')
    
    if distance_km is not None:
        chips_html.append(f'<span class="play-chip play-chip-slate">📍 ~{distance_km:.1f} km</span>')
    if online_order:
        chips_html.append('<span class="play-chip play-chip-green">✓ Online Order</span>')
    if book_table:
        chips_html.append('<span class="play-chip play-chip-blue">✓ Table Booking</span>')

    chips_str = "".join(chips_html)

    # Clean unindented HTML string to ensure no markdown code formatting
    card_html = (
        f'<div class="play-card">'
        f'  <div class="play-card-header">'
        f'    <div class="play-card-left">'
        f'      <div class="play-icon-box">{icon_emoji}</div>'
        f'      <div class="play-title-group">'
        f'        <span class="play-rank-label">#{rank} Ranked</span>'
        f'        <h3 class="play-restaurant-name">{name}</h3>'
        f'        <div class="play-subtitle">'
        f'          <span>📍 {area}</span>'
        f'          <span class="play-bullet">&bull;</span>'
        f'          <span>🍴 {rest_type}</span>'
        f'        </div>'
        f'      </div>'
        f'    </div>'
        f'    <div>'
        f'      <div class="play-price-badge">{CURRENCY_SYMBOL}{cost_for_two:,}</div>'
        f'      <div class="play-price-sub">for two &bull; {price_tier}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="play-rating-row">'
        f'    <span class="{rating_badge_class}">{rating_val_str}</span>'
        f'    <span class="play-reviews-text">({reviews:,} community reviews)</span>'
        f'    <span class="play-bullet">&bull;</span>'
        f'    <span class="play-reviews-text">{price_tier} Tier</span>'
        f'  </div>'
        f'  <div class="play-chips-row">'
        f'    {chips_str}'
        f'  </div>'
        f'</div>'
    )

    with st.container():
        st.markdown(card_html, unsafe_allow_html=True)

        # Embedded explanation
        explanation = item.get("explanation")
        explanation_metadata = item.get("explanation_metadata")
        if explanation or explanation_metadata:
            render_explanation_card(
                explanation=explanation,
                metadata=explanation_metadata,
                item=item
            )


