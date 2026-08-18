# -*- coding: utf-8 -*-
"""
User Onboarding & Profile Initialization Module
===============================================
Transforms initial onboarding questionnaire inputs into a structured preference profile
for instant zero-history recommendation bootstrapping.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class OnboardingQuestionnaire(BaseModel):
    """
    Onboarding questionnaire responses from a new user.
    """
    favorite_cuisines: List[str] = Field(
        ...,
        min_length=1,
        description="List of preferred regional/Indian cuisines (e.g. ['South Indian', 'Karnataka'])"
    )
    preferred_dining_types: Optional[List[str]] = Field(
        default=None,
        description="Preferred dining formats (e.g. ['Quick Bites', 'Cafe', 'Casual Dining'])"
    )
    preferred_area: Optional[str] = Field(
        default=None,
        description="Target Bengaluru neighborhood (e.g. 'Koramangala 5th Block', 'Indiranagar')"
    )
    price_tier: Optional[str] = Field(
        default="Moderate",
        description="Budget, Moderate, Premium, or Luxury"
    )
    max_budget_for_two: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum budget for two in INR (₹)"
    )
    is_pure_veg_preferred: Optional[bool] = Field(
        default=False,
        description="Preference for pure vegetarian outlets"
    )
    online_ordering: Optional[bool] = Field(
        default=False,
        description="Filter for restaurants offering online delivery"
    )


class OnboardingPreferenceHandler:
    """
    Constructs a warm user profile from onboarding questionnaire data.
    """

    @staticmethod
    def build_preference_payload(
        questionnaire: OnboardingQuestionnaire
    ) -> Dict[str, Any]:
        """
        Translates questionnaire into standardized recommender preference format.
        """
        cuisines_str = ", ".join(questionnaire.favorite_cuisines)
        types_str = ", ".join(questionnaire.preferred_dining_types) if questionnaire.preferred_dining_types else None

        prefs = {
            "preferred_cuisines": cuisines_str,
            "preferred_type": types_str,
            "preferred_area": questionnaire.preferred_area,
            "preferred_price_tier": questionnaire.price_tier,
            "max_cost_for_two": questionnaire.max_budget_for_two,
            "online_order_only": questionnaire.online_ordering
        }

        if questionnaire.is_pure_veg_preferred:
            prefs["preferred_cuisines"] = cuisines_str + ", Pure Vegetarian"

        # Strip None values
        return {k: v for k, v in prefs.items() if v is not None}
