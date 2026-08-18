# -*- coding: utf-8 -*-
"""
Recommendation API Endpoints (Content-Based, Collaborative SVD, & Hybrid)
Optimized for high-concurrency async performance with threadpool offloading and caching.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from starlette.concurrency import run_in_threadpool
from app.schemas.recommendation import (
    ContentPreferenceRequest,
    RecommendationResponse,
    RestaurantRecommendationItem,
    CollaborativeRecommendationResponse,
    CollaborativeRecommendationItem,
    HybridRecommendationRequest,
    HybridRecommendationResponse,
    HybridRecommendationItem,
    NearbyRecommendationResponse,
    NearbyRecommendationItem,
    PopularRecommendationResponse,
    PopularRecommendationItem,
    OnboardingRequest
)
from app.services.recommendation_service import (
    get_content_recommender,
    get_collaborative_recommender,
    get_hybrid_recommender,
    get_spatial_search_engine
)
from app.core.cache import get_recommendation_cache
from ml.spatial.coordinates import validate_coordinates
from ml.cold_start.onboarding import OnboardingQuestionnaire, OnboardingPreferenceHandler

router = APIRouter()


@router.get(
    "/similar/{restaurant_id}",
    response_model=RecommendationResponse,
    summary="Get similar restaurants (Mode A: Restaurant-to-Restaurant)",
    description="Returns top-K restaurants with the highest TF-IDF metadata cosine similarity to the given restaurant ID."
)
async def get_similar_restaurants(
    restaurant_id: int,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key("similar", restaurant_id=restaurant_id, top_k=top_k)
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        recommender = get_content_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Content recommendation engine unavailable: {str(e)}"
        )

    # Validate restaurant exists
    rest = recommender.get_restaurant_by_id(restaurant_id)
    if not rest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant with ID {restaurant_id} not found in the 12,481 Bengaluru catalog."
        )

    recs = await run_in_threadpool(
        recommender.recommend_similar_restaurants,
        restaurant_id=restaurant_id,
        top_k=top_k
    )

    items = [RestaurantRecommendationItem(**r) for r in recs]
    response = RecommendationResponse(
        query_type=f"similar_restaurant (ID: {restaurant_id} - {rest['name']})",
        count=len(items),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.post(
    "/content",
    response_model=RecommendationResponse,
    summary="Get recommendations from user preferences (Mode B: Preference-to-Restaurant)",
    description="Matches user preferences against restaurant TF-IDF feature vectors with support for hard constraint filters."
)
async def get_content_recommendations_for_preferences(
    payload: ContentPreferenceRequest
):
    prefs_dict = payload.model_dump(exclude_unset=True)
    top_k = payload.top_k

    cache = get_recommendation_cache()
    cache_key = cache.generate_key("content_pref", **prefs_dict)
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        recommender = get_content_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Content recommendation engine unavailable: {str(e)}"
        )

    recs = await run_in_threadpool(
        recommender.recommend_for_preferences,
        preferences=prefs_dict,
        top_k=top_k
    )

    items = [RestaurantRecommendationItem(**r) for r in recs]
    response = RecommendationResponse(
        query_type="preference_matching",
        count=len(items),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.get(
    "/collaborative/{user_id}",
    response_model=CollaborativeRecommendationResponse,
    summary="Get personalized Collaborative Filtering recommendations (Surprise SVD)",
    description=(
        "Returns top-K personalized restaurant recommendations for a known user using Surprise SVD "
        "trained on the Synthetic Collaborative Filtering Benchmark. Excludes restaurants already rated."
    )
)
async def get_collaborative_recommendations(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key("collab", user_id=user_id, top_k=top_k)
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        cf_recommender = get_collaborative_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Collaborative SVD recommendation engine unavailable: {str(e)}"
        )

    if not cf_recommender.is_known_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"User ID {user_id} is unknown to Collaborative SVD (cold-start user). "
                "Collaborative Filtering requires historical interaction ratings."
            )
        )

    recs = await run_in_threadpool(
        cf_recommender.recommend_for_user,
        user_id=user_id,
        top_k=top_k,
        exclude_rated=True
    )

    items = [CollaborativeRecommendationItem(**r) for r in recs]
    response = CollaborativeRecommendationResponse(
        status="success",
        user_id=user_id,
        count=len(items),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.get(
    "/nearby",
    response_model=NearbyRecommendationResponse,
    summary="Get nearest restaurants using Spatial BallTree Index",
    description="Finds nearest restaurants or restaurants within a specified radius using great-circle Haversine distance."
)
async def get_nearby_restaurants(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="User latitude in degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="User longitude in degrees"),
    radius_km: Optional[float] = Query(default=None, ge=0.1, le=50.0, description="Optional search radius in km"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    try:
        validate_coordinates(latitude, longitude)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid geographic coordinates: {str(ve)}"
        )

    cache = get_recommendation_cache()
    cache_key = cache.generate_key("nearby", latitude=latitude, longitude=longitude, radius_km=radius_km, top_k=top_k)
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        spatial_engine = get_spatial_search_engine()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Spatial search engine unavailable: {str(e)}"
        )

    if radius_km is not None:
        recs = await run_in_threadpool(
            spatial_engine.find_nearest_within_radius,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius_km,
            top_k=top_k
        )
    else:
        recs = await run_in_threadpool(
            spatial_engine.find_nearest,
            latitude=latitude,
            longitude=longitude,
            top_k=top_k
        )

    items = [
        NearbyRecommendationItem(
            restaurant_id=r["restaurant_id"],
            name=r["name"],
            distance_km=r["distance_km"],
            rating=r.get("rating"),
            review_count=r.get("review_count", 0),
            cuisines=r.get("cuisines", ""),
            restaurant_type=r.get("rest_type", r.get("restaurant_type", "Restaurant")),
            area=r.get("area", ""),
            address=r.get("address", ""),
            cost_for_two_inr=r.get("cost_for_two_inr", 0),
            price_tier=r.get("price_tier", "Moderate"),
            online_order=r.get("online_order", False),
            book_table=r.get("book_table", False),
            latitude=r.get("latitude", 0.0),
            longitude=r.get("longitude", 0.0),
            location_source=r.get("location_source", "Bengaluru locality centroid"),
            location_precision=r.get("location_precision", "locality-level")
        )
        for r in recs
    ]

    response = NearbyRecommendationResponse(
        status="success",
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        count=len(items),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.get(
    "/hybrid/{user_id}",
    response_model=HybridRecommendationResponse,
    summary="Get personalized Hybrid recommendations (GET by user ID)",
    description=(
        "Generates hybrid recommendations combining Content, Collaborative SVD, Location, and Bayesian Quality. "
        "Gracefully falls back for cold-start unknown users by dynamically redistributing weights."
    )
)
async def get_hybrid_recommendations_for_user(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    area: Optional[str] = Query(default=None, description="Optional hard filter for Bengaluru locality"),
    max_cost_for_two: Optional[int] = Query(default=None, ge=0, description="Optional hard budget constraint"),
    min_rating: Optional[float] = Query(default=None, ge=1.0, le=5.0, description="Optional hard minimum rating constraint"),
    price_tier: Optional[str] = Query(default=None, description="Optional hard price tier filter"),
    online_order_only: Optional[bool] = Query(default=False, description="Optional online order filter"),
    book_table_only: Optional[bool] = Query(default=False, description="Optional table booking filter"),
    latitude: Optional[float] = Query(default=None, ge=-90.0, le=90.0, description="User latitude for distance scoring"),
    longitude: Optional[float] = Query(default=None, ge=-180.0, le=180.0, description="User longitude for distance scoring"),
    mmr_enabled: Optional[bool] = Query(default=True, description="Enable MMR diversification"),
    mmr_lambda: Optional[float] = Query(default=0.75, ge=0.0, le=1.0, description="MMR lambda parameter")
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key(
        "hybrid_get",
        user_id=user_id,
        top_k=top_k,
        area=area,
        max_cost_for_two=max_cost_for_two,
        min_rating=min_rating,
        price_tier=price_tier,
        online_order_only=online_order_only,
        book_table_only=book_table_only,
        latitude=latitude,
        longitude=longitude,
        mmr_enabled=mmr_enabled,
        mmr_lambda=mmr_lambda
    )
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        hybrid_engine = get_hybrid_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hybrid recommendation engine unavailable: {str(e)}"
        )

    filters = {}
    if max_cost_for_two is not None:
        filters["max_cost_for_two"] = max_cost_for_two
    if min_rating is not None:
        filters["min_rating"] = min_rating
    if price_tier is not None:
        filters["price_tier"] = price_tier
    if area is not None:
        filters["area"] = area
    if online_order_only:
        filters["online_order_only"] = True
    if book_table_only:
        filters["book_table_only"] = True

    user_coords = (latitude, longitude) if (latitude is not None and longitude is not None) else None

    result = await run_in_threadpool(
        hybrid_engine.recommend,
        user_id=user_id,
        user_coords=user_coords,
        filters=filters if filters else None,
        mmr_enabled=mmr_enabled if mmr_enabled is not None else True,
        mmr_lambda=mmr_lambda if mmr_lambda is not None else 0.75,
        top_k=top_k
    )

    items = [HybridRecommendationItem(**r) for r in result["recommendations"]]
    response = HybridRecommendationResponse(
        status="success",
        user_id=result["user_id"],
        is_cold_start=result["is_cold_start"],
        strategy=result.get("strategy"),
        model_source=result["model_source"],
        effective_weights=result["effective_weights"],
        count=len(items),
        diversification=result.get("diversification"),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.post(
    "/hybrid",
    response_model=HybridRecommendationResponse,
    summary="Get Hybrid recommendations with custom preferences and weights (POST)",
    description="Flexible endpoint supporting preferences, coordinates, custom weights, and hard constraints."
)
async def get_hybrid_recommendations_post(
    payload: HybridRecommendationRequest
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key("hybrid_post", **payload.model_dump())
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        hybrid_engine = get_hybrid_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Hybrid recommendation engine unavailable: {str(e)}"
        )

    filters = {}
    if payload.max_cost_for_two is not None:
        filters["max_cost_for_two"] = payload.max_cost_for_two
    if payload.min_rating is not None:
        filters["min_rating"] = payload.min_rating
    if payload.preferred_price_tier is not None:
        filters["price_tier"] = payload.preferred_price_tier
    if payload.preferred_area is not None:
        filters["area"] = payload.preferred_area
    if payload.online_order_only:
        filters["online_order_only"] = True
    if payload.book_table_only:
        filters["book_table_only"] = True
    if payload.radius_km is not None:
        filters["radius_km"] = payload.radius_km

    user_coords = (
        (payload.latitude, payload.longitude)
        if (payload.latitude is not None and payload.longitude is not None)
        else None
    )

    preferences = {}
    if payload.preferred_cuisines:
        preferences["preferred_cuisines"] = payload.preferred_cuisines
    if payload.preferred_price_tier:
        preferences["preferred_price_tier"] = payload.preferred_price_tier
    if payload.preferred_type:
        preferences["preferred_type"] = payload.preferred_type
    if payload.preferred_area:
        preferences["preferred_area"] = payload.preferred_area

    result = await run_in_threadpool(
        hybrid_engine.recommend,
        user_id=payload.user_id,
        preferences=preferences if preferences else None,
        target_restaurant_id=payload.target_restaurant_id,
        user_coords=user_coords,
        filters=filters if filters else None,
        weights=payload.custom_weights,
        mmr_enabled=payload.mmr_enabled if payload.mmr_enabled is not None else True,
        mmr_lambda=payload.mmr_lambda if payload.mmr_lambda is not None else 0.75,
        top_k=payload.top_k
    )

    items = [HybridRecommendationItem(**r) for r in result["recommendations"]]
    response = HybridRecommendationResponse(
        status="success",
        user_id=result["user_id"],
        is_cold_start=result["is_cold_start"],
        strategy=result.get("strategy"),
        model_source=result["model_source"],
        effective_weights=result["effective_weights"],
        count=len(items),
        diversification=result.get("diversification"),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.get(
    "/popular",
    response_model=PopularRecommendationResponse,
    summary="Get popular restaurants (Global or by Locality/Cuisine)",
    description="Returns top-ranked popular restaurants using Bayesian popularity priors."
)
async def get_popular_restaurants(
    area: Optional[str] = Query(default=None, description="Optional Bengaluru neighborhood filter"),
    cuisine: Optional[str] = Query(default=None, description="Optional cuisine filter (e.g. 'South Indian', 'Biryani')"),
    max_cost_for_two: Optional[int] = Query(default=None, ge=0, description="Optional budget filter in INR"),
    min_rating: Optional[float] = Query(default=None, ge=1.0, le=5.0, description="Optional minimum rating"),
    online_order_only: Optional[bool] = Query(default=False, description="Online delivery only"),
    book_table_only: Optional[bool] = Query(default=False, description="Table booking only"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key(
        "popular",
        area=area,
        cuisine=cuisine,
        max_cost_for_two=max_cost_for_two,
        min_rating=min_rating,
        online_order_only=online_order_only,
        book_table_only=book_table_only,
        top_k=top_k
    )
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        hybrid_engine = get_hybrid_recommender()
        pop_engine = hybrid_engine.popularity_engine
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Popularity engine unavailable: {str(e)}"
        )

    filters = {}
    if max_cost_for_two is not None:
        filters["max_cost_for_two"] = max_cost_for_two
    if min_rating is not None:
        filters["min_rating"] = min_rating
    if online_order_only:
        filters["online_order_only"] = True
    if book_table_only:
        filters["book_table_only"] = True

    if area:
        recs = await run_in_threadpool(pop_engine.get_locality_popular, area=area, top_k=top_k, filters=filters)
        scope = f"locality: {area}"
    elif cuisine:
        recs = await run_in_threadpool(pop_engine.get_cuisine_popular, cuisine=cuisine, top_k=top_k, filters=filters)
        scope = f"cuisine: {cuisine}"
    else:
        recs = await run_in_threadpool(pop_engine.get_global_popular, top_k=top_k, filters=filters)
        scope = "global_bengaluru"

    items = [
        PopularRecommendationItem(
            restaurant_id=r["restaurant_id"],
            name=r["name"],
            popularity_score=r.get("popularity_score", 0.0),
            rating=r.get("rating"),
            review_count=r.get("review_count", 0),
            cuisines=r.get("cuisines", ""),
            restaurant_type=r.get("rest_type", r.get("restaurant_type", "Restaurant")),
            area=r.get("area", ""),
            address=r.get("address", ""),
            cost_for_two_inr=r.get("cost_for_two_inr", 0),
            price_tier=r.get("price_tier", "Moderate"),
            online_order=r.get("online_order", False),
            book_table=r.get("book_table", False),
            location_source=r.get("location_source", "Bengaluru locality centroid"),
            location_precision=r.get("location_precision", "locality-level")
        )
        for r in recs
    ]

    response = PopularRecommendationResponse(
        status="success",
        scope=scope,
        count=len(items),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


@router.post(
    "/onboarding",
    response_model=HybridRecommendationResponse,
    summary="Cold-Start Onboarding Recommendation",
    description="Takes new user onboarding choices and boots an instant personalized hybrid recommendation."
)
async def get_onboarding_recommendations(
    payload: OnboardingRequest
):
    cache = get_recommendation_cache()
    cache_key = cache.generate_key("onboarding", **payload.model_dump())
    cached_res = cache.get(cache_key)
    if cached_res is not None:
        return cached_res

    try:
        hybrid_engine = get_hybrid_recommender()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation engine unavailable: {str(e)}"
        )

    q = OnboardingQuestionnaire(
        favorite_cuisines=payload.favorite_cuisines,
        preferred_dining_types=payload.preferred_dining_types,
        preferred_area=payload.preferred_area,
        price_tier=payload.price_tier,
        max_budget_for_two=payload.max_budget_for_two,
        is_pure_veg_preferred=payload.is_pure_veg_preferred,
        online_ordering=payload.online_ordering
    )

    prefs = OnboardingPreferenceHandler.build_preference_payload(q)

    user_coords = (
        (payload.latitude, payload.longitude)
        if (payload.latitude is not None and payload.longitude is not None)
        else None
    )

    filters = {}
    if payload.max_budget_for_two is not None:
        filters["max_cost_for_two"] = payload.max_budget_for_two
    if payload.online_ordering:
        filters["online_order_only"] = True
    if payload.preferred_area:
        filters["area"] = payload.preferred_area

    result = await run_in_threadpool(
        hybrid_engine.recommend,
        user_id=None,  # Brand new cold-start user
        preferences=prefs,
        user_coords=user_coords,
        filters=filters if filters else None,
        mmr_enabled=payload.mmr_enabled if payload.mmr_enabled is not None else True,
        mmr_lambda=payload.mmr_lambda if payload.mmr_lambda is not None else 0.75,
        top_k=payload.top_k
    )

    items = [HybridRecommendationItem(**r) for r in result["recommendations"]]
    response = HybridRecommendationResponse(
        status="success",
        user_id=None,
        is_cold_start=True,
        strategy=result.get("strategy", "profile_content"),
        model_source="hybrid",
        effective_weights=result["effective_weights"],
        count=len(items),
        diversification=result.get("diversification"),
        recommendations=items
    )
    cache.set(cache_key, response)
    return response


