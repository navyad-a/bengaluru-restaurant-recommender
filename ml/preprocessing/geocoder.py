# -*- coding: utf-8 -*-
"""
Bengaluru Locality Centroid Geocoding Module
Maps authentic Bengaluru localities from the Zomato dataset to verified geographic coordinates.
Note: These represent locality centroid approximations, not exact building coordinates.
"""

from typing import Dict, Tuple, Optional

# Verified coordinate centroid approximations for all 93 Bengaluru localities
BENGALURU_LOCALITY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    'BTM': (12.9166, 77.6101),
    'Banashankari': (12.9255, 77.5468),
    'Banaswadi': (13.0142, 77.6519),
    'Bannerghatta Road': (12.8906, 77.5960),
    'Basavanagudi': (12.9416, 77.5753),
    'Basaveshwara Nagar': (12.9892, 77.5387),
    'Bellandur': (12.9304, 77.6784),
    'Bommanahalli': (12.9029, 77.6242),
    'Brigade Road': (12.9733, 77.6074),
    'Brookefield': (12.9654, 77.7186),
    'CV Raman Nagar': (12.9855, 77.6639),
    'Central Bangalore': (12.9716, 77.5946),
    'Church Street': (12.9750, 77.6050),
    'City Market': (12.9660, 77.5772),
    'Commercial Street': (12.9822, 77.6083),
    'Cunningham Road': (12.9880, 77.5960),
    'Domlur': (12.9609, 77.6387),
    'East Bangalore': (12.9719, 77.6412),
    'Ejipura': (12.9385, 77.6308),
    'Electronic City': (12.8399, 77.6770),
    'Frazer Town': (12.9968, 77.6130),
    'HBR Layout': (13.0359, 77.6324),
    'HSR': (12.9121, 77.6446),
    'Hebbal': (13.0358, 77.5970),
    'Hennur': (13.0400, 77.6400),
    'Hosur Road': (12.8900, 77.6400),
    'ITPL Main Road, Whitefield': (12.9856, 77.7315),
    'Indiranagar': (12.9784, 77.6408),
    'Infantry Road': (12.9820, 77.5980),
    'JP Nagar': (12.9063, 77.5857),
    'Jakkur': (13.0784, 77.6070),
    'Jalahalli': (13.0526, 77.5413),
    'Jayanagar': (12.9308, 77.5838),
    'Jeevan Bhima Nagar': (12.9669, 77.6575),
    'KR Puram': (13.0075, 77.6959),
    'Kaggadasapura': (12.9847, 77.6777),
    'Kalyan Nagar': (13.0221, 77.6403),
    'Kammanahalli': (13.0093, 77.6377),
    'Kanakapura Road': (12.8700, 77.5500),
    'Kengeri': (12.9177, 77.4838),
    'Koramangala': (12.9352, 77.6245),
    'Koramangala 1st Block': (12.9279, 77.6271),
    'Koramangala 2nd Block': (12.9240, 77.6200),
    'Koramangala 3rd Block': (12.9280, 77.6230),
    'Koramangala 4th Block': (12.9340, 77.6300),
    'Koramangala 5th Block': (12.9352, 77.6180),
    'Koramangala 6th Block': (12.9390, 77.6240),
    'Koramangala 7th Block': (12.9360, 77.6140),
    'Koramangala 8th Block': (12.9410, 77.6180),
    'Kumaraswamy Layout': (12.9081, 77.5552),
    'Langford Town': (12.9570, 77.6010),
    'Lavelle Road': (12.9710, 77.5990),
    'MG Road': (12.9756, 77.6066),
    'Magadi Road': (12.9750, 77.5350),
    'Majestic': (12.9767, 77.5713),
    'Malleshwaram': (13.0031, 77.5643),
    'Marathahalli': (12.9591, 77.6974),
    'Mysore Road': (12.9450, 77.5150),
    'Nagarbhavi': (12.9615, 77.5106),
    'Nagawara': (13.0435, 77.6187),
    'New BEL Road': (13.0382, 77.5703),
    'North Bangalore': (13.0300, 77.5800),
    'Old Airport Road': (12.9590, 77.6530),
    'Old Madras Road': (12.9900, 77.6600),
    'Peenya': (13.0329, 77.5273),
    'RT Nagar': (13.0247, 77.5948),
    'Race Course Road': (12.9830, 77.5840),
    'Rajajinagar': (12.9982, 77.5530),
    'Rajarajeshwari Nagar': (12.9274, 77.5154),
    'Rammurthy Nagar': (13.0163, 77.6785),
    'Residency Road': (12.9690, 77.6030),
    'Richmond Road': (12.9660, 77.6070),
    'Sadashiv Nagar': (13.0068, 77.5813),
    'Sahakara Nagar': (13.0623, 77.5919),
    'Sanjay Nagar': (13.0378, 77.5762),
    'Sankey Road': (12.9990, 77.5770),
    'Sarjapur Road': (12.9110, 77.6850),
    'Seshadripuram': (12.9910, 77.5760),
    'Shanti Nagar': (12.9540, 77.5980),
    'Shivajinagar': (12.9857, 77.6057),
    'South Bangalore': (12.9100, 77.5800),
    'St. Marks Road': (12.9730, 77.6020),
    'Thippasandra': (12.9738, 77.6546),
    'Ulsoor': (12.9817, 77.6284),
    'Uttarahalli': (12.9054, 77.5342),
    'Varthur Main Road, Whitefield': (12.9550, 77.7450),
    'Vasanth Nagar': (12.9900, 77.5900),
    'Vijay Nagar': (12.9719, 77.5304),
    'West Bangalore': (12.9800, 77.5400),
    'Whitefield': (12.9698, 77.7500),
    'Wilson Garden': (12.9480, 77.5960),
    'Yelahanka': (13.1007, 77.5963),
    'Yeshwantpur': (13.0238, 77.5529),
}

# Bengaluru default center fallback (MG Road coordinates)
BENGALURU_DEFAULT_CENTER = (12.9716, 77.5946)


def get_locality_coordinates(locality_name: str) -> Tuple[float, float]:
    """
    Returns the (latitude, longitude) centroid for a given Bengaluru locality.
    """
    loc_clean = str(locality_name).strip()
    return BENGALURU_LOCALITY_CENTROIDS.get(loc_clean, BENGALURU_DEFAULT_CENTER)
