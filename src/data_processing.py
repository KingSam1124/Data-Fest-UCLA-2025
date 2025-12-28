"""
Data Processing Pipeline for Manhattan Commercial Real Estate Analysis

This module handles the complete ETL pipeline:
1. Load raw lease data
2. Filter to Manhattan
3. Geocode addresses
4. Merge crime data for safety scores
5. Calculate accessibility scores
6. Apply pricing model
7. Aggregate to building level
"""

import pandas as pd
import numpy as np
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pricing model constants (derived from market analysis)
PRICING = {
    'base_rate': 75.09,        # $/SF base rent
    'face_discount': 0.998,    # 0.2% marketing discount
    'class_a_premium': 1.20,   # 20% premium for Class A
    'class_o_discount': 0.80,  # 20% discount for Class O
    'bulk_discount': 0.93,     # 7% off for >50,000 SF
    'bulk_threshold': 50000    # SF threshold for bulk discount
}


def load_raw_leases(filepath: Path) -> pd.DataFrame:
    """Load and perform initial cleaning of raw lease data."""
    logger.info(f"Loading raw leases from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} total leases")
    return df


def filter_manhattan(df: pd.DataFrame) -> pd.DataFrame:
    """Filter leases to Manhattan only."""
    # Normalize text columns
    df['city'] = df['city'].str.title()
    df['state'] = df['state'].str.upper()
    df['market'] = df['market'].str.title()

    # Filter for Manhattan/New York
    mask = (
        df['market'].str.contains('Manhattan', case=False, na=False) |
        df['city'].str.contains('New York', case=False, na=False)
    )

    df_manhattan = df[mask].copy()

    # Remove invalid records
    df_manhattan = df_manhattan[
        (df_manhattan['address'].notna()) &
        (df_manhattan['address'] != '') &
        (df_manhattan['leasedSF'].notna()) &
        (df_manhattan['leasedSF'] > 0)
    ]

    # Create full address for geocoding
    df_manhattan['full_address'] = (
        df_manhattan['address'] + ', ' +
        df_manhattan['city'] + ', ' +
        df_manhattan['state'] + ' ' +
        df_manhattan['zip'].astype(str)
    )

    logger.info(f"Filtered to {len(df_manhattan):,} Manhattan leases")
    return df_manhattan


def geocode_addresses(df: pd.DataFrame, cache_path: Path = None) -> pd.DataFrame:
    """
    Geocode addresses to get latitude/longitude.
    Uses caching to avoid re-geocoding known addresses.
    """
    if cache_path and cache_path.exists():
        logger.info("Loading cached geocoded data")
        return pd.read_csv(cache_path)

    logger.info("Geocoding addresses (this may take a while)...")

    geolocator = Nominatim(user_agent="manhattan_cre_analysis")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Get unique addresses to geocode
    unique_addresses = df['full_address'].unique()
    logger.info(f"Geocoding {len(unique_addresses)} unique addresses")

    address_coords = {}
    for i, addr in enumerate(unique_addresses):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(unique_addresses)}")
        try:
            location = geocode(addr)
            if location:
                address_coords[addr] = (location.latitude, location.longitude)
        except Exception as e:
            logger.warning(f"Failed to geocode {addr}: {e}")

    # Map coordinates back to dataframe
    df['latitude'] = df['full_address'].map(lambda x: address_coords.get(x, (None, None))[0])
    df['longitude'] = df['full_address'].map(lambda x: address_coords.get(x, (None, None))[1])

    # Drop rows without coordinates
    df = df.dropna(subset=['latitude', 'longitude'])

    if cache_path:
        df.to_csv(cache_path, index=False)
        logger.info(f"Cached geocoded data to {cache_path}")

    logger.info(f"Successfully geocoded {len(df):,} leases")
    return df


def merge_crime_data(df: pd.DataFrame, crime_path: Path) -> pd.DataFrame:
    """Merge crime data and calculate safety scores."""
    logger.info("Merging crime data")

    crime = pd.read_csv(crime_path)

    # Merge on coordinates
    df = df.merge(
        crime[['latitude', 'longitude', 'crime_score', 'crimes_within_500m']].drop_duplicates(),
        on=['latitude', 'longitude'],
        how='left'
    )

    # Fill missing crime scores with median (neutral assumption)
    df['crime_score'] = df['crime_score'].fillna(df['crime_score'].median())
    df['crimes_within_500m'] = df['crimes_within_500m'].fillna(0)

    # Calculate safety score (inverse of crime score, normalized 0-1)
    if df['crime_score'].max() > df['crime_score'].min():
        normalized_crime = (df['crime_score'] - df['crime_score'].min()) / \
                          (df['crime_score'].max() - df['crime_score'].min())
        df['safety_score'] = 1 - normalized_crime
    else:
        df['safety_score'] = 0.5

    logger.info(f"Crime data merged, safety scores calculated")
    return df


def calculate_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transit accessibility scores.
    Based on weighted transit route density near each location.
    """
    logger.info("Calculating accessibility scores")

    # If weighted_routes column exists, use it
    if 'weighted_routes' in df.columns:
        routes = pd.to_numeric(df['weighted_routes'], errors='coerce').fillna(0)
        if routes.max() > routes.min():
            df['accessibility_score'] = (routes - routes.min()) / (routes.max() - routes.min())
        else:
            df['accessibility_score'] = 0.5
    else:
        # Default to moderate accessibility for Manhattan
        df['accessibility_score'] = 0.7

    logger.info("Accessibility scores calculated")
    return df


def apply_pricing_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the pricing model to estimate annual rent.

    Model:
    - Base rate: $75.09/SF
    - Class A: +20% premium
    - Class O: -20% discount
    - Bulk (>50k SF): -7% discount
    """
    logger.info("Applying pricing model")

    # Calculate per-SF rates by class
    rate_a = PRICING['base_rate'] * PRICING['class_a_premium'] * PRICING['face_discount']
    rate_o = PRICING['base_rate'] * PRICING['class_o_discount'] * PRICING['face_discount']

    def calculate_rent(row):
        sf = row.get('total_leasedSF', row.get('leasedSF', 0))
        building_class = row.get('internal_class', 'O')

        if building_class == 'A':
            rate = rate_a
        else:
            rate = rate_o

        # Apply bulk discount
        if sf > PRICING['bulk_threshold']:
            rate *= PRICING['bulk_discount']

        return sf * rate

    df['estimated_annual_rent'] = df.apply(calculate_rent, axis=1)

    logger.info("Pricing model applied")
    return df


def aggregate_by_building(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lease data at the building level."""
    logger.info("Aggregating by building")

    building_agg = df.groupby(['latitude', 'longitude', 'full_address']).agg({
        'leasedSF': 'sum',
        'company_name': lambda x: '; '.join(x.dropna().unique()),
        'internal_industry': lambda x: '; '.join(x.dropna().unique()),
        'safety_score': 'first',
        'accessibility_score': 'first',
        'internal_class': 'first',
        'estimated_annual_rent': 'sum'
    }).reset_index()

    building_agg = building_agg.rename(columns={
        'leasedSF': 'total_leasedSF',
        'company_name': 'company_list',
        'internal_industry': 'sector_list'
    })

    # Count leases per building
    lease_counts = df.groupby(['latitude', 'longitude']).size().reset_index(name='lease_count')
    building_agg = building_agg.merge(lease_counts, on=['latitude', 'longitude'], how='left')

    # Calculate per-unit metrics for buyers
    building_agg['sf_per_unit'] = building_agg['total_leasedSF'] / building_agg['lease_count']
    building_agg['rent_per_unit'] = building_agg['estimated_annual_rent'] / building_agg['lease_count']

    logger.info(f"Aggregated to {len(building_agg):,} buildings")
    return building_agg


def run_pipeline(raw_data_path: Path, crime_data_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the complete data processing pipeline."""
    logger.info("Starting data processing pipeline")

    # Load and filter
    df = load_raw_leases(raw_data_path)
    df = filter_manhattan(df)

    # Geocode (with caching)
    cache_path = output_path.parent / 'manhattan_geocoded_cache.csv'
    df = geocode_addresses(df, cache_path)

    # Enrich with crime and accessibility
    df = merge_crime_data(df, crime_data_path)
    df = calculate_accessibility(df)

    # Apply pricing
    df = apply_pricing_model(df)

    # Aggregate by building
    buildings = aggregate_by_building(df)

    # Save output
    buildings.to_csv(output_path, index=False)
    logger.info(f"Pipeline complete! Output saved to {output_path}")

    return buildings


if __name__ == '__main__':
    # Default paths for running as script
    base_path = Path(__file__).parent.parent

    run_pipeline(
        raw_data_path=base_path / 'data' / 'raw' / 'Leases.csv',
        crime_data_path=base_path / 'data' / 'raw' / 'crime_manhattan_geo.csv',
        output_path=base_path / 'data' / 'processed' / 'manhattan_buildings.csv'
    )
