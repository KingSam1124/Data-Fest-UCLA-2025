"""
Manhattan CRE Buyer-Side Explorer

An interactive Streamlit application for exploring Manhattan commercial
real estate leasing opportunities.

Features:
- Filter by square footage, safety score, and accessibility
- Interactive map with color-coded markers
- Optional ZIP-code choropleth showing industry market friendliness

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, BeautifyIcon
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from pathlib import Path
import json

st.set_page_config(
    page_title="Manhattan CRE Explorer",
    layout="wide"
)

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'


@st.cache_data
def load_building_data() -> pd.DataFrame:
    """Load and prepare building-level lease data."""
    # Try multiple possible data files
    possible_files = [
        DATA_DIR / 'manhattan_buildings.csv',
        DATA_DIR / 'manhattan_geo_access_price.csv',
        Path(__file__).parent / 'manhattan_geo_access_price.csv',
    ]

    df = None
    for filepath in possible_files:
        if filepath.exists():
            df = pd.read_csv(filepath)
            break

    if df is None:
        st.error("No data file found. Please run the data processing pipeline first.")
        st.stop()

    df = df.dropna(subset=['latitude', 'longitude']).copy()

    # Standardize column names
    if 'total_leasedSF' in df.columns:
        df.rename(columns={'total_leasedSF': 'totalSF'}, inplace=True)
    if 'totalSF' not in df.columns and 'leasedSF' in df.columns:
        df.rename(columns={'leasedSF': 'totalSF'}, inplace=True)

    # Calculate safety score if not present
    if 'safety_score' not in df.columns:
        if 'crime_score' in df.columns:
            c = pd.to_numeric(df['crime_score'], errors='coerce').fillna(0)
            if c.max() > c.min():
                c = (c - c.min()) / (c.max() - c.min())
            df['safety_score'] = 1 - c
        else:
            df['safety_score'] = 0.5

    # Calculate accessibility score if not present
    if 'accessibility_score' not in df.columns:
        acc_cols = [c for c in df.columns if 'access' in c.lower() or 'route' in c.lower()]
        if acc_cols:
            a = pd.to_numeric(df[acc_cols[0]], errors='coerce').fillna(0)
            if a.max() > a.min():
                df['accessibility_score'] = (a - a.min()) / (a.max() - a.min())
            else:
                df['accessibility_score'] = 0.5
        else:
            df['accessibility_score'] = 0.7  # Manhattan default

    # Calculate per-unit metrics
    if 'lease_count' in df.columns:
        df['sf_per_unit'] = np.where(
            df['lease_count'] > 0,
            df['totalSF'] / df['lease_count'],
            df['totalSF']
        )
        rent_col = 'estimated_annual_rent' if 'estimated_annual_rent' in df.columns else 'price_multiplier'
        if rent_col in df.columns:
            df['rent_per_unit'] = np.where(
                df['lease_count'] > 0,
                df[rent_col] / df['lease_count'],
                df[rent_col]
            )
    else:
        df['sf_per_unit'] = df['totalSF']
        df['lease_count'] = 1

    # Extract ZIP code
    if 'zip' not in df.columns and 'full_address' in df.columns:
        df['zip'] = df['full_address'].str.extract(r'\b(\d{5})\b')

    return df


@st.cache_data
def load_zip_scores() -> pd.DataFrame:
    """Load ZIP-code sector friendliness scores."""
    possible_files = [
        DATA_DIR / 'zip_sector_scores.csv',
        Path(__file__).parent / 'zip_sector_scores.csv',
    ]

    for filepath in possible_files:
        if filepath.exists():
            z = pd.read_csv(filepath)
            z['zip'] = z['zip'].astype(str).str.zfill(5)
            return z

    return pd.DataFrame()


def get_accessibility_icon(score: float) -> str:
    """Map accessibility score to FontAwesome icon."""
    if score >= 0.80:
        return "subway"
    elif score >= 0.60:
        return "bus"
    elif score >= 0.40:
        return "train"
    elif score >= 0.20:
        return "bicycle"
    else:
        return "walking"


def create_map(df: pd.DataFrame, show_choropleth: bool = False,
               sector: str = None, zip_scores: pd.DataFrame = None) -> folium.Map:
    """Create the interactive Folium map."""
    m = folium.Map(
        location=[40.75, -73.97],
        zoom_start=12,
        tiles='cartodbpositron'
    )

    # Safety color scale
    cmap = LinearColormap(
        ['#e74c3c', '#f39c12', '#f1c40f', '#90EE90', '#27ae60'],
        vmin=0, vmax=1
    )
    cmap.caption = 'Safety Score (red = higher risk, green = safer)'

    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        rent_display = row.get('rent_per_unit', 0)
        if rent_display > 1_000_000:
            rent_str = f"${rent_display / 1_000_000:.2f}M"
        else:
            rent_str = f"${rent_display:,.0f}"

        popup_html = f"""
            <b>{row['full_address']}</b><br>
            <b>Available Units:</b> {int(row['lease_count'])}<br>
            <b>Avg SF/Unit:</b> {int(row['sf_per_unit']):,}<br>
            <b>Est. Annual Rent:</b> {rent_str}<br>
            <b>Safety Score:</b> {row['safety_score']:.2f}<br>
            <b>Accessibility:</b> {row['accessibility_score']:.2f}
        """

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=BeautifyIcon(
                icon=get_accessibility_icon(row['accessibility_score']),
                prefix='fa',
                icon_shape='marker',
                background_color=cmap(row['safety_score']),
                text_color='black',
                border_color='#333',
                border_width=1
            ),
            tooltip=row['full_address'],
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(cluster)

    cmap.add_to(m)

    # Add choropleth if requested
    if show_choropleth and sector and not zip_scores.empty:
        geo_path = DATA_DIR / 'ny_zips.json'
        if not geo_path.exists():
            geo_path = Path(__file__).parent / 'ny_zips.json'

        if geo_path.exists():
            sector_cols = {
                'Technology': 'tech_estabs_exp_score',
                'Finance': 'fin_estabs_exp_score',
                'Legal': 'law_estabs_exp_score'
            }

            if sector in sector_cols and sector_cols[sector] in zip_scores.columns:
                df_choro = zip_scores[['zip', sector_cols[sector]]].copy()
                df_choro = df_choro.rename(columns={sector_cols[sector]: 'score'})

                folium.Choropleth(
                    geo_data=str(geo_path),
                    name='Market Friendliness',
                    data=df_choro,
                    columns=['zip', 'score'],
                    key_on='feature.properties.ZCTA5CE10',
                    fill_color='YlGnBu',
                    fill_opacity=0.5,
                    line_opacity=0.2,
                    nan_fill_color='white',
                    legend_name=f'{sector} Market Friendliness'
                ).add_to(m)

                folium.LayerControl().add_to(m)

    return m


# Main App
def main():
    st.title("Manhattan Commercial Real Estate Explorer")

    st.markdown("""
    **Find your ideal office space in Manhattan**

    This tool helps commercial tenants find office space based on:
    - **Square footage needs** - Filter by your space requirements
    - **Safety** - Neighborhood safety scores based on local crime data
    - **Accessibility** - Transit accessibility (subway, bus, walking distance)
    - **Market fit** - See which areas are popular with your industry
    """)

    # Load data
    buildings = load_building_data()
    zip_scores = load_zip_scores()

    # Sidebar filters
    st.sidebar.header("Filter Properties")

    sf_min = int(buildings['sf_per_unit'].min())
    sf_max = int(buildings['sf_per_unit'].quantile(0.99))
    sf_range = st.sidebar.slider(
        "Square Footage per Unit",
        sf_min, sf_max, (sf_min, sf_max), step=500
    )

    safety_range = st.sidebar.slider(
        "Safety Score",
        0.0, 1.0, (0.0, 1.0), step=0.05
    )

    accessibility_range = st.sidebar.slider(
        "Accessibility Score",
        0.0, 1.0, (0.0, 1.0), step=0.05
    )

    sector_options = ['Technology', 'Finance', 'Legal']
    selected_sector = st.sidebar.selectbox("Your Industry", sector_options)

    show_heatmap = st.sidebar.checkbox("Show Market Friendliness Overlay")

    # Apply filters
    filtered = buildings[
        (buildings['sf_per_unit'].between(sf_range[0], sf_range[1])) &
        (buildings['safety_score'].between(safety_range[0], safety_range[1])) &
        (buildings['accessibility_score'].between(accessibility_range[0], accessibility_range[1]))
    ]

    st.sidebar.markdown(f"**Showing {len(filtered):,} of {len(buildings):,} properties**")

    # Create and display map
    m = create_map(
        filtered,
        show_choropleth=show_heatmap,
        sector=selected_sector,
        zip_scores=zip_scores
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st_folium(m, height=600, width=None, returned_objects=[])

    with col2:
        st.markdown("### Map Legend")
        st.markdown("""
        **Pin Colors** = Safety Score
        - Red = Higher risk
        - Yellow = Moderate
        - Green = Safer

        **Pin Icons** = Accessibility
        - Subway = Excellent transit
        - Bus = Good transit
        - Train = Moderate
        - Bicycle = Limited
        - Walking = Low access
        """)

        if len(filtered) > 0:
            st.markdown("### Quick Stats")
            st.metric("Avg SF/Unit", f"{filtered['sf_per_unit'].mean():,.0f}")
            st.metric("Avg Safety", f"{filtered['safety_score'].mean():.2f}")
            st.metric("Properties", len(filtered))


if __name__ == '__main__':
    main()
