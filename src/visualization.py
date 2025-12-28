"""
Visualization Module for Manhattan CRE Analysis

Generates interactive Folium maps for exploring lease data:
1. Building-level clusters by total SF
2. Leases colored by size category
3. Leases colored by industry sector
4. Multi-layer interactive map with toggles
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import branca
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manhattan center coordinates
MANHATTAN_CENTER = [40.78, -73.97]

# Color schemes
SIZE_COLORS = {
    'Small (<5K SF)': '#2ecc71',      # Green
    'Medium (5K-20K SF)': '#f39c12',  # Orange
    'Large (20K+ SF)': '#e74c3c',     # Red
    'Unknown': '#95a5a6'              # Gray
}

INDUSTRY_COLORS = {
    'Technology': '#3498db',
    'Finance': '#2ecc71',
    'Legal': '#9b59b6',
    'Healthcare': '#e74c3c',
    'Real Estate': '#f39c12',
    'Other': '#95a5a6'
}


def categorize_lease_size(sf: float) -> str:
    """Categorize lease by square footage."""
    if pd.isna(sf):
        return 'Unknown'
    elif sf < 5000:
        return 'Small (<5K SF)'
    elif sf < 20000:
        return 'Medium (5K-20K SF)'
    else:
        return 'Large (20K+ SF)'


def create_building_cluster_map(df: pd.DataFrame, output_path: Path) -> folium.Map:
    """
    Create a map showing building-level clusters.
    Circle size = total SF, color = SF category.
    """
    logger.info("Creating building cluster map")

    m = folium.Map(location=MANHATTAN_CENTER, zoom_start=13, tiles='CartoDB positron')

    # Create colormap for SF
    sf_col = 'total_leasedSF' if 'total_leasedSF' in df.columns else 'leasedSF'
    max_sf = df[sf_col].max()

    colormap = branca.colormap.StepColormap(
        colors=['#ffffcc', '#ffeda0', '#feb24c', '#f03b20', '#bd0026'],
        index=[0, 5000, 20000, 50000, 100000, max_sf],
        vmin=0,
        vmax=max_sf,
        caption='Total Leased SF'
    )

    cluster = MarkerCluster(options={
        'spiderfyOnMaxZoom': True,
        'showCoverageOnHover': True,
        'zoomToBoundsOnClick': True,
        'maxClusterRadius': 20
    }).add_to(m)

    for _, row in df.iterrows():
        sf = row[sf_col]
        popup_html = f"""
            <strong>{row['full_address']}</strong><br>
            <b>Total SF:</b> {int(sf):,}<br>
            <b>Leases:</b> {int(row.get('lease_count', 1))}<br>
            <b>Companies:</b> {row.get('company_list', 'N/A')[:100]}...
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=np.sqrt(sf) / 100,
            color=colormap(sf),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(cluster)

    colormap.add_to(m)
    m.save(str(output_path))
    logger.info(f"Saved building cluster map to {output_path}")
    return m


def create_lease_size_map(df: pd.DataFrame, output_path: Path) -> folium.Map:
    """Create a map with leases colored by size category."""
    logger.info("Creating lease size map")

    m = folium.Map(location=MANHATTAN_CENTER, zoom_start=12, tiles='CartoDB positron')

    sf_col = 'leasedSF' if 'leasedSF' in df.columns else 'total_leasedSF'
    df['size_category'] = df[sf_col].apply(categorize_lease_size)

    cluster = MarkerCluster(options={'maxClusterRadius': 120}).add_to(m)

    for _, row in df.iterrows():
        color = SIZE_COLORS.get(row['size_category'], '#95a5a6')
        popup_html = f"""
            <strong>{row.get('company_name', 'Building')}</strong><br>
            <b>SF:</b> {int(row[sf_col]):,}<br>
            <b>Category:</b> {row['size_category']}<br>
            <b>Address:</b> {row['full_address']}
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(cluster)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px;
                border: 2px solid grey; background: white;
                padding: 10px; z-index: 9999;">
        <b>Lease Size</b><br>
        <i style="background: #2ecc71; width: 12px; height: 12px;
           display: inline-block; margin-right: 5px;"></i>Small (&lt;5K SF)<br>
        <i style="background: #f39c12; width: 12px; height: 12px;
           display: inline-block; margin-right: 5px;"></i>Medium (5K-20K SF)<br>
        <i style="background: #e74c3c; width: 12px; height: 12px;
           display: inline-block; margin-right: 5px;"></i>Large (20K+ SF)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))
    logger.info(f"Saved lease size map to {output_path}")
    return m


def create_industry_map(df: pd.DataFrame, output_path: Path) -> folium.Map:
    """Create a map with leases colored by industry sector."""
    logger.info("Creating industry map")

    m = folium.Map(location=MANHATTAN_CENTER, zoom_start=12, tiles='CartoDB positron')

    industry_col = 'internal_industry' if 'internal_industry' in df.columns else 'sector_list'
    industries = df[industry_col].dropna().unique()

    # Create color mapping
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12',
              '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
    ind_colors = {ind: colors[i % len(colors)] for i, ind in enumerate(industries)}

    cluster = MarkerCluster(options={'maxClusterRadius': 100}).add_to(m)

    for _, row in df.iterrows():
        ind = row[industry_col] if pd.notna(row[industry_col]) else 'Unknown'
        # Handle semicolon-separated lists
        first_ind = ind.split(';')[0].strip() if ';' in str(ind) else ind
        color = ind_colors.get(first_ind, '#95a5a6')

        popup_html = f"""
            <strong>{row.get('company_name', row.get('company_list', 'N/A'))}</strong><br>
            <b>Industry:</b> {ind}<br>
            <b>Address:</b> {row['full_address']}
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(cluster)

    # Add legend
    legend_items = ''.join([
        f'<i style="background: {color}; width: 12px; height: 12px; '
        f'display: inline-block; margin-right: 5px;"></i>{ind[:20]}<br>'
        for ind, color in list(ind_colors.items())[:10]
    ])
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px;
                border: 2px solid grey; background: white;
                padding: 10px; z-index: 9999; max-height: 200px; overflow-y: auto;">
        <b>Industry</b><br>{legend_items}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(str(output_path))
    logger.info(f"Saved industry map to {output_path}")
    return m


def create_multi_layer_map(df: pd.DataFrame, output_path: Path) -> folium.Map:
    """
    Create an interactive map with multiple toggleable layers:
    - By Lease Size
    - By Industry
    - By Number of Leases per Building
    """
    logger.info("Creating multi-layer interactive map")

    m = folium.Map(location=MANHATTAN_CENTER, zoom_start=13, tiles='CartoDB positron')

    sf_col = 'leasedSF' if 'leasedSF' in df.columns else 'total_leasedSF'
    df['size_category'] = df[sf_col].apply(categorize_lease_size)

    # Layer 1: By Size
    fg_size = folium.FeatureGroup(name='By Lease Size', show=True)
    for _, row in df.iterrows():
        color = SIZE_COLORS.get(row['size_category'], '#95a5a6')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{row.get('company_name', 'Building')}: {int(row[sf_col]):,} SF"
        ).add_to(fg_size)
    fg_size.add_to(m)

    # Layer 2: By Industry
    industry_col = 'internal_industry' if 'internal_industry' in df.columns else 'sector_list'
    if industry_col in df.columns:
        fg_ind = folium.FeatureGroup(name='By Industry', show=False)
        industries = df[industry_col].dropna().unique()
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12',
                  '#1abc9c', '#e67e22', '#34495e']
        ind_colors = {ind: colors[i % len(colors)] for i, ind in enumerate(industries)}

        for _, row in df.iterrows():
            ind = row[industry_col] if pd.notna(row[industry_col]) else 'Unknown'
            first_ind = ind.split(';')[0].strip() if ';' in str(ind) else ind
            color = ind_colors.get(first_ind, '#95a5a6')
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row.get('company_name', 'Building')}: {first_ind}"
            ).add_to(fg_ind)
        fg_ind.add_to(m)

    # Layer 3: By Lease Count (if available)
    if 'lease_count' in df.columns:
        fg_count = folium.FeatureGroup(name='By Lease Count', show=False)
        max_count = df['lease_count'].max()
        count_cmap = branca.colormap.StepColormap(
            colors=['#ffffcc', '#ffeda0', '#feb24c', '#f03b20', '#bd0026'],
            index=[0, 1, 3, 5, 10, max_count],
            vmin=0,
            vmax=max_count,
            caption='Number of Leases'
        )

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=np.sqrt(row[sf_col]) / 150,
                color=count_cmap(row['lease_count']),
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['full_address']}: {int(row['lease_count'])} leases"
            ).add_to(fg_count)
        fg_count.add_to(m)
        count_cmap.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(output_path))
    logger.info(f"Saved multi-layer map to {output_path}")
    return m


def generate_all_maps(data_path: Path, output_dir: Path):
    """Generate all visualization maps from processed data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['latitude', 'longitude'])

    output_dir.mkdir(parents=True, exist_ok=True)

    create_building_cluster_map(df, output_dir / 'building_clusters.html')
    create_lease_size_map(df, output_dir / 'lease_sizes.html')
    create_industry_map(df, output_dir / 'industries.html')
    create_multi_layer_map(df, output_dir / 'interactive_explorer.html')

    logger.info("All maps generated successfully!")


if __name__ == '__main__':
    base_path = Path(__file__).parent.parent

    generate_all_maps(
        data_path=base_path / 'data' / 'processed' / 'manhattan_buildings.csv',
        output_dir=base_path / 'outputs'
    )
