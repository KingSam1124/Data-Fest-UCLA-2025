# main.py

import os
import sys
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import branca

# 1) Filenames in current folder
GEO_CSV = "leases_geocoded.csv"
BLD_CSV = "leases_by_building.csv"

# 2) Sanity check
for f in (GEO_CSV, BLD_CSV):
    if not os.path.isfile(f):
        print(f"ERROR: '{f}' not found in {os.getcwd()}")
        sys.exit(1)

# 3) Load
print("Loading data…")
leases_geo = pd.read_csv(GEO_CSV)
leases_bld = pd.read_csv(BLD_CSV)
print(f" → {len(leases_geo)} geocoded leases, {len(leases_bld)} buildings")

# 4) Drop missing coords
leases_geo = leases_geo.dropna(subset=['latitude','longitude'])
leases_bld  = leases_bld.dropna(subset=['latitude','longitude'])

# 5) Map #1: Individual leases
print("Building individual-leases map…")
required = {'company_name','leasedSF','internal_industry','full_address'}
if required.issubset(leases_geo.columns):
    # Size grouping
    conds = [
        leases_geo['leasedSF'] < 5000,
        leases_geo['leasedSF'].between(5000,20000, inclusive='left'),
        leases_geo['leasedSF'] >= 20000
    ]
    labels = ["Small (<5K SF)","Medium (5K–20K SF)","Large (20K+ SF)"]
    leases_geo['size_group'] = np.select(conds, labels, default="Unknown")
    cmap = {"Small (<5K SF)":"green","Medium (5K–20K SF)":"orange","Large (20K+ SF)":"red","Unknown":"gray"}

    m1 = folium.Map(location=[40.78,-73.97], zoom_start=12, tiles="CartoDB positron")
    cl1 = MarkerCluster(options={
        'spiderfyOnMaxZoom': True,
        'showCoverageOnHover': True,
        'zoomToBoundsOnClick': True,
        'maxClusterRadius': 120
    }).add_to(m1)

    for _, r in leases_geo.iterrows():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=5, color=cmap[r.size_group],
            fill=True, fill_opacity=0.8,
            popup=folium.Popup(
                f"<strong>{r.company_name}</strong><br>"
                f"{r.leasedSF} SF — {r.size_group}<br>"
                f"{r.internal_industry}<br>"
                f"{r.full_address}", max_width=300
            )
        ).add_to(cl1)

    # Legend
    legend = """
    <div style="
      position: fixed; bottom:50px; left:50px;
      border:2px solid grey; background:white; padding:8px;
      z-index:9999; opacity:0.8;">
      <b>Lease Size</b><br>
      <span style="background:green; width:12px; display:inline-block;"></span> Small<br>
      <span style="background:orange; width:12px; display:inline-block;"></span> Medium<br>
      <span style="background:red; width:12px; display:inline-block;"></span> Large<br>
    </div>
    """
    m1.get_root().html.add_child(folium.Element(legend))
    m1.save("manhattan_leases_map.html")
    print(" → manhattan_leases_map.html saved")
else:
    print("Skipping map1: missing required columns in leases_geocoded.csv")

# 6) Map #2: Building-level clusters
print("Building building-level clusters map…")
bins   = [0,5000,20000,50000,100000]
colors = ['#ffffcc','#ffeda0','#feb24c','#f03b20','#bd0026']
step   = branca.colormap.StepColormap(
    colors=colors,
    index=bins + [leases_bld['total_leasedSF'].max()],
    vmin=0,
    vmax=leases_bld['total_leasedSF'].max(),
    caption="Total Leased SF"
)

m2 = folium.Map(location=[40.78,-73.97], zoom_start=13, tiles="CartoDB positron")
cl2 = MarkerCluster(options={
    'spiderfyOnMaxZoom': True,
    'showCoverageOnHover': True,
    'zoomToBoundsOnClick': True,
    'maxClusterRadius': 20
}).add_to(m2)

for _, r in leases_bld.iterrows():
    folium.CircleMarker(
        [r.latitude, r.longitude],
        radius=np.sqrt(r.total_leasedSF)/100,
        color=step(r.total_leasedSF),
        fill=True, fill_opacity=0.8,
        popup=folium.Popup(
            f"<strong>{r.full_address}</strong><br>"
            f"{r.company_list}<br>"
            f"{r.sector_list}<br>"
            f"{r.total_leasedSF} SF across {r.lease_count} leases",
            max_width=300
        )
    ).add_to(cl2)

step.add_to(m2)
m2.save("manhattan_building_clusters.html")
print(" → manhattan_building_clusters.html saved")

# 7) Map #3: Jittered leases
print("Building jittered-leases map…")
dfj = leases_geo.copy() if 'leasedSF' in leases_geo.columns else leases_bld.copy()
if 'total_leasedSF' in dfj.columns and 'leasedSF' not in dfj.columns:
    dfj['leasedSF'] = dfj['total_leasedSF']

dfj['latitude']  += np.random.normal(scale=0.0001, size=len(dfj))
dfj['longitude'] += np.random.normal(scale=0.0001, size=len(dfj))

c_bins = np.linspace(dfj.leasedSF.min(), dfj.leasedSF.max(), 6)
cmap_j = branca.colormap.StepColormap(
    colors=colors,
    index=c_bins.tolist(),
    vmin=c_bins.min(),
    vmax=c_bins.max(),
    caption="Lease Size (SF)"
)

m3 = folium.Map(location=[40.78,-73.97], zoom_start=13, tiles="CartoDB positron")
cl3 = MarkerCluster(options={
    'spiderfyOnMaxZoom': True,
    'showCoverageOnHover': True,
    'zoomToBoundsOnClick': True,
    'maxClusterRadius': 80
}).add_to(m3)

for _, r in dfj.iterrows():
    folium.CircleMarker(
        [r.latitude, r.longitude],
        radius=4,
        color=cmap_j(r.leasedSF),
        fill=True, fill_opacity=0.7,
        popup=folium.Popup(
            f"<strong>{getattr(r,'company_name',r.company_list)}</strong><br>"
            f"{r.leasedSF} SF — {getattr(r,'internal_industry',r.sector_list)}<br>"
            f"{r.full_address}", max_width=300
        )
    ).add_to(cl3)

cmap_j.add_to(m3)
m3.save("manhattan_jittered_leases_map.html")
print(" → manhattan_jittered_leases_map.html saved")

print("All maps built—open the HTML files in your browser.")
