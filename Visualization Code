# fixingsqft.py
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
import re

# -----------------------------------------------------------------------------#
# Streamlit page setup                                                         #
# -----------------------------------------------------------------------------#
st.set_page_config(page_title="Manhattan Leases Explorer", layout="wide")

# -----------------------------------------------------------------------------#
# Helpers                                                                       #
# -----------------------------------------------------------------------------#
def _first_match(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return the first column whose name matches any regex in *patterns*."""
    for pat in patterns:
        for col in df.columns:
            if re.search(pat, col, flags=re.I):
                return col
    return None


@st.cache_data
def load_data() -> pd.DataFrame:
    """Read the CSV and normalise safety, accessibility and square-footage columns."""
    df = (
        pd.read_csv("manhattan_geo_access.csv")
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )

    # ── 1  Safety (0–1, higher = safer) ───────────────────────────────────
    crime_col = _first_match(df, [r"crime", r"risk", r"safety"])
    if crime_col is None:
        st.warning("No crime/safety column found – assigning neutral 0.5 scores.")
        df["safety_score"] = 0.5
    else:
        df[crime_col] = pd.to_numeric(df[crime_col], errors="coerce")
        df[crime_col].fillna(df[crime_col].mean(skipna=True), inplace=True)
        if df[crime_col].max() > 1 or df[crime_col].min() < 0:
            df[crime_col] = (df[crime_col] - df[crime_col].min()) / (
                df[crime_col].max() - df[crime_col].min()
            )
        df["safety_score"] = 1 - df[crime_col]

    # ── 2  Accessibility (0–1, higher = better) ───────────────────────────
    acc_col = _first_match(
        df,
        [
            r"(access|walk|transit).*score",
            r"accessibility",
            r"weighted_routes",
        ],
    )
    if acc_col is None:
        st.warning("No accessibility column found – assigning neutral 0.5 scores.")
        df["accessibility_score"] = 0.5
    else:
        df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
        df[acc_col].fillna(df[acc_col].mean(skipna=True), inplace=True)
        if df[acc_col].max() > 1 or df[acc_col].min() < 0:
            df[acc_col] = (df[acc_col] - df[acc_col].min()) / (
                df[acc_col].max() - df[acc_col].min()
            )
        df.rename(columns={acc_col: "accessibility_score"}, inplace=True)

    # ── 3  Square footage (SF) ────────────────────────────────────────────
    sf_col = _first_match(
        df,
        [
            r"(total_)?leased.*sf",   # total_leasedSF, leasedSF …
            r"\bsf$",                # …_sf
            r"sq.?ft",               # sqft, sq_ft, squarefoot
            r"sf",                   # last-ditch fallback
        ],
    )
    if sf_col is None:
        raise ValueError(
            "Could not detect a square-footage column. "
            "Rename the CSV column or add another regex pattern."
        )

    df[sf_col] = pd.to_numeric(df[sf_col], errors="coerce")
    df[sf_col].fillna(df[sf_col].mean(skipna=True), inplace=True)
    df.rename(columns={sf_col: "leasedSF"}, inplace=True)

    return df


# -----------------------------------------------------------------------------#
# Load data & build UI                                                         #
# -----------------------------------------------------------------------------#
df = load_data()

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

# 1) Square-footage slider
sf_min = int(df["leasedSF"].min())
sf_max_99 = int(df["leasedSF"].quantile(0.99))
sf_low, sf_high = st.sidebar.slider(
    "Square Footage Range (SF)",
    min_value=sf_min,
    max_value=sf_max_99,
    value=(sf_min, sf_max_99),
    step=500,
)

# 2) Safety slider
safe_low, safe_high = st.sidebar.slider(
    "Safety Score",
    0.0,
    1.0,
    (0.0, 1.0),
    0.01,
)

# 3) Accessibility slider
acc_low, acc_high = st.sidebar.slider(
    "Accessibility Score",
    0.0,
    1.0,
    (0.0, 1.0),
    0.01,
)

# Apply filters
df_filt = df[
    (df["leasedSF"].between(sf_low, sf_high))
    & (df["safety_score"].between(safe_low, safe_high))
    & (df["accessibility_score"].between(acc_low, acc_high))
]

st.sidebar.markdown(f"**Showing {len(df_filt):,} of {len(df):,} leases**")

# -----------------------------------------------------------------------------#
# Map helpers                                                                  #
# -----------------------------------------------------------------------------#
cmap = LinearColormap(
    colors=["red", "orange", "yellow", "lightgreen", "green"],
    vmin=0,
    vmax=1,
)

ACC_ICONS = {
    4: ("bus",     "#2b83ba"),  # 0.81–1.00
    3: ("train",   "#abdda4"),  # 0.61–0.80
    2: ("subway",  "#ffffbf"),  # 0.41–0.60
    1: ("bicycle", "#fdae61"),  # 0.21–0.40
    0: ("walk",    "#d7191c"),  # 0.00–0.20
}


def icon_for_acc(score: float) -> folium.Icon:
    bucket = min(4, int(score * 5))  # 0–0.999 → 0–4
    name, colour = ACC_ICONS[bucket]
    return folium.Icon(icon=name, prefix="fa", color="white", icon_color=colour)


# -----------------------------------------------------------------------------#
# Build Folium map                                                             #
# -----------------------------------------------------------------------------#
m = folium.Map(
    location=[40.75, -73.97],
    zoom_start=12,
    tiles="cartodbpositron",
)
cluster = MarkerCluster().add_to(m)

for _, row in df_filt.iterrows():
    folium.Marker(
        location=[row.latitude, row.longitude],
        icon=icon_for_acc(row["accessibility_score"]),
        tooltip=row.get("full_address", ""),
        popup=folium.Popup(
            html=(
                f"<b>{row.get('full_address', 'No address')}</b><br>"
                f"SF&nbsp;:&nbsp;{int(row['leasedSF']):,}<br>"
                f"Safety&nbsp;:&nbsp;{row['safety_score']:.2f}<br>"
                f"Access&nbsp;:&nbsp;{row['accessibility_score']:.2f}"
            ),
            max_width=300,
        ),
    ).add_to(cluster)

cmap.caption = "Safety Score (red = riskier → green = safer)"
cmap.add_to(m)

# -----------------------------------------------------------------------------#
# Streamlit layout                                                             #
# -----------------------------------------------------------------------------#
st.title("Manhattan Leases Explorer")
st.write(
    """
    Use the sidebar to filter by square footage, neighbourhood safety, and public-
    transport accessibility. Markers are coloured by accessibility; the colour bar
    shows relative safety.
    """
)

st_folium(m, height=600, width=900, returned_objects=[])
