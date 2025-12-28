"""
DataFest 2025: Commercial Real Estate Analysis
Run this script to generate all insights and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Leases.csv'
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR.parent / '2025 Data Files' / 'Leases.csv'

OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("DATAFEST 2025: COMMERCIAL REAL ESTATE ANALYSIS")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} lease records")
print(f"Date range: {df['year'].min()} - {df['year'].max()}")
print(f"Markets: {df['market'].nunique()}")

# Create time columns
quarter_map = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['quarter'].map(quarter_map) + '-01')

def categorize_period(year):
    if year < 2020:
        return 'Pre-Pandemic'
    elif year <= 2021:
        return 'Pandemic'
    else:
        return 'Recovery'

df['period'] = df['year'].apply(categorize_period)

# ============================================================
# 1. PANDEMIC IMPACT ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("1. PANDEMIC IMPACT ANALYSIS")
print("=" * 70)

quarterly = df.groupby('date').agg({
    'leasedSF': ['sum', 'count', 'mean', 'median']
}).reset_index()
quarterly.columns = ['date', 'total_sf', 'num_leases', 'avg_sf', 'median_sf']

# Key metrics by period
pre = df[df['year'].isin([2018, 2019])]
pandemic = df[df['year'].isin([2020, 2021])]
recovery = df[df['year'] >= 2022]

pre_annual = pre['leasedSF'].sum() / 2
pandemic_annual = pandemic['leasedSF'].sum() / 2
recovery_annual = recovery['leasedSF'].sum() / recovery['year'].nunique()

pandemic_change = (pandemic_annual / pre_annual - 1) * 100
recovery_rate = (recovery_annual / pre_annual) * 100

print(f"\nPre-Pandemic Annual Average (2018-19): {pre_annual/1e6:.1f}M SF")
print(f"Pandemic Annual Average (2020-21): {pandemic_annual/1e6:.1f}M SF ({pandemic_change:+.1f}%)")
print(f"Recovery Annual Average (2022+): {recovery_annual/1e6:.1f}M SF ({recovery_rate:.0f}% of baseline)")

# Create pandemic impact visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
ax1.plot(quarterly['date'], quarterly['total_sf'] / 1e6, 'b-', linewidth=2)
ax1.fill_between(quarterly['date'], 0, quarterly['total_sf'] / 1e6, alpha=0.3)
ax1.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', alpha=0.7, label='COVID-19')
ax1.set_ylabel('Total SF Leased (Millions)')
ax1.set_title('Quarterly Leasing Volume')
ax1.legend()

ax2 = axes[0, 1]
ax2.plot(quarterly['date'], quarterly['num_leases'], 'g-', linewidth=2)
ax2.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', alpha=0.7)
ax2.set_ylabel('Number of Leases')
ax2.set_title('Number of Lease Transactions')

ax3 = axes[1, 0]
ax3.plot(quarterly['date'], quarterly['avg_sf'], 'orange', linewidth=2, label='Mean')
ax3.plot(quarterly['date'], quarterly['median_sf'], 'purple', linewidth=2, label='Median')
ax3.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', alpha=0.7)
ax3.set_ylabel('Square Footage')
ax3.set_title('Average vs Median Lease Size')
ax3.legend()

ax4 = axes[1, 1]
period_totals = df.groupby('period')['leasedSF'].sum() / df.groupby('period')['year'].nunique()
colors = ['#3498db', '#e74c3c', '#2ecc71']
period_totals.plot(kind='bar', ax=ax4, color=colors)
ax4.set_ylabel('Annual SF (Millions)')
ax4.set_title('Average Annual Leasing by Period')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
for i, v in enumerate(period_totals / 1e6):
    ax4.text(i, v + 0.5, f'{v:.1f}M', ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pandemic_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pandemic_impact.png")

# ============================================================
# 2. INDUSTRY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("2. INDUSTRY ANALYSIS")
print("=" * 70)

# Top industries
top_industries = df.groupby('internal_industry')['leasedSF'].sum().nlargest(10)
print("\nTop 10 Industries by Total Leased SF:")
for i, (ind, sf) in enumerate(top_industries.items(), 1):
    print(f"  {i}. {ind}: {sf/1e6:.1f}M SF")

# Industry changes pre vs post pandemic
industry_period = df.groupby(['internal_industry', 'period'])['leasedSF'].sum().unstack(fill_value=0)
industry_period['pre_annual'] = industry_period.get('Pre-Pandemic', 0) / 2
industry_period['post_annual'] = industry_period.get('Recovery', 0) / recovery['year'].nunique()
industry_period['change_pct'] = ((industry_period['post_annual'] / industry_period['pre_annual']) - 1) * 100
industry_period = industry_period.replace([np.inf, -np.inf], np.nan).dropna()
industry_period = industry_period.sort_values('change_pct', ascending=False)

print("\nIndustry Recovery Rates (Top 10 by volume):")
for ind in top_industries.index[:10]:
    if ind in industry_period.index:
        change = industry_period.loc[ind, 'change_pct']
        print(f"  {ind}: {change:+.1f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
(top_industries / 1e6).plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_xlabel('Total SF (Millions)')
ax1.set_title('Top 10 Industries by Total Leasing (2018-Present)')

ax2 = axes[1]
top_10_changes = industry_period.loc[top_industries.index[:10], 'change_pct'].sort_values()
colors = ['green' if x > 0 else 'red' for x in top_10_changes]
top_10_changes.plot(kind='barh', ax=ax2, color=colors)
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlabel('% Change (Recovery vs Pre-Pandemic Annual)')
ax2.set_title('Industry Recovery: Winners & Losers')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'industry_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: industry_analysis.png")

# ============================================================
# 3. FLIGHT TO QUALITY (CLASS A vs O)
# ============================================================
print("\n" + "=" * 70)
print("3. FLIGHT TO QUALITY ANALYSIS")
print("=" * 70)

class_year = df.groupby(['year', 'internal_class'])['leasedSF'].sum().unstack(fill_value=0)
class_year['total'] = class_year.sum(axis=1)
class_year['A_share'] = class_year.get('A', 0) / class_year['total'] * 100

pre_a_share = class_year.loc[[2018, 2019], 'A_share'].mean()
post_a_share = class_year.loc[class_year.index >= 2022, 'A_share'].mean()

print(f"\nClass A Share of Total Leasing:")
print(f"  Pre-Pandemic (2018-19): {pre_a_share:.1f}%")
print(f"  Post-Pandemic (2022+): {post_a_share:.1f}%")
print(f"  Change: {post_a_share - pre_a_share:+.1f} percentage points")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(class_year.index, class_year['A_share'], 'b-o', linewidth=2, markersize=8)
ax1.axhline(class_year['A_share'].mean(), color='gray', linestyle='--', alpha=0.7)
ax1.axvline(2020, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Year')
ax1.set_ylabel('Class A Share (%)')
ax1.set_title('Class A Share of Total Leased SF')
ax1.set_ylim(0, 100)

ax2 = axes[1]
class_pivot = class_year[['A', 'O']].copy() if 'O' in class_year.columns else class_year[['A']].copy()
(class_pivot / 1e6).plot(kind='bar', ax=ax2, color=['#2ecc71', '#3498db'])
ax2.set_ylabel('Total SF (Millions)')
ax2.set_xlabel('Year')
ax2.set_title('Leasing Volume by Building Class')
ax2.legend(title='Class')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'flight_to_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: flight_to_quality.png")

# ============================================================
# 4. CBD VS SUBURBAN
# ============================================================
print("\n" + "=" * 70)
print("4. CBD VS SUBURBAN ANALYSIS")
print("=" * 70)

loc_year = df.groupby(['year', 'CBD_suburban'])['leasedSF'].sum().unstack(fill_value=0)
loc_year['total'] = loc_year.sum(axis=1)

for loc in ['CBD', 'Suburban']:
    if loc in loc_year.columns:
        loc_year[f'{loc}_share'] = loc_year[loc] / loc_year['total'] * 100

if 'Suburban_share' in loc_year.columns:
    pre_suburban = loc_year.loc[[2018, 2019], 'Suburban_share'].mean()
    post_suburban = loc_year.loc[loc_year.index >= 2022, 'Suburban_share'].mean()

    print(f"\nSuburban Share of Total Leasing:")
    print(f"  Pre-Pandemic: {pre_suburban:.1f}%")
    print(f"  Post-Pandemic: {post_suburban:.1f}%")
    print(f"  Change: {post_suburban - pre_suburban:+.1f} percentage points")

# Visualization
fig, ax = plt.subplots(figsize=(12, 5))
loc_cols = [c for c in ['CBD', 'Suburban'] if c in loc_year.columns]
(loc_year[loc_cols] / 1e6).plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
ax.set_ylabel('Total SF (Millions)')
ax.set_xlabel('Year')
ax.set_title('CBD vs Suburban Leasing Volume')
ax.legend(title='Location')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cbd_vs_suburban.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: cbd_vs_suburban.png")

# ============================================================
# 5. LEASE SIZE FRAGMENTATION
# ============================================================
print("\n" + "=" * 70)
print("5. LEASE SIZE FRAGMENTATION")
print("=" * 70)

def size_cat(sf):
    if sf < 5000: return 'Small (<5K)'
    elif sf < 20000: return 'Medium (5-20K)'
    elif sf < 50000: return 'Large (20-50K)'
    else: return 'Enterprise (50K+)'

df['size_cat'] = df['leasedSF'].apply(size_cat)

size_year = df.groupby(['year', 'size_cat']).size().unstack(fill_value=0)
size_year_pct = size_year.div(size_year.sum(axis=1), axis=0) * 100

pre_small = size_year_pct.loc[[2018, 2019], 'Small (<5K)'].mean()
post_small = size_year_pct.loc[size_year_pct.index >= 2022, 'Small (<5K)'].mean()

print(f"\nSmall Lease Share (<5K SF):")
print(f"  Pre-Pandemic: {pre_small:.1f}%")
print(f"  Post-Pandemic: {post_small:.1f}%")
print(f"  Change: {post_small - pre_small:+.1f} percentage points")

# Median lease size trend
median_by_year = df.groupby('year')['leasedSF'].median()
print(f"\nMedian Lease Size:")
print(f"  2018-2019 Avg: {median_by_year.loc[[2018, 2019]].mean():,.0f} SF")
print(f"  2022+ Avg: {median_by_year.loc[median_by_year.index >= 2022].mean():,.0f} SF")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
size_year_pct.plot(kind='area', stacked=True, ax=ax1, colormap='RdYlGn_r', alpha=0.8)
ax1.axvline(2020, color='black', linestyle='--', linewidth=2)
ax1.set_ylabel('Share of Leases (%)')
ax1.set_xlabel('Year')
ax1.set_title('Lease Size Distribution Over Time')
ax1.legend(title='Size', bbox_to_anchor=(1.02, 1), loc='upper left')

ax2 = axes[1]
ax2.plot(median_by_year.index, median_by_year.values, 'b-o', linewidth=2, markersize=8)
ax2.axvline(2020, color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel('Median Lease Size (SF)')
ax2.set_xlabel('Year')
ax2.set_title('Median Lease Size Trend')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'lease_fragmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: lease_fragmentation.png")

# ============================================================
# 6. MARKET RECOVERY COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("6. MARKET RECOVERY COMPARISON")
print("=" * 70)

top_markets = df.groupby('market')['leasedSF'].sum().nlargest(12).index

market_period = df[df['market'].isin(top_markets)].groupby(['market', 'period'])['leasedSF'].sum().unstack(fill_value=0)
market_period['pre_annual'] = market_period.get('Pre-Pandemic', 0) / 2
market_period['post_annual'] = market_period.get('Recovery', 0) / recovery['year'].nunique()
market_period['recovery_rate'] = (market_period['post_annual'] / market_period['pre_annual']) * 100
market_period = market_period.sort_values('recovery_rate', ascending=False)

print("\nTop 12 Markets Recovery Rate:")
for market in market_period.index:
    rate = market_period.loc[market, 'recovery_rate']
    print(f"  {market}: {rate:.0f}%")

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['green' if x >= 100 else 'red' for x in market_period['recovery_rate']]
market_period['recovery_rate'].plot(kind='barh', ax=ax, color=colors)
ax.axvline(100, color='black', linestyle='--', linewidth=2, label='100% = Pre-Pandemic Level')
ax.set_xlabel('Recovery Rate (%)')
ax.set_title('Market Recovery: 2022+ vs Pre-Pandemic Annual Average')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'market_recovery.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: market_recovery.png")

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

fig = plt.figure(figsize=(16, 10))
fig.suptitle('DataFest 2025: Post-Pandemic Office Market Analysis', fontsize=18, fontweight='bold', y=0.98)

# Metrics
metrics_text = f"""
KEY FINDINGS:

1. PANDEMIC IMPACT: {pandemic_change:.0f}% drop in annual leasing (2020-21 vs 2018-19)

2. RECOVERY STATUS: {recovery_rate:.0f}% of pre-pandemic levels reached

3. FLIGHT TO QUALITY: Class A share {"increased" if post_a_share > pre_a_share else "decreased"} by {abs(post_a_share - pre_a_share):.1f}pp

4. FRAGMENTATION: Small leases (<5K SF) {"increased" if post_small > pre_small else "decreased"} from {pre_small:.0f}% to {post_small:.0f}%

5. TOP INDUSTRY: {top_industries.index[0]} ({top_industries.iloc[0]/1e6:.0f}M SF total)

RECOMMENDATIONS FOR SAVILLS CLIENTS:

- TENANTS: Leverage tenant-favorable market for better terms
- LANDLORDS: Invest in amenities to attract quality tenants
- ALL: Focus on Class A properties in recovering markets
"""

ax_text = fig.add_subplot(1, 2, 1)
ax_text.text(0.05, 0.95, metrics_text, transform=ax_text.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_text.axis('off')

ax_chart = fig.add_subplot(1, 2, 2)
quarterly.plot(x='date', y='total_sf', ax=ax_chart, legend=False)
ax_chart.axvline(pd.Timestamp('2020-03-01'), color='red', linestyle='--', label='COVID-19')
ax_chart.set_ylabel('Total SF')
ax_chart.set_title('Quarterly Leasing Volume')
ax_chart.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'executive_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: executive_summary.png")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print(f"All visualizations saved to: {OUTPUT_DIR}")
print("=" * 70)
