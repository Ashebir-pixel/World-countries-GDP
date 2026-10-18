"""
World Countries GDP Intelligence Dashboard (1960 - 2020)
Interactive Streamlit Application featuring Macro Trends, Continental Power Shifts,
Hegemony Race, Lorenz Inequality, Country Deep-Dive, and Live SQL Query Console.
"""

import os
import sqlite3
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="World Countries GDP Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM STYLING (MODERN DARK THEME) ---
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2433 0%, #161a25 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    /* Header & Badge Styling */
    .dashboard-header {
        margin-bottom: 25px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 8px;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }

    /* Code block and container borders */
    .stCodeBlock {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. MULTI-TIER ROBUST DATA LOADER ---
@st.cache_data(show_spinner="Loading World Countries GDP dataset...")
def load_gdp_data():
    """Load dataset from SQLite database, CSV, or ZIP archive fallback."""
    db_path = "world_gdp.db"
    csv_path = "gdp_1960_2020.csv"
    zip_path = "archive.zip"
    source_name = "Unknown"
    df = None

    # Tier 1: SQLite Database
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM world_gdp", conn)
            conn.close()
            source_name = f"SQLite Database ({db_path})"
        except Exception:
            df = None

    # Tier 2: Precompiled CSV
    if df is None and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            source_name = f"Precompiled CSV ({csv_path})"
        except Exception:
            df = None

    # Tier 3: Extract from ZIP archive
    if df is None and os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                df = pd.read_csv(z.open("gdp_1960_2020.csv"))
            source_name = f"Compiled from Archive ({zip_path})"
        except Exception:
            df = None

    if df is None:
        st.error("❌ Dataset not found! Please ensure `world_gdp.db`, `gdp_1960_2020.csv`, or `archive.zip` is available.")
        st.stop()

    # Normalize columns
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Text cleaning
    for col in ["country", "state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "state" in df.columns and "continent" not in df.columns:
        df["continent"] = df["state"]

    # Numeric typing
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype(int)
    df["gdp"] = pd.to_numeric(df["gdp"], errors="coerce")
    df["gdp_percent"] = pd.to_numeric(df["gdp_percent"], errors="coerce")
    
    # Calculate GDP in Trillions and Billions for easy display
    df["gdp_trillions"] = df["gdp"] / 1e12
    df["gdp_billions"] = df["gdp"] / 1e9

    return df, source_name


# Helper for Gini coefficient calculation
def calculate_gini(array: np.ndarray) -> float:
    """Calculate the Gini coefficient of an array of GDP values."""
    array = np.array(array, dtype=np.float64)
    array = array[array > 0]
    if len(array) == 0:
        return 0.0
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


# Load Data
df_raw, data_source = load_gdp_data()

# --- 4. SIDEBAR CONTROLS & FILTERS ---
st.sidebar.markdown("## 🌍 Filter Global Dataset")
st.sidebar.caption(f"Connected to: **{data_source}**")

# Preset Groupings
G7_COUNTRIES = ["the United States", "Japan", "Germany", "United Kingdom", "France", "Italy", "Canada"]
BRICS_COUNTRIES = ["Brazil", "Russia", "India", "China", "South Africa"]
TOP10_MODERN = ["the United States", "China", "Japan", "Germany", "United Kingdom", "India", "France", "Italy", "Canada", "South Korea"]

# Country Selection Mode
country_filter_mode = st.sidebar.radio(
    "Country Scope Preset",
    options=["All Countries", "Top 10 Modern Economies", "G7 Nations", "BRICS Nations", "Custom Selection"],
    index=0
)

all_countries_list = sorted(df_raw["country"].unique().tolist())
if country_filter_mode == "All Countries":
    selected_countries = all_countries_list
elif country_filter_mode == "Top 10 Modern Economies":
    selected_countries = [c for c in TOP10_MODERN if c in all_countries_list]
elif country_filter_mode == "G7 Nations":
    selected_countries = [c for c in G7_COUNTRIES if c in all_countries_list]
elif country_filter_mode == "BRICS Nations":
    selected_countries = [c for c in BRICS_COUNTRIES if c in all_countries_list]
else:
    selected_countries = st.sidebar.multiselect(
        "Select Specific Countries",
        options=all_countries_list,
        default=TOP10_MODERN[:5]
    )

# Year Range Slider
min_year = int(df_raw["year"].min())
max_year = int(df_raw["year"].max())
selected_year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

# Continent Filter
all_continents = sorted(df_raw["continent"].unique().tolist())
select_all_continents = st.sidebar.checkbox("Select All Continents", value=True)
if select_all_continents:
    selected_continents = all_continents
else:
    selected_continents = st.sidebar.multiselect(
        "Filter by Continent",
        options=all_continents,
        default=all_continents
    )

# Max Rank Filter
max_rank_val = int(df_raw["rank"].max())
selected_max_rank = st.sidebar.slider(
    "Max Rank Ceiling",
    min_value=1,
    max_value=max_rank_val,
    value=max_rank_val,
    help="Filter to include only countries ranked up to this ceiling (e.g. Top 20, Top 50)."
)

# Apply Filters
filtered_df = df_raw[
    (df_raw["country"].isin(selected_countries)) &
    (df_raw["year"] >= selected_year_range[0]) &
    (df_raw["year"] <= selected_year_range[1]) &
    (df_raw["continent"].isin(selected_continents)) &
    (df_raw["rank"] <= selected_max_rank)
].copy()

# Sidebar Stats
st.sidebar.markdown("---")
filtered_pct = (len(filtered_df) / len(df_raw)) * 100
st.sidebar.info(f"📊 Filtered Records: **{len(filtered_df):,}** / {len(df_raw):,} ({filtered_pct:.1f}%)")

if filtered_df.empty:
    st.warning("⚠️ No data matches your filter combination. Please adjust the sidebar filters.")
    st.stop()


# --- 5. HEADER & TOP KPI ROW ---
st.markdown("""
<div class="dashboard-header">
    <span class="badge">GLOBAL MACROECONOMIC INTELLIGENCE</span>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800;">🌍 World Countries GDP Intelligence Dashboard</h1>
    <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 6px;">
        Six Decades of Economic Evolution, Continental Power Shifts, Hegemony Races, and Inequality (1960–2020)
    </p>
</div>
""", unsafe_allow_html=True)

# Metrics calculation for the latest year in filtered selection
latest_year_in_filter = filtered_df["year"].max()
latest_data = filtered_df[filtered_df["year"] == latest_year_in_filter]

total_latest_gdp = latest_data["gdp"].sum()
countries_count = latest_data["country"].nunique()

# Global YoY growth for latest available year
all_years_sum = df_raw.groupby("year")["gdp"].sum()
if latest_year_in_filter - 1 in all_years_sum.index and latest_year_in_filter in all_years_sum.index:
    prev_gdp = all_years_sum.loc[latest_year_in_filter - 1]
    curr_gdp = all_years_sum.loc[latest_year_in_filter]
    yoy_growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
    yoy_str = f"{yoy_growth:+.2f}% YoY"
else:
    yoy_str = "N/A"

top_economy = latest_data.sort_values("rank").iloc[0] if not latest_data.empty else None
top_economy_name = top_economy["country"] if top_economy is not None else "N/A"
top_economy_gdp = f"${top_economy['gdp']/1e12:.2f}T" if top_economy is not None else "N/A"

# Inter-country Gini for latest year
current_gini = calculate_gini(latest_data["gdp"].values) if not latest_data.empty else 0.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Selected Output (Latest Year)", f"${total_latest_gdp/1e12:.2f}T", yoy_str, help=f"Total output of filtered selection in {latest_year_in_filter}")
m2.metric("Tracked Nations in Scope", f"{countries_count} Countries", f"Year {latest_year_in_filter}")
m3.metric("#1 Economy in Scope", top_economy_name, top_economy_gdp)
m4.metric("Temporal Span", f"{selected_year_range[0]} – {selected_year_range[1]}", f"{selected_year_range[1] - selected_year_range[0] + 1} Years")
m5.metric("Inter-Country Gini", f"{current_gini:.3f}", help="0 = Perfect equality across countries, 1 = Complete concentration")

st.markdown("<br>", unsafe_allow_html=True)


# --- 6. INTERACTIVE TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Macro Trajectory & Map",
    "🏆 Top Economies & Hegemony",
    "🗺️ Continental Dynamics",
    "⚖️ Inequality & Lorenz",
    "🔬 Country Deep-Dive & Comparison",
    "🔍 Data Explorer & SQL Console"
])


# ==========================================
# TAB 1: MACRO TRAJECTORY & CHOROPLETH MAP
# ==========================================
with tab1:
    st.subheader("Global GDP Growth Trajectory & Historic Macroeconomic Shocks")
    
    yearly_totals = df_raw.groupby("year")["gdp"].sum().reset_index()
    yearly_totals["gdp_trillions"] = yearly_totals["gdp"] / 1e12
    yearly_totals["pct_change"] = yearly_totals["gdp"].pct_change() * 100

    col_line, col_bar = st.columns([3, 2])

    with col_line:
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=yearly_totals["year"],
            y=yearly_totals["gdp_trillions"],
            mode="lines+markers",
            name="World GDP ($ Trillions)",
            line=dict(color="#3b82f6", width=3),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.15)",
            hovertemplate="Year: %{x}<br>World GDP: $%{y:.2f} Trillion<extra></extra>"
        ))

        # Add Crisis Event Annotations
        shocks = [
            (1973, 4.6, "1973 Oil Crisis", "red"),
            (1982, 11.4, "1982 Debt Crisis", "orange"),
            (1997, 31.4, "1997 Asian Crisis", "yellow"),
            (2008, 64.1, "2008 Great Recession", "red"),
            (2020, 77.5, "2020 COVID-19 (-10.1%)", "purple")
        ]
        for yr, val, label, col in shocks:
            fig_traj.add_annotation(
                x=yr, y=val, text=label, showarrow=True,
                arrowhead=2, arrowcolor="#ef4444", ax=0, ay=-35,
                bgcolor="#1e2433", bordercolor="#ef4444", borderwidth=1,
                font=dict(size=9, color="#f8fafc")
            )

        fig_traj.update_layout(
            title="World GDP Expansion (1960 – 2020) in Trillions USD",
            xaxis_title="Year",
            yaxis_title="GDP ($ Trillion USD)",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=40),
            height=400
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_bar:
        # Annual YoY Growth Bars
        bar_colors = ["#22c55e" if x >= 0 else "#ef4444" for x in yearly_totals["pct_change"].fillna(0)]
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Bar(
            x=yearly_totals["year"],
            y=yearly_totals["pct_change"],
            marker_color=bar_colors,
            name="YoY Growth %",
            hovertemplate="Year: %{x}<br>YoY Growth: %{y:+.2f}%<extra></extra>"
        ))
        fig_growth.update_layout(
            title="Annual Global GDP Growth Rate (%)",
            xaxis_title="Year",
            yaxis_title="Growth Rate (%)",
            template="plotly_dark",
            margin=dict(l=40, r=20, t=50, b=40),
            height=400
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Global GDP Choropleth Map")

    map_year = st.slider("Select Map Snapshot Year", min_value=min_year, max_value=max_year, value=max_year, step=1)
    map_metric = st.radio("Map Metric", options=["Nominal GDP ($ USD)", "Share of World GDP (%)"], horizontal=True)

    map_df = df_raw[df_raw["year"] == map_year].copy()
    color_col = "gdp" if "Nominal" in map_metric else "gdp_percent"
    color_title = "GDP ($ USD)" if "Nominal" in map_metric else "World Share"

    fig_map = px.choropleth(
        map_df,
        locations="country",
        locationmode="country names",
        color=color_col,
        hover_name="country",
        hover_data={"gdp": ":$,.0f", "rank": True, "continent": True, "gdp_percent": ":.2%"},
        color_continuous_scale="Viridis",
        title=f"Global Economic Landscape in {map_year}"
    )
    fig_map.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular", bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ==========================================
# TAB 2: TOP ECONOMIES & HEGEMONY
# ==========================================
with tab2:
    st.subheader("Top Global Economies & Ranking Transitions")

    c1, c2 = st.columns([1, 1])
    with c1:
        snapshot_year = st.slider("Ranking Snapshot Year", min_value=min_year, max_value=max_year, value=max_year)
        top_n = st.slider("Top N Countries", min_value=5, max_value=25, value=10)

        snap_df = df_raw[df_raw["year"] == snapshot_year].sort_values("rank").head(top_n)
        fig_rank_bar = px.bar(
            snap_df,
            x="gdp_trillions",
            y="country",
            orientation="h",
            color="continent",
            labels={"gdp_trillions": "GDP ($ Trillions)", "country": "Country"},
            title=f"Top {top_n} Largest Economies in {snapshot_year}",
            text="gdp_trillions"
        )
        fig_rank_bar.update_traces(texttemplate="$%{text:.2f}T", textposition="outside")
        fig_rank_bar.update_layout(
            template="plotly_dark",
            yaxis=dict(autorange="reversed"),
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_rank_bar, use_container_width=True)

    with c2:
        st.markdown(f"#### Economic Hegemony: China vs US Convergence")
        
        us_s = df_raw[df_raw["country"] == "the United States"].set_index("year")["gdp"]
        cn_s = df_raw[df_raw["country"] == "China"].set_index("year")["gdp"]
        jp_s = df_raw[df_raw["country"] == "Japan"].set_index("year")["gdp"]

        heg_df = pd.DataFrame({"US": us_s, "China": cn_s, "Japan": jp_s})
        heg_df["China_pct_of_US"] = (heg_df["China"] / heg_df["US"]) * 100
        heg_df["Japan_pct_of_US"] = (heg_df["Japan"] / heg_df["US"]) * 100

        fig_heg = go.Figure()
        fig_heg.add_trace(go.Scatter(
            x=heg_df.index, y=heg_df["China_pct_of_US"],
            mode="lines+markers", name="China (% of US GDP)",
            line=dict(color="#ef4444", width=3)
        ))
        j_clean = heg_df["Japan_pct_of_US"].dropna()
        fig_heg.add_trace(go.Scatter(
            x=j_clean.index, y=j_clean,
            mode="lines", name="Japan (% of US GDP, to 2019)",
            line=dict(color="#22c55e", width=2, dash="dash")
        ))
        fig_heg.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="US Parity (100%)")
        
        fig_heg.update_layout(
            title="Economy Size Relative to the US Economy (% of US GDP)",
            xaxis_title="Year",
            yaxis_title="% of US GDP",
            template="plotly_dark",
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heg, use_container_width=True)

    st.markdown("---")
    st.subheader("Historical Rank Movement (Bump Chart)")
    
    top_bump_countries = st.multiselect(
        "Select Economies to Track on Rank Chart",
        options=all_countries_list,
        default=TOP10_MODERN[:8]
    )
    if top_bump_countries:
        bump_df = df_raw[df_raw["country"].isin(top_bump_countries)]
        fig_bump = px.line(
            bump_df,
            x="year",
            y="rank",
            color="country",
            markers=True,
            title="Global Rank Trajectory Over Time (Rank 1 is Top)",
            labels={"rank": "Global Rank (Inverted)", "year": "Year"}
        )
        fig_bump.update_yaxes(autorange="reversed")
        fig_bump.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_bump, use_container_width=True)


# ==========================================
# TAB 3: CONTINENTAL DYNAMICS & POWER SHIFT
# ==========================================
with tab3:
    st.subheader("Continental Power Shifts: The Rise of Asia")

    cont_yearly = df_raw.groupby(["year", "continent"])["gdp"].sum().unstack(fill_value=0)
    cont_pct = cont_yearly.div(cont_yearly.sum(axis=1), axis=0) * 100

    col_area, col_donut = st.columns([3, 2])

    with col_area:
        fig_area = px.area(
            cont_pct.reset_index().melt(id_vars="year", var_name="continent", value_name="share_pct"),
            x="year",
            y="share_pct",
            color="continent",
            title="Evolution of Continental Share of Global GDP (1960 – 2020)",
            labels={"share_pct": "World GDP Share (%)", "year": "Year"}
        )
        fig_area.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_area, use_container_width=True)

    with col_donut:
        pie_year = st.selectbox("Select Year for Continental Pie/Donut Chart", options=sorted(df_raw["year"].unique(), reverse=True), index=0)
        pie_data = df_raw[df_raw["year"] == pie_year].groupby("continent")["gdp"].sum().reset_index()
        fig_pie = px.pie(
            pie_data,
            values="gdp",
            names="continent",
            hole=0.45,
            title=f"Continental GDP Breakdown ({pie_year})"
        )
        fig_pie.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("Hierarchical Treemap: World Economy by Continent & Country")
    
    treemap_year = st.slider("Treemap Snapshot Year", min_value=min_year, max_value=max_year, value=max_year, key="tm_slider")
    tm_df = df_raw[df_raw["year"] == treemap_year].copy()
    
    fig_tm = px.treemap(
        tm_df,
        path=["continent", "country"],
        values="gdp",
        color="continent",
        title=f"Hierarchical World GDP Composition in {treemap_year}",
        hover_data={"gdp": ":$,.0f", "rank": True, "gdp_percent": ":.2%"}
    )
    fig_tm.update_layout(template="plotly_dark", height=520)
    st.plotly_chart(fig_tm, use_container_width=True)


# ==========================================
# TAB 4: INEQUALITY & LORENZ ANALYSIS
# ==========================================
with tab4:
    st.subheader("Global Economic Inequality & Wealth Concentration")

    col_lorenz, col_gini = st.columns([1, 1])

    with col_lorenz:
        st.markdown("#### Lorenz Curves Across Benchmark Decades")
        selected_lorenz_years = st.multiselect(
            "Select Benchmark Years for Lorenz Curve",
            options=sorted(df_raw["year"].unique().tolist()),
            default=[1960, 1980, 2000, 2020]
        )

        fig_lorenz = go.Figure()
        # Perfect equality line
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="Perfect Equality",
            line=dict(color="gray", dash="dash")
        ))

        for yr in selected_lorenz_years:
            vals = np.sort(df_raw[df_raw["year"] == yr]["gdp"].values)
            cum_gdp = np.cumsum(vals) / np.sum(vals)
            cum_pop = np.linspace(0, 1, len(vals))
            g_val = calculate_gini(vals)
            fig_lorenz.add_trace(go.Scatter(
                x=cum_pop, y=cum_gdp,
                mode="lines", name=f"{yr} (Gini: {g_val:.3f})",
                line=dict(width=2.5)
            ))

        fig_lorenz.update_layout(
            title="Lorenz Curves of Inter-Country GDP Distribution",
            xaxis_title="Cumulative Share of Countries",
            yaxis_title="Cumulative Share of World GDP",
            template="plotly_dark",
            height=430
        )
        st.plotly_chart(fig_lorenz, use_container_width=True)

    with col_gini:
        st.markdown("#### Inter-Country Gini Coefficient Trend (1960 – 2020)")
        yearly_gini_series = df_raw.groupby("year")["gdp"].apply(calculate_gini).reset_index(name="gini")

        fig_gini = px.line(
            yearly_gini_series,
            x="year",
            y="gini",
            markers=True,
            title="Evolution of Inter-Country GDP Gini Index",
            labels={"gini": "Gini Coefficient (0=Equal, 1=Inequal)", "year": "Year"}
        )
        fig_gini.update_traces(line_color="#a855f7", line_width=3)
        fig_gini.update_layout(template="plotly_dark", height=430)
        st.plotly_chart(fig_gini, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Pareto Distribution & Wealth Concentration Tiers")
    
    pareto_year = st.slider("Select Pareto Analysis Year", min_value=min_year, max_value=max_year, value=max_year, key="pareto_slider")
    p_df = df_raw[df_raw["year"] == pareto_year].sort_values("gdp", ascending=False).reset_index(drop=True)
    total_p_gdp = p_df["gdp"].sum()
    n_total = len(p_df)

    top1_pct = p_df.iloc[:max(1, int(0.01 * n_total))]["gdp"].sum() / total_p_gdp * 100
    top5_pct = p_df.iloc[:max(1, int(0.05 * n_total))]["gdp"].sum() / total_p_gdp * 100
    top10_pct = p_df.iloc[:max(1, int(0.10 * n_total))]["gdp"].sum() / total_p_gdp * 100
    top20_pct = p_df.iloc[:max(1, int(0.20 * n_total))]["gdp"].sum() / total_p_gdp * 100
    bottom50_pct = p_df.iloc[int(0.50 * n_total):]["gdp"].sum() / total_p_gdp * 100

    p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
    p_c1.metric("Top 1% Countries Share", f"{top1_pct:.1f}%")
    p_c2.metric("Top 5% Countries Share", f"{top5_pct:.1f}%")
    p_c3.metric("Top 10% Countries Share", f"{top10_pct:.1f}%")
    p_c4.metric("Top 20% Countries Share", f"{top20_pct:.1f}%")
    p_c5.metric("Bottom 50% Countries Share", f"{bottom50_pct:.2f}%")


# ==========================================
# TAB 5: COUNTRY DEEP-DIVE & COMPARISON
# ==========================================
with tab5:
    st.subheader("Single Country Analysis & Head-to-Head Comparison")

    col_dd1, col_dd2 = st.columns([1, 1])
    with col_dd1:
        country_a = st.selectbox("Primary Country (Country A)", options=all_countries_list, index=all_countries_list.index("the United States") if "the United States" in all_countries_list else 0)
    with col_dd2:
        compare_mode = st.checkbox("Enable Head-to-Head Comparison", value=True)
        if compare_mode:
            country_b = st.selectbox("Comparison Country (Country B)", options=all_countries_list, index=all_countries_list.index("China") if "China" in all_countries_list else 1)
        else:
            country_b = None

    c_a_df = df_raw[df_raw["country"] == country_a].sort_values("year")
    c_b_df = df_raw[df_raw["country"] == country_b].sort_values("year") if country_b else None

    # Compound Annual Growth Rate (CAGR) calculation
    def get_cagr(df_country):
        if len(df_country) < 2:
            return 0.0
        start_val = df_country.iloc[0]["gdp"]
        end_val = df_country.iloc[-1]["gdp"]
        n_yrs = df_country.iloc[-1]["year"] - df_country.iloc[0]["year"]
        if start_val <= 0 or n_yrs <= 0:
            return 0.0
        return ((end_val / start_val) ** (1 / n_yrs) - 1) * 100

    cagr_a = get_cagr(c_a_df)
    cagr_b = get_cagr(c_b_df) if c_b_df is not None else 0.0

    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    stat_c1.metric(f"{country_a} Latest GDP", f"${c_a_df.iloc[-1]['gdp']/1e12:.2f}T" if not c_a_df.empty else "N/A", f"Rank #{c_a_df.iloc[-1]['rank']}")
    stat_c1.caption(f"Historical CAGR: **{cagr_a:.2f}%**")
    
    if c_b_df is not None and not c_b_df.empty:
        stat_c2.metric(f"{country_b} Latest GDP", f"${c_b_df.iloc[-1]['gdp']/1e12:.2f}T", f"Rank #{c_b_df.iloc[-1]['rank']}")
        stat_c2.caption(f"Historical CAGR: **{cagr_b:.2f}%**")
        
        ratio_ab = (c_a_df.iloc[-1]["gdp"] / c_b_df.iloc[-1]["gdp"]) if c_b_df.iloc[-1]["gdp"] > 0 else 0
        stat_c3.metric(f"Relative Ratio (A / B)", f"{ratio_ab:.2f}x")
        stat_c4.metric(f"Peak Rank Comparison", f"A: #{c_a_df['rank'].min()} vs B: #{c_b_df['rank'].min()}")

    # Comparative Plot
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(
        x=c_a_df["year"], y=c_a_df["gdp_trillions"],
        mode="lines+markers", name=f"{country_a} GDP ($T)",
        line=dict(color="#3b82f6", width=3)
    ))
    if c_b_df is not None and not c_b_df.empty:
        fig_comp.add_trace(go.Scatter(
            x=c_b_df["year"], y=c_b_df["gdp_trillions"],
            mode="lines+markers", name=f"{country_b} GDP ($T)",
            line=dict(color="#ef4444", width=3)
        ))
    fig_comp.update_layout(
        title=f"GDP Trajectory Comparison: {country_a}" + (f" vs {country_b}" if country_b else ""),
        xaxis_title="Year",
        yaxis_title="GDP ($ Trillion USD)",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Future Growth Simulator
    st.markdown("---")
    st.markdown("#### 🔮 Future GDP Growth Projection Simulator")
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        proj_years = st.slider("Projection Horizon (Years forward from latest)", min_value=5, max_value=30, value=15, step=5)
    with sim_col2:
        rate_a = st.slider(f"{country_a} Projected Annual Growth Rate (%)", min_value=-2.0, max_value=12.0, value=3.0, step=0.5)
    with sim_col3:
        rate_b = st.slider(f"{country_b} Projected Annual Growth Rate (%)", min_value=-2.0, max_value=12.0, value=5.0, step=0.5) if country_b else 3.0

    last_yr = int(c_a_df["year"].max())
    future_years = np.arange(last_yr, last_yr + proj_years + 1)
    
    val_a = c_a_df[c_a_df["year"] == last_yr]["gdp"].values[0] / 1e12
    proj_a = [val_a * ((1 + rate_a / 100) ** i) for i in range(len(future_years))]

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=future_years, y=proj_a, mode="lines+markers", name=f"{country_a} Proj ({rate_a}%)", line=dict(color="#3b82f6", dash="dot", width=2.5)))

    if country_b and c_b_df is not None and not c_b_df.empty:
        val_b = c_b_df[c_b_df["year"] == c_b_df["year"].max()]["gdp"].values[0] / 1e12
        proj_b = [val_b * ((1 + rate_b / 100) ** i) for i in range(len(future_years))]
        fig_sim.add_trace(go.Scatter(x=future_years, y=proj_b, mode="lines+markers", name=f"{country_b} Proj ({rate_b}%)", line=dict(color="#ef4444", dash="dot", width=2.5)))

    fig_sim.update_layout(
        title=f"Projected Economic Scale to Year {last_yr + proj_years}",
        xaxis_title="Year",
        yaxis_title="Projected GDP ($ Trillion USD)",
        template="plotly_dark",
        height=380
    )
    st.plotly_chart(fig_sim, use_container_width=True)


# ==========================================
# TAB 6: DATA EXPLORER & SQL QUERY CONSOLE
# ==========================================
with tab6:
    st.subheader("Data Explorer & Live SQL Query Console")

    st.markdown("#### Filtered Dataset View")
    st.dataframe(
        filtered_df[["year", "rank", "country", "continent", "gdp", "gdp_percent"]],
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="world_gdp_filtered.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("#### ⚡ Live SQLite Query Console (`world_gdp.db`)")
    st.caption("Execute read-only SQL queries directly against the `world_gdp` table.")

    sample_queries = {
        "Top 10 Economies in 2020": "SELECT rank, country, continent, gdp, ROUND(gdp_percent * 100, 2) AS world_share_pct FROM world_gdp WHERE year = 2020 ORDER BY rank ASC LIMIT 10;",
        "Continental Total GDP in 2020": "SELECT continent, COUNT(*) as nations, SUM(gdp) as total_gdp, ROUND(SUM(gdp_percent)*100, 2) as global_share FROM world_gdp WHERE year = 2020 GROUP BY continent ORDER BY total_gdp DESC;",
        "Fastest Growing Economies (1990 vs 2020)": "SELECT t1.country, t1.continent, t1.gdp as gdp_1990, t2.gdp as gdp_2020, ROUND((CAST(t2.gdp AS FLOAT) / t1.gdp), 1) as growth_factor FROM world_gdp t1 JOIN world_gdp t2 ON t1.country = t2.country WHERE t1.year = 1990 AND t2.year = 2020 ORDER BY growth_factor DESC LIMIT 10;"
    }

    selected_sample = st.selectbox("Load Sample SQL Query:", options=list(sample_queries.keys()))
    default_query = sample_queries[selected_sample]

    user_sql = st.text_area("SQL Query", value=default_query, height=120)

    if st.button("🚀 Run Query"):
        # Prevent any destructive queries
        destructive_keywords = ["drop", "delete", "insert", "update", "alter", "truncate", "create"]
        if any(keyword in user_sql.lower() for keyword in destructive_keywords):
            st.error("⛔ Security Alert: Only read-only `SELECT` queries are permitted.")
        else:
            try:
                db_conn = sqlite3.connect("world_gdp.db")
                sql_result = pd.read_sql_query(user_sql, db_conn)
                db_conn.close()
                st.success(f"Query returned {len(sql_result):,} rows.")
                st.dataframe(sql_result, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

