# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(layout="wide", page_title="Bird Data — Overview + Site Detail")
st.title("Bird Data — Overview (all sites) + Site Detail")

FILE = "WBDAT_Final_Analysed_Thalava_27 Feb 2024.xlsx"

# ------------------ helpers ------------------
def excel_col_to_index(col_label: str) -> int:
    if not isinstance(col_label, str) or not col_label:
        return -1
    col_label = col_label.strip().upper()
    val = 0
    for ch in col_label:
        if ch < "A" or ch > "Z":
            return -1
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return val - 1

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for tup in df.columns:
            parts = [str(x).strip() for x in tup if x is not None and str(x).strip() and str(x).lower() != "nan"]
            cols.append(" | ".join(parts) if parts else "")
        df.columns = cols
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df

def extract_common_before_paren(header: str) -> str:
    h = str(header).strip()
    if "(" in h:
        before = h.split("(", 1)[0].strip()
        before = before.strip("|").strip()
        return before
    return h

def year_label(y):
    try:
        yf = float(y)
        if abs(yf - int(yf)) < 1e-8:
            return str(int(yf))
        return str(round(yf, 4)).rstrip("0").rstrip(".")
    except Exception:
        return str(y)

@st.cache_data
def load_df(filepath=FILE):
    for header in ([0,1,2], [0,1], [0]):
        try:
            df = pd.read_excel(filepath, header=header)
            df = flatten_multiindex_columns(df)
            return df
        except Exception:
            continue
    df = pd.read_excel(filepath, header=None)
    df.columns = [f"COL{idx+1}" for idx in range(len(df.columns))]
    return df

# ------------------ load data ------------------
try:
    df = load_df(FILE)
except Exception as e:
    st.error(f"Failed to read file '{FILE}': {e}")
    st.stop()

# ------------------ sidebar: general options (trim/date cols) ------------------
st.sidebar.header("Data & column options")
stop_label = st.sidebar.text_input("Trim everything FROM this Excel column onward (label)", value="DK")
stop_index = excel_col_to_index(stop_label)
if stop_index >= 0 and stop_index < len(df.columns):
    df = df.iloc[:, :stop_index]
    st.sidebar.caption(f"Trimmed to columns before {stop_label} (kept {len(df.columns)} columns).")
elif stop_index >= len(df.columns):
    st.sidebar.caption(f"{stop_label} beyond file columns ({len(df.columns)}). No trimming applied.")

cols = list(df.columns)
year_guess = next((c for c in cols if "year" in c.lower()), None)
site_guess = next((c for c in cols if "site" in c.lower()), None)

st.sidebar.markdown("Select key columns (auto-detected if correct)")
year_col = st.sidebar.selectbox("Year column", options=cols, index=cols.index(year_guess) if year_guess in cols else 0)
site_col = st.sidebar.selectbox("Site column", options=cols, index=cols.index(site_guess) if site_guess in cols else 1)
year_col = str(year_col)
site_col = str(site_col)

# coerce year to numeric for grouping but treat as discrete categories later
df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

# ------------------ OVERVIEW: All-sites presence heatmap ------------------
st.header("Overview — All sites × Years (presence of any recorded data)")

# Build site-year presence matrix: 1 if any row exists for that site-year, else 0
# Use the loaded (and trimmed) df; presence means at least one survey row for that site+year
presence = df.dropna(subset=[site_col, year_col]).groupby([site_col, year_col]).size().unstack(fill_value=0)

# Ensure years are sorted in ascending order (use numeric ordering where possible)
try:
    years_all = sorted(presence.columns.astype(float))
    # map back to original column labels if needed
    years_all = [c for c in sorted(presence.columns, key=lambda x: float(x))]
except Exception:
    years_all = list(presence.columns)
# Build full list of sites in consistent order
# Count number of years with data per site
years_count_per_site = (presence > 0).sum(axis=1)

# Sort sites ascending by number of years recorded
sites_all = years_count_per_site.sort_values(ascending=True).index.tolist()

# Reindex presence to match sorted sites and all years
presence = presence.reindex(index=sites_all, columns=years_all, fill_value=0)
# Convert to binary: >0 -> 1 (recorded), 0 -> 0 (no data)
presence_bin = (presence > 0).astype(int)

# Prepare labels
x_labels_overview = [year_label(y) for y in presence_bin.columns]
y_labels_overview = presence_bin.index.tolist()

# Build heatmap: white for 1 (data exists), black for 0 (no data)
viz = presence_bin.values.astype(float)
# Colorscale: 0->black, 1->white
colorscale_overview = [(0.0, "#000000"), (1.0, "#FFFFFF")]

fig_over = go.Figure(
    data=go.Heatmap(
        z=viz,
        x=x_labels_overview,
        y=y_labels_overview,
        colorscale=colorscale_overview,
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate="Site: %{y}<br>Year: %{x}<br>Recorded: %{z}<extra></extra>"
    )
)

fig_over.update_xaxes(type="category", tickmode="array", tickvals=x_labels_overview, ticktext=x_labels_overview, tickangle=45)
fig_over.update_yaxes(type="category", automargin=True)
fig_over.update_layout(
    title="Overview: white = data recorded for that site-year; black = no data recorded",
    height=3000,
    margin=dict(t=70, b=120, l=100)
)

st.plotly_chart(fig_over, use_container_width=True)

# Small summary
num_sites = len(sites_all)
num_years = len(x_labels_overview)
st.caption(f"Sites: {num_sites} — Years: {num_years}. Hover cells to see whether data was recorded.")

st.markdown("---")

# ------------------ Continue with Site-detail UI (unchanged logic) ------------------
# The rest of the app below is the site-detail functionality (table, heatmap, trends)
# ------------------ species detection (only headers with parentheses) ------------------

species_cols = [c for c in df.columns if c not in (year_col, site_col) and "(" in c and ")" in c]
if not species_cols:
    st.error("No species-level columns found (headers with parentheses). Please check file/headers.")
    st.stop()

# full header -> common name (text before '(')
species_common = {full: extract_common_before_paren(full) for full in species_cols}
common_list_all = [species_common[f] for f in species_cols]
cnt_all = Counter(common_list_all)
name_counts_all = {}
display_names_all = []
for nm in common_list_all:
    name_counts_all.setdefault(nm, 0)
    name_counts_all[nm] += 1
    if cnt_all[nm] == 1:
        display_names_all.append(nm)
    else:
        display_names_all.append(f"{nm} ({name_counts_all[nm]})")

display_to_full_all = {disp: full for disp, full in zip(display_names_all, species_cols)}
full_to_display_all = {full: disp for disp, full in zip(display_names_all, species_cols)}

st.sidebar.markdown(f"Detected **{len(species_cols)}** species-level columns (contain parentheses).")

# ------------------ site selector ------------------
st.sidebar.header("Site selection (detail view)")
all_sites = sorted(df[site_col].dropna().unique().tolist())
selected_site = st.sidebar.selectbox("Pick a site for detail view", options=all_sites)

if not selected_site:
    st.info("Please pick a site from the sidebar to view site detail.")
    st.stop()

# ------------------ heatmap controls (for site-detail) ------------------
st.sidebar.header("Heatmap options (site detail)")
plot_choice = st.sidebar.selectbox("Graph type", options=["Heatmap (Years bottom × Species left)", "Bar chart (Birds × Counts by Year)"], key="pc")
show_numbers = st.sidebar.checkbox("Show numbers inside heatmap cells", value=True, key="nums")
hide_zero_text = st.sidebar.checkbox("Hide 0 text (still show black cells)", value=False, key="hide0")
use_log = st.sidebar.checkbox("Use log color scale (log1p for colors)", value=False, key="log")
cap_outliers = st.sidebar.checkbox("Cap extreme counts (upper cap)", value=False, key="cap")
cap_value = None
if cap_outliers:
    cap_value = st.sidebar.number_input("Cap value (max count shown)", min_value=0, value=100, key="capval")

sort_by = st.sidebar.radio("Sort species rows by", options=["Alphabetical (A→Z)", "Total count (desc)"], index=1, key="sort")
graph_short_names = st.sidebar.checkbox("Use scientific names in graph (inside parentheses) for x labels", value=False, key="short")

# ------------------ prepare data for selected site ------------------
dff = df[df[site_col] == selected_site].copy()
if dff.shape[0] == 0:
    st.error("No rows found for that site.")
    st.stop()

# coerce species columns numeric
for s in species_cols:
    dff[s] = pd.to_numeric(dff[s], errors="coerce").fillna(0)

# get years present for this site (discrete, ascending)
years_present = sorted(dff[year_col].dropna().unique().tolist())
if not years_present:
    st.error("No valid years for this site.")
    st.stop()

# pivot: rows = years, cols = full species header
pivot_years_by_species = dff.groupby(year_col)[species_cols].sum().reindex(years_present).fillna(0)

# filter out species zero for all years at this site
species_totals = pivot_years_by_species.sum(axis=0)
included_species = species_totals[species_totals > 0].index.tolist()
if len(included_species) == 0:
    st.warning("No species with non-zero counts for the selected site. Nothing to plot.")
    st.stop()
pivot_years_by_species = pivot_years_by_species[included_species]

# ------------------ TWO-LEVEL SORT as requested ------------------
species_stats = {}
for sp in included_species:
    vals = pivot_years_by_species[sp].reindex(years_present).fillna(0).values
    years_nonzero = [y for y, v in zip(years_present, vals) if v and v > 0]
    if years_nonzero:
        last_seen = max(years_nonzero)
        years_seen_count = len(years_nonzero)
    else:
        last_seen = float("-inf")
        years_seen_count = 0
    species_stats[sp] = (last_seen, years_seen_count)

included_common = [species_common[full] for full in included_species]
cnt_inc = Counter(included_common)
name_counts_inc = {}
display_names_inc = []
for nm in included_common:
    name_counts_inc.setdefault(nm, 0)
    name_counts_inc[nm] += 1
    if cnt_inc[nm] == 1:
        display_names_inc.append(nm)
    else:
        display_names_inc.append(f"{nm} ({name_counts_inc[nm]})")

display_to_full = {disp: full for disp, full in zip(display_names_inc, included_species)}
full_to_display = {full: disp for disp, full in zip(display_names_inc, included_species)}

species_order = sorted(
    included_species,
    key=lambda sp: (
        species_stats[sp][0],
        species_stats[sp][1],
        full_to_display.get(sp, extract_common_before_paren(sp)).lower()
    )
)
pivot_years_by_species = pivot_years_by_species[species_order]

# apply cap if requested
if cap_outliers and cap_value is not None:
    pivot_plot = pivot_years_by_species.clip(upper=cap_value)
else:
    pivot_plot = pivot_years_by_species.copy()

# prepare labels for graph
def get_scientific(full_lbl):
    if "(" in full_lbl and ")" in full_lbl:
        return full_lbl.split("(", 1)[-1].split(")", 1)[0].strip()
    return full_lbl

if graph_short_names:
    species_labels_for_graph = [get_scientific(s) for s in pivot_plot.columns]
else:
    species_labels_for_graph = [full_to_display.get(s, extract_common_before_paren(s)) for s in pivot_plot.columns]

# ------------------ Table: species rows x years columns (common names) ------------------
table_df = pivot_years_by_species.transpose().copy()
display_index = [full_to_display.get(full, extract_common_before_paren(full)) for full in table_df.index]
table_df.index = display_index
table_display = table_df.fillna(0).astype(int)

st.header(f"Site detail — {selected_site}")
st.subheader("Table — Birds (rows) × Years (columns)")
st.caption("Rows show common names (text before '('). Species with zero counts for every year are excluded. Sorted by last-seen year (asc) and years-seen (asc).")
st.dataframe(table_display, height=420)

# ------------------ Heatmap (Years bottom x-axis, Species left y-axis) ------------------
st.subheader("Heatmap — Years (bottom) × Species (left)")

plot_matrix_df = pivot_plot.transpose()  # rows=species, cols=years
y_labels = [full_to_display.get(full, extract_common_before_paren(full)) for full in plot_matrix_df.index]
x_labels = [year_label(y) for y in plot_matrix_df.columns]

raw_values = plot_matrix_df.values.astype(float)
if use_log:
    viz_values = np.log1p(raw_values)
    color_label = "log(1 + count)"
else:
    viz_values = raw_values
    color_label = "count"

max_v = float(np.nanmax(viz_values)) if viz_values.size else 0.0

if max_v <= 0:
    colorscale = [(0.0, "#000000"), (1.0, "#000000")]
else:
    colorscale = [
        (0.0, "#000000"),
        (1e-6, "#e6f5e6"),
        (1.0, "#006400"),
    ]

# text matrix for annotations: hide zeros if chosen
text_matrix = np.full(raw_values.shape, "", dtype=object)
for i in range(raw_values.shape[0]):
    for j in range(raw_values.shape[1]):
        v = int(raw_values[i, j])
        if v == 0:
            text_matrix[i, j] = "" if hide_zero_text else "0"
        else:
            text_matrix[i, j] = str(v)

n_species = plot_matrix_df.shape[0]
height_px = min(max(300, 24 * n_species), 2400)
tick_font_size = max(8, min(12, int(600 / max(1, n_species))))

fig = go.Figure(
    data=go.Heatmap(
        z=viz_values,
        x=x_labels,
        y=y_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=max(9, tick_font_size)),
        colorscale=colorscale,
        zmin=0,
        zmax=max_v if max_v > 0 else 1,
        colorbar=dict(title=color_label),
        hovertemplate="Species: %{y}<br>Year: %{x}<br>Count: %{customdata}<extra></extra>",
        customdata=raw_values.astype(int),
        xgap=0, ygap=0,
    )
)

fig.update_xaxes(type="category", tickmode="array", tickvals=x_labels, ticktext=x_labels, tickangle=45)
fig.update_yaxes(type="category", tickmode="array", tickvals=y_labels, ticktext=y_labels, automargin=True, tickfont=dict(size=tick_font_size))

fig.update_layout(
    title=f"Heatmap — Site: {selected_site} — X: Years (bottom) | Y: Species (left)",
    xaxis_title="Year",
    yaxis_title="Species",
    height=height_px,
    margin=dict(t=70, b=160, l=100)
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ Line trends (empty initially) ------------------
st.subheader("Line trends — choose species to plot")
trend_options = [full_to_display[full] for full in pivot_years_by_species.columns]
selected_for_trend = st.multiselect("Select species (common names) to show year-wise trends", options=trend_options, default=[])

fig_lines = go.Figure()
if selected_for_trend:
    for disp in selected_for_trend:
        full = display_to_full.get(disp)
        if full is None or full not in pivot_years_by_species.columns:
            continue
        y_vals = pivot_years_by_species[full].reindex(years_present).fillna(0).values
        fig_lines.add_trace(go.Scatter(x=[year_label(y) for y in years_present], y=y_vals, mode="lines+markers", name=disp,
                                       hovertemplate="Year: %{x}<br>Count: %{y}<extra></extra>"))
    fig_lines.update_layout(title="Selected species trends", xaxis_title="Year", yaxis_title="Count", height=420)
else:
    fig_lines.update_layout(title="Select one or more species from the multiselect to display trends", xaxis_title="Year", yaxis_title="Count", height=180)

st.plotly_chart(fig_lines, use_container_width=True)

st.markdown("---")
st.write("Notes:")
st.write("- Top overview shows presence (white) or absence (black) of any recorded data for each site-year across the whole file.")
st.write("- Site-detail view below retains earlier behaviour: species-level heatmap, table and trends (species with all-zero for the site are excluded).")
