from pathlib import Path

import folium
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import HeatMapWithTime
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots


SITE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SITE_ROOT.parent
DATA_PATH = WORKSPACE_ROOT / "02806Assignment1" / "Combined_incidents_2003-present.csv"

STATIC_PATH = SITE_ROOT / "assets" / "images" / "drug_story_static.png"
TEMPORAL_HTML_PATH = SITE_ROOT / "visualizations" / "drug_temporal_story.html"
MAP_HTML_PATH = SITE_ROOT / "visualizations" / "drug_hotspot_map.html"

PALETTE = {
    "cream": "#F8E9A1",
    "coral": "#F76C6C",
    "sky": "#A8D0E6",
    "indigo": "#374785",
    "navy": "#24305E",
    "line": "#5164A8",
}

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

CATEGORY_NORMALIZATION = {
    "DRUG/NARCOTIC": "Drug Offense",
    "DRUG OFFENSE": "Drug Offense",
    "DRUG VIOLATION": "Drug Offense",
    "Drug Offense": "Drug Offense",
    "Drug Violation": "Drug Offense",
    "WARRANTS": "Warrant",
    "WARRANT": "Warrant",
    "Warrant": "Warrant",
}


def load_story_frame() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "crime_category",
            "incident_date",
            "incident_time",
            "incident_weekday",
            "police_district",
            "latitude",
            "longitude",
        ],
    )
    df["incident_date"] = pd.to_datetime(df["incident_date"])
    df = df[(df["incident_date"] >= "2003-01-01") & (df["incident_date"] <= "2025-12-31")].copy()
    df["police_district"] = df["police_district"].str.title()
    df["crime_category"] = df["crime_category"].map(lambda value: CATEGORY_NORMALIZATION.get(value, value))
    df["hour"] = pd.to_datetime(df["incident_time"], format="%H:%M").dt.hour
    return df.dropna(subset=["crime_category", "incident_weekday", "police_district"])


def make_static_figure(df: pd.DataFrame) -> None:
    drug = df[df["crime_category"].eq("Drug Offense")].copy()
    annual = (
        drug.groupby(drug["incident_date"].dt.year)
        .size()
        .reindex(range(2003, 2026), fill_value=0)
    )

    city_share = len(drug) / len(df)
    district_share = drug.groupby("police_district").size().div(df.groupby("police_district").size())
    district_ratio = district_share.div(city_share).sort_values(ascending=False).head(8).sort_values()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PALETTE["navy"],
            "figure.facecolor": PALETTE["navy"],
            "axes.edgecolor": PALETTE["sky"],
            "axes.labelcolor": PALETTE["cream"],
            "xtick.color": PALETTE["sky"],
            "ytick.color": PALETTE["sky"],
            "text.color": PALETTE["cream"],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180, gridspec_kw={"width_ratios": [1.02, 1.1]})
    fig.patch.set_facecolor(PALETTE["navy"])

    ax = axes[0]
    ax.plot(annual.index, annual.values, color=PALETTE["sky"], linewidth=3.1, marker="o", markersize=4.5)
    covid = annual.loc[2020:2021]
    ax.scatter(covid.index, covid.values, color=PALETTE["coral"], s=60, zorder=4)
    ax.axvspan(2019.5, 2021.5, color=PALETTE["coral"], alpha=0.12)
    ax.fill_between(annual.index, annual.values, color=PALETTE["sky"], alpha=0.12)
    ax.set_title("Annual Drug Offense counts, 2003-2025", loc="left", fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="y", color=PALETTE["line"], alpha=0.32, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.set_xticks([2003, 2006, 2009, 2012, 2015, 2018, 2021, 2025])
    ax.text(2009, annual.loc[2009] + 260, f"{annual.loc[2009]:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(2021, annual.loc[2021] - 220, f"{annual.loc[2021]:,}", ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(2025, annual.loc[2025] + 190, f"{annual.loc[2025]:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.annotate(
        "2009 peak",
        xy=(2009, annual.loc[2009]),
        xytext=(2005.5, annual.max() * 0.95),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": PALETTE["cream"], "lw": 1.2},
        fontsize=11,
    )
    ax.annotate(
        "COVID drop",
        xy=(2020, annual.loc[2020]),
        xytext=(2015.4, annual.max() * 0.78),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": PALETTE["cream"], "lw": 1.2},
        fontsize=11,
    )
    ax.annotate(
        "2025 rebound",
        xy=(2025, annual.loc[2025]),
        xytext=(2018.4, annual.max() * 0.58),
        textcoords="data",
        arrowprops={"arrowstyle": "-", "color": PALETTE["cream"], "lw": 1.2},
        fontsize=11,
    )

    ax = axes[1]
    colors = [PALETTE["coral"] if district == "Tenderloin" else PALETTE["sky"] for district in district_ratio.index]
    ax.barh(district_ratio.index, district_ratio.values, color=colors, edgecolor=PALETTE["cream"], linewidth=0.9)
    ax.axvline(1, color=PALETTE["cream"], linestyle="--", linewidth=1.6)
    ax.text(1.02, len(district_ratio) - 0.35, "city average", color=PALETTE["cream"], fontsize=11)
    ax.set_title("District ratio relative to citywide share", loc="left", fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="x", color=PALETTE["line"], alpha=0.32, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for district, value in district_ratio.items():
        ax.text(value + 0.05, district, f"{value:.2f}x", va="center", ha="left", fontsize=11, fontweight="bold")

    fig.suptitle(
        "A harmonized Drug Offense category stays concentrated in Tenderloin across the full 2003-2025 record",
        fontsize=22,
        fontweight="heavy",
        y=1.02,
    )
    fig.text(
        0.5,
        0.955,
        "Category harmonization follows Assignment 1: DRUG/NARCOTIC + Drug Offense + Drug Violation. Left: annual counts. Right: district share relative to the citywide Drug Offense share.",
        ha="center",
        fontsize=12,
        color=PALETTE["sky"],
    )
    fig.tight_layout()
    STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STATIC_PATH, bbox_inches="tight", facecolor=PALETTE["navy"])
    plt.close(fig)


def make_temporal_plot(df: pd.DataFrame) -> None:
    drug = df[df["crime_category"].eq("Drug Offense")].copy()
    warrant = df[df["crime_category"].eq("Warrant")].copy()

    heatmap = (
        drug.groupby(["incident_weekday", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=WEEKDAY_ORDER, columns=range(24), fill_value=0)
    )
    hour_labels = [f"{hour:02d}:00" for hour in range(24)]
    drug_hour = drug.groupby("hour").size().reindex(range(24), fill_value=0)
    warrant_hour = warrant.groupby("hour").size().reindex(range(24), fill_value=0)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.56, 0.44],
        vertical_spacing=0.14,
        subplot_titles=(
            "Drug Offense by weekday and hour",
            "Drug Offense and Warrant share the same daytime rhythm",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=heatmap.values,
            x=hour_labels,
            y=WEEKDAY_ORDER,
            colorscale=[
                [0.0, PALETTE["cream"]],
                [0.45, PALETTE["coral"]],
                [1.0, PALETTE["indigo"]],
            ],
            hovertemplate="%{y}<br>%{x}<br>Drug Offense: %{z}<extra></extra>",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hour_labels,
            y=drug_hour / drug_hour.max(),
            mode="lines",
            name="Drug Offense",
            line={"color": PALETTE["coral"], "width": 3.4},
            hovertemplate="Drug Offense<br>%{x}<br>Normalized level: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hour_labels,
            y=warrant_hour / warrant_hour.max(),
            mode="lines",
            name="Warrant",
            line={"color": PALETTE["sky"], "width": 3.4},
            hovertemplate="Warrant<br>%{x}<br>Normalized level: %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="", tickangle=-35, gridcolor=PALETTE["line"], zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="Hour of day", tickangle=-35, gridcolor=PALETTE["line"], zeroline=False, row=2, col=1)
    fig.update_yaxes(title_text="Weekday", gridcolor=PALETTE["line"], zeroline=False, row=1, col=1)
    fig.update_yaxes(title_text="Normalized level", range=[0, 1.05], gridcolor=PALETTE["line"], zeroline=False, row=2, col=1)
    fig.update_layout(
        height=760,
        paper_bgcolor=PALETTE["navy"],
        plot_bgcolor=PALETTE["navy"],
        font={"family": "Manrope, sans-serif", "color": PALETTE["cream"], "size": 15},
        margin={"l": 72, "r": 28, "t": 90, "b": 70},
        legend={"orientation": "h", "y": -0.14, "x": 0},
    )
    TEMPORAL_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(TEMPORAL_HTML_PATH, include_plotlyjs="cdn", full_html=True)


def make_hotspot_map(df: pd.DataFrame) -> None:
    drug = df[df["crime_category"].eq("Drug Offense")].dropna(subset=["latitude", "longitude"]).copy()
    drug["year"] = drug["incident_date"].dt.year
    cells = (
        drug.assign(lat_cell=drug["latitude"].round(3), lon_cell=drug["longitude"].round(3))
        .groupby(["year", "lat_cell", "lon_cell"])
        .size()
        .reset_index(name="count")
    )
    yearly_top = cells.sort_values(["year", "count"], ascending=[True, False]).groupby("year").head(140).reset_index(drop=True)
    max_count = yearly_top["count"].max()
    frames = []
    labels = []
    for year in range(2003, 2026):
        frame = yearly_top[yearly_top["year"].eq(year)].copy()
        frame["weight"] = frame["count"] / max_count
        frames.append(frame[["lat_cell", "lon_cell", "weight"]].values.tolist())
        labels.append(str(year))

    center = [yearly_top["lat_cell"].median(), yearly_top["lon_cell"].median()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    title_html = """
    <div style="
      position: fixed;
      top: 10px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 999999;
      background: rgba(247, 242, 232, 0.94);
      color: #24305E;
      padding: 10px 16px;
      border: 1px solid rgba(36,48,94,0.14);
      border-radius: 12px;
      font: 700 20px Fraunces, Georgia, serif;
      box-shadow: 0 10px 24px rgba(36,48,94,0.12);
    ">
      Drug Offense hotspots by year, 2003-2025
    </div>
    """
    subtitle_html = """
    <div style="
      position: fixed;
      top: 58px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 999998;
      background: rgba(247, 242, 232, 0.88);
      color: #24305E;
      padding: 6px 12px;
      border-radius: 10px;
      font: 500 12px Manrope, Segoe UI, sans-serif;
      box-shadow: 0 6px 18px rgba(36,48,94,0.1);
    ">
      HeatMapWithTime animation using the harmonized Drug Offense category. Darker cells indicate higher yearly concentration.
    </div>
    """
    legend_html = """
    <div style="
      position: fixed;
      bottom: 18px;
      right: 18px;
      z-index: 999999;
      background: rgba(247, 242, 232, 0.94);
      color: #24305E;
      padding: 10px 12px;
      border: 1px solid rgba(36,48,94,0.14);
      border-radius: 12px;
      font: 500 12px Manrope, Segoe UI, sans-serif;
      box-shadow: 0 10px 24px rgba(36,48,94,0.12);
      line-height: 1.45;
    ">
      <strong style="display:block; margin-bottom:6px;">How to read</strong>
      Drag the slider or press play to move through years.<br>
      Darker heat means stronger concentration that year.
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(subtitle_html))
    m.get_root().html.add_child(folium.Element(legend_html))

    HeatMapWithTime(
        data=frames,
        index=labels,
        radius=24,
        auto_play=False,
        max_opacity=0.88,
        min_opacity=0.08,
        use_local_extrema=False,
        gradient={
            0.2: PALETTE["sky"],
            0.55: PALETTE["coral"],
            1.0: PALETTE["navy"],
        },
        position="bottomleft",
        display_index=True,
        index_steps=1,
        speed_step=0.25,
    ).add_to(m)

    MAP_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    m.save(MAP_HTML_PATH)


def main() -> None:
    df = load_story_frame()
    make_static_figure(df)
    make_temporal_plot(df)
    make_hotspot_map(df)


if __name__ == "__main__":
    main()
