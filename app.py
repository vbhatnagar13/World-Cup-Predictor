import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import unicodedata
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0a1628; }
[data-testid="stSidebar"] * { color: #e0e6f0 !important; }
.metric-card {
    background: #0d1f3c; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #1e3a5f; margin-bottom: 8px;
}
.prob-label { font-size: 0.75rem; color: #8a9bbf; text-transform: uppercase; letter-spacing: 1px; }
h1, h2, h3 { font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── ELO RATINGS ─────────────────────────────────────────────────────────────
WC2026_ELO = {
    'Argentina': 2052, 'France': 1992, 'Brazil': 1962, 'Portugal': 1958,
    'Netherlands': 1952, 'Spain': 1981, 'England': 1966, 'Italy': 1930,
    'Germany': 1932, 'Uruguay': 1912, 'Croatia': 1918, 'Belgium': 1916,
    'Switzerland': 1906, 'Japan': 1880, 'Mexico': 1878, 'Morocco': 1857,
    'South Korea': 1853, 'USA': 1855, 'Australia': 1832, 'Ecuador': 1831,
    'Austria': 1893, 'Denmark': 1888, 'Turkey': 1875, 'Senegal': 1848,
    'Iran': 1826, 'Serbia': 1862, 'Colombia': 1898, 'Qatar': 1782,
    'Saudi Arabia': 1794, 'Canada': 1804, 'Costa Rica': 1810, 'Poland': 1857,
    'Ghana': 1780, 'Tunisia': 1782, 'Cameroon': 1796, 'South Africa': 1789,
    'Venezuela': 1799, 'Bolivia': 1742, 'Panama': 1763, 'Honduras': 1758,
    'Jamaica': 1766, 'New Zealand': 1701, 'Iraq': 1769, 'Uzbekistan': 1751,
    'DR Congo': 1765, "Côte d'Ivoire": 1809, 'Egypt': 1820, 'Nigeria': 1815,
}

ROLE_COLORS = {
    'Possession Mid':           '#2a9d8f',
    'Defender':                 '#6a4c93',
    'Attacking Forward/Winger': '#e63946',
    'Box-to-Box Mid':           '#457b9d',
    'Target Forward':           '#f4a261',
    'Goalkeeper':               '#6c757d',
}

RADAR_STATS  = ['shots_p90', 'xg_p90', 'kp_p90', 'pass_pct',
                'carries_p90', 'drib_p90', 'press_p90', 'def_p90']
RADAR_LABELS = ['Shots/90', 'xG/90', 'KP/90', 'Pass %',
                'Carries/90', 'Dribbles/90', 'Press/90', 'Def/90']

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    csv_path = Path(__file__).parent / "wc_players.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    # Normalise column name differences between WC/Euro/Copa slices
    if 'pressures_p90' in df.columns and 'press_p90' not in df.columns:
        df['press_p90'] = df['pressures_p90']
    if 'assists_p90' not in df.columns:
        df['assists_p90'] = 0.0

    p90_cols = ['shots_p90', 'xg_p90', 'goals_p90', 'passes_p90', 'kp_p90',
                'assists_p90', 'carries_p90', 'press_p90', 'drib_p90', 'def_p90', 'pass_pct']
    for col in p90_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


@st.cache_data(show_spinner=False)
def compute_pca(_df):
    """PCA on outfield players across all tournaments."""
    outfield = _df[_df['position'] != 'Goalkeeper'].copy()
    X = outfield[RADAR_STATS].fillna(0).values
    Xsc = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xsc)
    outfield = outfield.copy()
    outfield['pc1'] = coords[:, 0]
    outfield['pc2'] = coords[:, 1]
    return outfield, pca.explained_variance_ratio_


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def norm_str(s):
    return unicodedata.normalize('NFD', str(s)).encode('ascii', 'ignore').decode().lower()


def elo_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def top_players(df, team, n=6):
    t = df[df['team'] == team].copy()
    if t.empty:
        return pd.DataFrame()
    t['_score'] = (
        t['xg_p90'].fillna(0) * 3 +
        t['assists_p90'].fillna(0) * 2 +
        t['kp_p90'].fillna(0) * 1
    )
    return (t.sort_values('_score', ascending=False)
             .drop_duplicates('player', keep='first')
             .head(n))


def radar_chart(player_row, role):
    color = ROLE_COLORS.get(role, '#888')
    norm_vals = []
    for stat in RADAR_STATS:
        v = float(player_row.get(stat, 0) or 0)
        # rough normalisation: clip at 95th percentile for the column
        norm_vals.append(min(v, 10) / 10)  # simplistic but visual
    norm_vals.append(norm_vals[0])
    labels_closed = RADAR_LABELS + [RADAR_LABELS[0]]

    fig = go.Figure(go.Scatterpolar(
        r=norm_vals, theta=labels_closed,
        fill='toself', line_color=color,
        fillcolor=color, opacity=0.45,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False, height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ─── PAGE: MATCH PREDICTOR ───────────────────────────────────────────────────
def page_predictor(df):
    st.title("⚽ Match Predictor")
    st.caption("Elo-based win probabilities · key player breakdown from WC 2022, Euro 2024 & Copa 2024")

    teams = sorted(WC2026_ELO.keys())
    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        team_a = st.selectbox("Team A", teams, index=teams.index("Brazil"))
    with col_vs:
        st.markdown("<br><h3 style='text-align:center;color:#8a9bbf'>vs</h3>", unsafe_allow_html=True)
    with col2:
        team_b = st.selectbox("Team B", teams, index=teams.index("France"))

    if team_a == team_b:
        st.warning("Select two different teams.")
        return

    ea, eb = WC2026_ELO[team_a], WC2026_ELO[team_b]
    pa = elo_prob(ea, eb) * 0.75
    pb = elo_prob(eb, ea) * 0.75
    draw = 0.25
    diff = ea - eb

    st.markdown("---")

    # ── Probability stacked bar ──
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=team_a, x=[round(pa * 100, 1)], y=[""],
        orientation='h', marker_color='#2a9d8f',
        text=[f"  {team_a}  {pa*100:.1f}%"], textposition='inside',
        insidetextanchor='start', textfont=dict(size=14, color='white'),
    ))
    fig.add_trace(go.Bar(
        name="Draw", x=[round(draw * 100, 1)], y=[""],
        orientation='h', marker_color='#4a5568',
        text=["Draw  25%"], textposition='inside',
        textfont=dict(size=13, color='#ccc'),
    ))
    fig.add_trace(go.Bar(
        name=team_b, x=[round(pb * 100, 1)], y=[""],
        orientation='h', marker_color='#e63946',
        text=[f"{pb*100:.1f}%  {team_b}  "], textposition='inside',
        insidetextanchor='end', textfont=dict(size=14, color='white'),
    ))
    fig.update_layout(
        barmode='stack', height=80,
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{team_a} Elo", ea)
    m2.metric(f"{team_a} win", f"{pa*100:.1f}%")
    m3.metric("Draw", "25.0%")
    m4.metric(f"{team_b} win", f"{pb*100:.1f}%")
    m5.metric(f"{team_b} Elo", eb)

    if abs(diff) < 50:
        st.info("🤝 Roughly equal matchup — could go either way.")
    elif diff > 0:
        st.success(f"📈 **{team_a}** are favourites (Elo +{diff})")
    else:
        st.success(f"📈 **{team_b}** are favourites (Elo +{-diff})")

    st.markdown("---")
    st.subheader("Key Players by Attack Contribution")

    c1, c2 = st.columns(2)
    for col, team, color in [(c1, team_a, '#2a9d8f'), (c2, team_b, '#e63946')]:
        with col:
            st.markdown(f"<h4 style='color:{color}'>{team}</h4>", unsafe_allow_html=True)
            kp = top_players(df, team)
            if kp.empty:
                st.info("No StatsBomb data for this team (Elo prediction still applies).")
            else:
                display = kp[['player', 'role', 'source', 'goals', 'xg_p90', 'kp_p90', 'pass_pct']].copy()
                display.columns = ['Player', 'Role', 'Tournament', 'Goals', 'xG/90', 'KP/90', 'Pass%']
                display['xG/90'] = display['xG/90'].round(2)
                display['KP/90'] = display['KP/90'].round(2)
                display['Pass%'] = display['Pass%'].round(1)
                st.dataframe(display, hide_index=True, use_container_width=True)


# ─── PAGE: PLAYER SEARCH ─────────────────────────────────────────────────────
def page_search(df):
    st.title("🔍 Player Search")
    st.caption("Search across all three tournaments · accent-insensitive")

    query = st.text_input(
        "Player name", placeholder="Messi · Yamal · Bellingham · Vinicius · Modric…"
    )

    if not query:
        st.info("Type a name above to search.")
        # Show some star players as suggestions
        stars = ['Lionel', 'Kylian', 'Lamine', 'Jude', 'Cristiano', 'Neymar',
                 'Vinicius', 'Musiala', 'Bellingham', 'Havertz']
        st.markdown("**Popular searches:**")
        cols = st.columns(5)
        for i, name in enumerate(stars):
            if cols[i % 5].button(name, key=f"suggest_{name}"):
                query = name
        if not query:
            return

    qn = norm_str(query)
    hits = df[df['player'].apply(lambda x: qn in norm_str(x))]

    if hits.empty:
        st.warning(f"No players found matching **'{query}'**.")
        return

    players_found = hits['player'].unique()
    st.success(f"Found {len(players_found)} player(s)")

    for player_name in players_found[:10]:  # cap at 10 results
        rows = hits[hits['player'] == player_name].copy()
        team = rows['team'].iloc[0]
        role = rows['role'].dropna().iloc[0] if rows['role'].notna().any() else 'Unknown'
        role_color = ROLE_COLORS.get(role, '#888')

        with st.expander(f"**{player_name}** · {team} · {role}", expanded=(len(players_found) == 1)):
            tab_stats, tab_radar = st.tabs(["📊 Stats", "🕸️ Radar"])

            with tab_stats:
                display_cols = {
                    'source': 'Tournament', 'minutes': 'Mins', 'goals': 'Goals',
                    'xg_p90': 'xG/90', 'kp_p90': 'KP/90', 'assists_p90': 'Ast/90',
                    'pass_pct': 'Pass%', 'shots_p90': 'Shots/90',
                    'carries_p90': 'Carries/90', 'drib_p90': 'Drib/90',
                    'press_p90': 'Press/90', 'def_p90': 'Def/90',
                }
                avail = {k: v for k, v in display_cols.items() if k in rows.columns}
                display = rows[list(avail.keys())].rename(columns=avail)
                float_cols = [c for c in display.columns if c not in ('Tournament', 'Mins', 'Goals')]
                display[float_cols] = display[float_cols].round(2)
                st.dataframe(display.reset_index(drop=True), hide_index=True, use_container_width=True)

            with tab_radar:
                # Use best tournament row (highest xg_p90)
                best = rows.sort_values('xg_p90', ascending=False).iloc[0]
                src_label = best.get('source', '')
                st.caption(f"Radar based on **{src_label}** performance (normalised to 0–10 per stat)")
                fig = radar_chart(best, role)
                st.plotly_chart(fig, use_container_width=True)

    if len(players_found) > 10:
        st.caption(f"Showing first 10 of {len(players_found)} matches. Try a more specific name.")


# ─── PAGE: CLUSTER VISUALIZATION ─────────────────────────────────────────────
def page_clusters(df):
    st.title("🗺️ Player Clusters")
    st.caption("PCA projection of all outfield players coloured by their statistical role")

    outfield, var_ratio = compute_pca(df)

    filter_col, info_col = st.columns([3, 7])
    with filter_col:
        sources = ['All'] + sorted(outfield['source'].dropna().unique().tolist())
        src = st.selectbox("Tournament", sources)
        roles_all = ['All'] + sorted(outfield['role'].dropna().unique().tolist())
        role_filter = st.selectbox("Role", roles_all)

    plot_df = outfield.copy()
    if src != 'All':
        plot_df = plot_df[plot_df['source'] == src]
    if role_filter != 'All':
        plot_df = plot_df[plot_df['role'] == role_filter]

    with info_col:
        st.markdown(f"**{len(plot_df)}** players shown")
        role_md = '  '.join(
            f"<span style='color:{c}'>⬤</span> {r}"
            for r, c in ROLE_COLORS.items() if r != 'Goalkeeper'
        )
        st.markdown(role_md, unsafe_allow_html=True)

    # ── Scatter ──
    fig = px.scatter(
        plot_df, x='pc1', y='pc2', color='role',
        color_discrete_map=ROLE_COLORS,
        hover_name='player',
        hover_data={
            'pc1': False, 'pc2': False,
            'team': True, 'role': True,
            'source': True,
            'xg_p90': ':.2f', 'kp_p90': ':.2f', 'pass_pct': ':.1f',
        },
        labels={
            'pc1': f'PC1 ({var_ratio[0]*100:.1f}% var)',
            'pc2': f'PC2 ({var_ratio[1]*100:.1f}% var)',
            'role': 'Role',
        },
        height=560,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color='white')))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend_title='Role',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.07)'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Role breakdown bars ──
    st.subheader("Role Distribution")
    role_counts = plot_df['role'].value_counts().reset_index()
    role_counts.columns = ['Role', 'Count']
    fig2 = px.bar(
        role_counts, x='Role', y='Count', color='Role',
        color_discrete_map=ROLE_COLORS, text='Count',
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(
        showlegend=False, height=280,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, showticklabels=False),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─── SIDEBAR + MAIN ──────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## ⚽ WC 2026 Predictor")
        st.caption("StatsBomb open data · WC 2022 · Euro 2024 · Copa 2024")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["⚽ Match Predictor", "🔍 Player Search", "🗺️ Cluster Viz"],
            label_visibility="collapsed",
        )
        st.markdown("---")

    with st.spinner("Loading player data…"):
        df = load_data()

    if df is None:
        st.error("**`wc_players.csv` not found.**")
        st.markdown("""
Run the data pipeline first:
```bash
pip install statsbombpy pandas scikit-learn
python generate_data.py
```
Then restart the app.
        """)
        return

    with st.sidebar:
        st.caption(f"📊 {len(df):,} records")
        st.caption(f"🏆 {', '.join(df['source'].dropna().unique())}")
        st.caption(f"👥 {df['team'].nunique()} teams · {df['player'].nunique()} players")

    if page == "⚽ Match Predictor":
        page_predictor(df)
    elif page == "🔍 Player Search":
        page_search(df)
    elif page == "🗺️ Cluster Viz":
        page_clusters(df)


if __name__ == "__main__":
    main()
