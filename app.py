"""
app.py - Falconix Engineering GmbH
Streamlit Web-Applikation fuer das Tool "Biegewiderstand (MRd)" nach DAfStb-Richtlinie Stahlfaserbeton
inkl. Mitarbeiter-Login-Zuordnung, dynamischer Querschnittsgrafik, Momenten- & Druckzonen-Diagrammen,
Simplified & Full Bewehrungsmodus (mit Stabbündeln) und 7-seitigem WeasyPrint-PDF-Export.
"""
import datetime
import math
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from mrd_core import (
    CONCRETE_CLASSES,
    STEEL_CLASSES,
    PERFORMANCE_CLASS_VALUES,
    REBAR_DIAMETERS,
    UserMetadata,
    MRdInput,
    SimplifiedRebarInput,
    RebarLayerInput,
     MRdResult,
    calculate_mrd,
)
from mrd_pdf import generate_mrd_pdf

st.set_page_config(page_title="Falconix Engineering - MRd Slab", layout="wide")


def plot_ui_cross_section(res: MRdResult):
    """Zeichnet den Querschnitt exakt im Stil von Antonios Excel-UI (grüne obere Lagen, orangene untere Lagen, Bemaßungen)."""
    inp = res.inputs
    h = inp.h_mm
    w_slab = 900.0

    fig, ax = plt.subplots(figsize=(8.0, 2.7), dpi=140)
    fig.patch.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")

    # Betonquerschnitt
    slab_rect = patches.Rectangle((0, 0), w_slab, h, linewidth=1.2, edgecolor="#222222", facecolor="#cccccc", zorder=2)
    ax.add_patch(slab_rect)

    # Obere Bewehrung (Grün wie in Antonios UI)
    t1 = res.steel_layers.get("t1")
    if t1 and t1.as_cm2_m > 0:
        d_top = t1.diam_1
        s_top = t1.spacing
        y_bar_top = h - t1.z_from_top_mm
        # Querlage als grüner Balken darunter
        bar_rect_top = patches.Rectangle((0, y_bar_top - 1.5 * d_top), w_slab, d_top, facecolor="#00b000", edgecolor="none", zorder=3)
        ax.add_patch(bar_rect_top)
        # Längsstäbe als grüne Kreise
        xs_t = np.arange(115.0, w_slab - 80.0, s_top)
        for x_pos in xs_t:
            circ = patches.Circle((x_pos, y_bar_top), radius=d_top / 2.0, facecolor="#00b000", edgecolor="#008000", linewidth=0.5, zorder=4)
            ax.add_patch(circ)
        # Bemaßung Stababstand oben (s und lichter Abstand s - d)
        if len(xs_t) >= 2:
            x0, x1 = xs_t[0], xs_t[1]
            lichte_top = int(round(s_top - d_top))
            ax.plot([x0, x0], [h + 6, h + 75], color="#111", lw=1.0)
            ax.plot([x1, x1], [h + 6, h + 75], color="#111", lw=1.0)
            ax.plot([x0 + d_top/2, x0 + d_top/2], [h + 6, h + 40], color="#111", lw=0.9)
            ax.plot([x1 - d_top/2, x1 - d_top/2], [h + 6, h + 40], color="#111", lw=0.9)
            ax.annotate("", xy=(x1, h + 68), xytext=(x0, h + 68), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.1))
            ax.text((x0 + x1)/2, h + 73, f"{int(round(s_top))}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            ax.annotate("", xy=(x1 - d_top/2, h + 32), xytext=(x0 + d_top/2, h + 32), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.0))
            ax.text((x0 + x1)/2, h + 37, f"{lichte_top}", ha="center", va="bottom", fontsize=9)
        # Betondeckung oben links
        ax.plot([-55, -8], [h, h], color="#111", lw=1.0)
        ax.plot([-55, -8], [h - t1.c_mm, h - t1.c_mm], color="#111", lw=1.0)
        ax.annotate("", xy=(-32, h), xytext=(-32, h - t1.c_mm), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.0))
        ax.text(-42, h - t1.c_mm / 2, f"{int(round(t1.c_mm))}", ha="right", va="center", rotation=90, fontsize=9)

    # Untere Bewehrung (Orange wie in Antonios UI)
    b1 = res.steel_layers.get("b1")
    if b1 and b1.as_cm2_m > 0:
        d_bot = b1.diam_1
        s_bot = b1.spacing
        y_bar_bot = h - b1.z_from_top_mm
        # Querlage als orangener Balken darüber
        bar_rect_bot = patches.Rectangle((0, y_bar_bot + 0.5 * d_bot), w_slab, d_bot, facecolor="#ff8000", edgecolor="none", zorder=3)
        ax.add_patch(bar_rect_bot)
        # Längsstäbe als orangene Kreise
        xs_b = np.arange(115.0, w_slab - 80.0, s_bot)
        for x_pos in xs_b:
            circ = patches.Circle((x_pos, y_bar_bot), radius=d_bot / 2.0, facecolor="#ff8000", edgecolor="#cc6600", linewidth=0.5, zorder=4)
            ax.add_patch(circ)
        # Bemaßung Stababstand unten
        if len(xs_b) >= 2:
            x0, x1 = xs_b[0], xs_b[1]
            lichte_bot = int(round(s_bot - d_bot))
            ax.plot([x0, x0], [-6, -75], color="#111", lw=1.0)
            ax.plot([x1, x1], [-6, -75], color="#111", lw=1.0)
            ax.plot([x0 + d_bot/2, x0 + d_bot/2], [-6, -40], color="#111", lw=0.9)
            ax.plot([x1 - d_bot/2, x1 - d_bot/2], [-6, -40], color="#111", lw=0.9)
            ax.annotate("", xy=(x1 - d_bot/2, -32), xytext=(x0 + d_bot/2, -32), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.0))
            ax.text((x0 + x1)/2, -28, f"{lichte_bot}", ha="center", va="bottom", fontsize=9)
            ax.annotate("", xy=(x1, -68), xytext=(x0, -68), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.1))
            ax.text((x0 + x1)/2, -64, f"{int(round(s_bot))}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        # Betondeckung unten links
        ax.plot([-55, -8], [0, 0], color="#111", lw=1.0)
        ax.plot([-55, -8], [b1.c_mm, b1.c_mm], color="#111", lw=1.0)
        ax.annotate("", xy=(-32, b1.c_mm), xytext=(-32, 0), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.0))
        ax.text(-42, b1.c_mm / 2, f"{int(round(b1.c_mm))}", ha="right", va="center", rotation=90, fontsize=9)

    # Gesamthöhe h rechts
    ax.plot([w_slab + 15, w_slab + 85], [0, 0], color="#111", lw=1.0)
    ax.plot([w_slab + 15, w_slab + 85], [h, h], color="#111", lw=1.0)
    ax.annotate("", xy=(w_slab + 65, h), xytext=(w_slab + 65, 0), arrowprops=dict(arrowstyle="<->", color="#111", lw=1.3))
    ax.text(w_slab + 52, h / 2, f"{int(round(h))}", ha="right", va="center", rotation=90, fontsize=12, fontweight="bold")

    ax.set_xlim(-85, w_slab + 105)
    ax.set_ylim(-95, h + 95)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_ui_moments_chart(res: MRdResult):
    """Zeichnet das kombinierte Momenten-Dehnungs-Diagramm (Negative Moments blau, Positive Moments orange) wie in Antonios Excel-UI."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=130)
    ax.plot(res.neg_moment.curve_eps_permille, res.neg_moment.curve_m_knm_m, color="#4a7ebb", linewidth=1.8, label="Negative Moments")
    ax.plot(res.pos_moment.curve_eps_permille, res.pos_moment.curve_m_knm_m, color="#e46c0a", linewidth=1.8, label="Positive Moments")
    ax.set_title("Moments", fontsize=11, color="#444444", pad=8)
    ax.set_xlabel("Tensile Strain ‰", fontsize=8.5, color="#444444")
    ax.set_ylabel("Bending Moment kNm/m", fontsize=8.5, color="#444444")
    ax.set_xlim(0.0, 25.0)
    m_ext = max(abs(min(res.neg_moment.curve_m_knm_m)), abs(max(res.pos_moment.curve_m_knm_m)), 10.0)
    y_bound = math.ceil(m_ext * 1.15 / 10.0) * 10.0
    ax.set_ylim(y_bound, -y_bound) # Oben negativ (-150), unten positiv (+200) wie bei Antonio!
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.tick_params(labelsize=8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    plt.tight_layout()
    return fig


def plot_ui_compression_height_chart(res: MRdResult):
    """Zeichnet den Verlauf der Betondruckzonenhoehe x [mm] ueber der Zugdehnung [‰] wie in Antonios Excel-UI."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=130)
    ax.plot(res.neg_moment.curve_eps_permille[1:], res.neg_moment.curve_x_mm[1:], color="#4a7ebb", linewidth=1.8, label="Negative Moments")
    ax.plot(res.pos_moment.curve_eps_permille[1:], res.pos_moment.curve_x_mm[1:], color="#e46c0a", linewidth=1.8, label="Positive Moments")
    ax.set_title("Compression Height", fontsize=11, color="#444444", pad=8)
    ax.set_xlabel("Tensile Strain ‰", fontsize=8.5, color="#444444")
    ax.set_ylabel("Compression Height mm", fontsize=8.5, color="#444444")
    ax.set_xlim(0.0, 25.0)
    x_max = max(max(res.neg_moment.curve_x_mm[1:]), max(res.pos_moment.curve_x_mm[1:]), 50.0)
    ax.set_ylim(math.ceil(x_max * 1.15 / 10.0) * 10.0, 0.0) # 0 oben, max unten wie bei Antonio!
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#f0f0f0", linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.tick_params(labelsize=8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1. MITARBEITER-DATENBANK
# ---------------------------------------------------------------------------
EMPLOYEES = {
    "Pavel Barleben": {
        "kuerzel": "PB",
        "email": "barleben@falconixengineering.com",
        "phone": "+49 6081 0000-01",
    },
    "Antonio Brito": {
        "kuerzel": "AB",
        "email": "brito@falconixengineering.com",
        "phone": "+351 96 139 14 66",
    },
    "Weiterer Mitarbeiter": {
        "kuerzel": "WM",
        "email": "info@falconixengineering.com",
        "phone": "+49 6081 0000-00",
    },
}

# ---------------------------------------------------------------------------
# 2. SIDEBAR: LOGIN / BEARBEITER & PROJEKTDATEN
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Angemeldeter Bearbeiter")
    selected_user = st.selectbox("Mitarbeiter auswählen", list(EMPLOYEES.keys()), index=0)
    emp = EMPLOYEES[selected_user]
    user_email = st.text_input("E-Mail", value=emp["email"])
    user_phone = st.text_input("Telefon", value=emp["phone"])

    st.markdown("---")
    st.markdown("### 📁 Projekt & Kunde")
    project_name = st.text_input("Projekt / Bauvorhaben", value="Test Dehnungen_1")
    position_name = st.text_input("Bauteil / Position", value="250 mm")
    client_name = st.text_input("Kunde / Auftraggeber", value="")
    calc_date = st.date_input("Datum", value=datetime.date.today())

# ---------------------------------------------------------------------------
# 3. HAUPTBEREICH: EINGABEMASKE (LINKS) & ERGEBNISSE + GRAFIKEN (RECHTS)
# ---------------------------------------------------------------------------
st.title("Moment Capacity for SFRC – Slabs (MRd)")
st.caption("Bemessung nach DAfStb-Richtlinie Stahlfaserbeton (06/2021) in Verbindung mit DIN EN 1992-1-1")

col_in, col_out = st.columns([1.05, 1.15], gap="large")

with col_in:
    st.subheader("1. Querschnitt & Baustoffe")
    c1, c2, c3 = st.columns(3)
    with c1:
        h_mm = st.number_input("Bauteilstärke h [mm]", min_value=70.0, max_value=1500.0, value=250.0, step=10.0)
        tensile_dir = st.selectbox("Berechnungsrichtung", ["r", "s"], index=0)
    with c2:
        concrete_class = st.selectbox("Betonfestigkeitsklasse", list(CONCRETE_CLASSES.keys()), index=1)
        steel_class = st.selectbox("Betonstahlklasse", list(STEEL_CLASSES.keys()), index=2)
    with c3:
        n_ed = st.number_input("Normalkraft N_Ed (+ Druck) [kN/m]", value=0.0, step=5.0)
        kappa_g = st.number_input("Bauteilgröße κ_G^f", min_value=1.0, max_value=1.7, value=1.70, step=0.05, disabled=True)

    st.subheader("2. Stahlfaserbeton")
    fiber_concept = st.radio(
        "Stahlfaser-Konzept",
        ["Performance Class", "No Steel Fibers", "Residual Strength"],
        horizontal=True,
        index=0,
    )
    L1_val, L2_val = 1.5, 1.5
    f_R1m_val, f_R4m_val = 1.80, 1.70

    if fiber_concept == "Performance Class":
        fc1, fc2 = st.columns(2)
        with fc1:
            L1_val = st.selectbox("Leistungsklasse L1 (GZG / δ=0,5 mm)", PERFORMANCE_CLASS_VALUES[1:], index=4)
        with fc2:
            L2_val = st.selectbox("Leistungsklasse L2 (GZT / δ=3,5 mm)", PERFORMANCE_CLASS_VALUES[1:], index=4)
        if L2_val / L1_val < 0.7:
            st.warning("Hinweis: L2/L1 < 0,7 – nach DAfStb-Richtlinie R.3.6.3 (3) ist für GZT der Spannungsblock zu prüfen.")
    elif fiber_concept == "Residual Strength":
        rc1, rc2 = st.columns(2)
        with fc1 if "fc1" in locals() else rc1:
            f_R1m_val = st.number_input("f_R1,m [N/mm²]", min_value=0.0, value=1.80, step=0.1)
        with rc2:
            f_R4m_val = st.number_input("f_R4,m [N/mm²]", min_value=0.0, value=1.70, step=0.1)

    st.subheader("3. Betonstahlbewehrung")
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        rebar_concept = st.selectbox(
            "Bewehrungskonzept",
            ["Fibers Only", "Top + Bottom", "Top", "Bottom"],
            index=1,
        )
    with rc_col2:
        rebar_definition = st.selectbox(
            "Eingabemodus",
            ["Simplified", "Full"],
            index=0,
            disabled=(rebar_concept == "Fibers Only"),
        )

    cov1, cov2 = st.columns(2)
    with cov1:
        cover_top = st.number_input(
            "Betondeckung c_oben [mm]", min_value=15.0, max_value=150.0, value=35.0, step=5.0,
            disabled=(rebar_concept not in ("Top", "Top + Bottom"))
        )
    with cov2:
        cover_bot = st.number_input(
            "Betondeckung c_unten [mm]", min_value=15.0, max_value=150.0, value=35.0, step=5.0,
            disabled=(rebar_concept not in ("Bottom", "Top + Bottom"))
        )

    simp_input = SimplifiedRebarInput()
    full_top_list = []
    full_bot_list = []

    if rebar_concept != "Fibers Only":
        if rebar_definition == "Simplified":
            st.markdown("##### Vereinfachte Bewehrungseingabe (Kreuznetz Lage 1 & 2)")
            top_active = rebar_concept in ("Top", "Top + Bottom")
            bot_active = rebar_concept in ("Bottom", "Top + Bottom")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Oben (Top – Layers 1 and 2)**")
                t_d = st.selectbox("Ø Oben [mm]", REBAR_DIAMETERS, index=7, key="s_td", disabled=not top_active)
                t_s = st.number_input("Abstand s Oben [mm]", min_value=20.0, value=150.0, step=25.0, key="s_ts", disabled=not top_active)
                dir_top = st.selectbox("Richtung 1. Lage Oben", ["r", "s"], index=0, disabled=not top_active)
            with sc2:
                st.markdown("**Unten (Bottom – Layers 1 and 2)**")
                b_d = st.selectbox("Ø Unten [mm]", REBAR_DIAMETERS, index=9, key="s_bd", disabled=not bot_active)
                b_s = st.number_input("Abstand s Unten [mm]", min_value=20.0, value=150.0, step=25.0, key="s_bs", disabled=not bot_active)
                dir_bot = st.selectbox("Richtung 1. Lage Unten", ["r", "s"], index=0, disabled=not bot_active)

            simp_input = SimplifiedRebarInput(
                top_diam=float(t_d), top_spacing=float(t_s),
                bottom_diam=float(b_d), bottom_spacing=float(b_s),
            )
            dir_cen = "r"
        else:
            st.markdown("##### Volle Bewehrungsdefinition (inkl. Mischbewehrung Ø1+Ø2 & Stabbündel)")
            default_top_rows = [
                {"Lage": "1 (r)", "Richtung": "r", "Diam_1": 16, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "2 (s)", "Richtung": "s", "Diam_1": 16, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "3 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "4 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "5 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "6 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
            ]
            default_bot_rows = [
                {"Lage": "1 (r)", "Richtung": "r", "Diam_1": 20, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "2 (s)", "Richtung": "s", "Diam_1": 20, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "3 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "4 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "5 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "6 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
            ]
            if rebar_concept in ("Top", "Top + Bottom"):
                st.markdown("**Obere Bewehrungslagen (1 bis 6)**")
                df_top = st.data_editor(pd.DataFrame(default_top_rows), key="ed_top", use_container_width=True, hide_index=True)
                for idx, row in df_top.iterrows():
                    full_top_list.append(RebarLayerInput(
                        layer_index=int(idx) + 1,
                        direction=row["Richtung"],
                        diam_1=float(row["Diam_1"]),
                        diam_2=float(row["Diam_2"]),
                        spacing=float(row["Abstand_s"]),
                        bundled=bool(row["Stabbuendel"]),
                    ))
            if rebar_concept in ("Bottom", "Top + Bottom"):
                st.markdown("**Untere Bewehrungslagen (1 bis 6)**")
                df_bot = st.data_editor(pd.DataFrame(default_bot_rows), key="ed_bot", use_container_width=True, hide_index=True)
                for idx, row in df_bot.iterrows():
                    full_bot_list.append(RebarLayerInput(
                        layer_index=int(idx) + 1,
                        direction=row["Richtung"],
                        diam_1=float(row["Diam_1"]),
                        diam_2=float(row["Diam_2"]),
                        spacing=float(row["Abstand_s"]),
                        bundled=bool(row["Stabbuendel"]),
                    ))
            dir_top = dir_cen = dir_bot = "r"
    else:
        dir_top = dir_cen = dir_bot = "r"

# ---------------------------------------------------------------------------
# 4. BERECHNUNG AUSFÜHREN & ERGEBNISSE + GRAFIKEN ANZEIGEN
# ---------------------------------------------------------------------------
meta_obj = UserMetadata(
    user_name=selected_user,
    user_kuerzel=emp["kuerzel"],
    email=user_email,
    phone=user_phone,
    project_name=project_name,
    client_name=client_name,
    position_name=position_name,
    date_str=calc_date.strftime("%d-%m-%Y"),
    version="V1.00.00",
)

inp_obj = MRdInput(
    h_mm=h_mm,
    tensile_direction=tensile_dir,
    n_ed_kn_m=n_ed,
    concrete_class=concrete_class,
    steel_class=steel_class,
    fiber_concept=fiber_concept,
    L1=L1_val,
    L2=L2_val,
    f_R1m=f_R1m_val,
    f_R4m=f_R4m_val,
    rebar_concept=rebar_concept,
    rebar_definition=rebar_definition,
    cover_top_mm=cover_top,
    cover_bottom_mm=cover_bot,
    dir_layer1_top=dir_top,
    dir_layer1_center=dir_cen,
    dir_layer1_bottom=dir_bot,
    simplified_rebar=simp_input,
    full_top_layers=full_top_list,
    full_bottom_layers=full_bot_list,
)

res = calculate_mrd(inp_obj, meta_obj)

with col_out:
    st.subheader("📐 Querschnitt & Ergebnisse (Results)")
    fig_cs = plot_ui_cross_section(res)
    st.pyplot(fig_cs, use_container_width=True)
    plt.close(fig_cs)

    # Ergebniskarten wie in Antonios Excel-UI (Negative Moments, Positive Moments, Equivalent Reinforcement)
    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.markdown("**Negative Moments (Zug oben)**")
        st.metric("M_Rd⁻", f"{res.neg_moment.m_rd_knm_m:.2f} kNm/m")
        d_top_val = res.d_top_mm if res.d_top_mm else inp_obj.h_mm
        st.caption(f"**A_s,Top:** `{res.as_top_total_cm2_m:.2f} cm²/m` | **d_Top:** `{d_top_val:.1f} mm`")
        st.write(f"ε_Top: `{res.neg_moment.eps_top_permille:.3f} ‰`")
        st.write(f"ε_Bottom: `{res.neg_moment.eps_bottom_permille:.3f} ‰`")
        st.write(f"x_Druckzone: `{res.neg_moment.x_mm:.1f} mm`")
    with r_col2:
        st.markdown("**Positive Moments (Zug unten)**")
        st.metric("M_Rd⁺", f"{res.pos_moment.m_rd_knm_m:.2f} kNm/m")
        d_bot_val = res.d_bottom_mm if res.d_bottom_mm else inp_obj.h_mm
        st.caption(f"**A_s,Bottom:** `{res.as_bottom_total_cm2_m:.2f} cm²/m` | **d_Bottom:** `{d_bot_val:.1f} mm`")
        st.write(f"ε_Top: `{res.pos_moment.eps_top_permille:.3f} ‰`")
        st.write(f"ε_Bottom: `{res.pos_moment.eps_bottom_permille:.3f} ‰`")
        st.write(f"x_Druckzone: `{res.pos_moment.x_mm:.1f} mm`")
    with r_col3:
        st.markdown("**Äquivalenzbewehrung (ohne SF)**")
        st.write(
            f"**A_s,Top,equiv:** `{res.as_top_equiv_cm2_m:.2f} cm²/m`" if res.as_top_equiv_cm2_m else "**A_s,Top,equiv:** `n.a.`"
        )
        st.write(
            f"**A_s,Bottom,equiv:** `{res.as_bottom_equiv_cm2_m:.2f} cm²/m`" if res.as_bottom_equiv_cm2_m else "**A_s,Bottom,equiv:** `n.a.`"
        )
        st.markdown("---")
        st.write(f"**f_ctd,L1:** `{res.f_ctd_L1:.3f} N/mm²`")
        st.write(f"**f_ctd,L2:** `{res.f_ctd_L2:.3f} N/mm²`")
        st.write(f"**β_L2:** `{res.beta_L2:.3f}`")

    # Beide Diagramme (Moments & Compression Height)
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_m = plot_ui_moments_chart(res)
        st.pyplot(fig_m, use_container_width=True)
        plt.close(fig_m)
    with ch2:
        fig_x = plot_ui_compression_height_chart(res)
        st.pyplot(fig_x, use_container_width=True)
        plt.close(fig_x)

    with st.expander("📋 Detaillierte Querschnittskräfte (Zug oben & Zug unten) einblenden", expanded=False):
        t_neg, t_pos = st.columns(2)
        with t_neg:
            st.markdown("**Zug oben (M_Rd⁻)**")
            rows_n = [
                {
                    "Anteil": r.name,
                    "z [mm]": round(r.z_mm, 1) if r.z_mm is not None else None,
                    "ε [‰]": round(r.eps_permille, 3) if r.eps_permille is not None else None,
                    "σ [N/mm²]": round(r.sigma_mpa, 2) if r.sigma_mpa is not None else None,
                    "F [kN/m]": round(r.f_kn_m, 2),
                }
                for r in res.neg_moment.rows if r.f_kn_m is not None
            ]
            st.dataframe(pd.DataFrame(rows_n), use_container_width=True, hide_index=True)
        with t_pos:
            st.markdown("**Zug unten (M_Rd⁺)**")
            rows_p = [
                {
                    "Anteil": r.name,
                    "z [mm]": round(r.z_mm, 1) if r.z_mm is not None else None,
                    "ε [‰]": round(r.eps_permille, 3) if r.eps_permille is not None else None,
                    "σ [N/mm²]": round(r.sigma_mpa, 2) if r.sigma_mpa is not None else None,
                    "F [kN/m]": round(r.f_kn_m, 2),
                }
                for r in res.pos_moment.rows if r.f_kn_m is not None
            ]
            st.dataframe(pd.DataFrame(rows_p), use_container_width=True, hide_index=True)

    # PDF-Generierung & Download-Button
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        generate_mrd_pdf(res, tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

    st.download_button(
        label=f"📄 Prüffähigen 7-Seiten-PDF-Nachweis herunterladen (Ersteller: {selected_user})",
        data=pdf_bytes,
        file_name=f"MRd_{project_name.replace(' ', '_')}_{emp['kuerzel']}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
