"""
app.py - Falconix Engineering GmbH
Streamlit Web-Applikation fuer das Tool "Biegewiderstand (MRd)" nach DAfStb-Richtlinie Stahlfaserbeton
inkl. Mitarbeiter-Login-Zuordnung, Simplified & Full Bewehrungsmodus (mit Stabbündeln)
und 7-seitigem PDF-Export.
"""
import datetime
import tempfile
import pandas as pd
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
    calculate_mrd,
)
from mrd_pdf import generate_mrd_pdf

st.set_page_config(page_title="Falconix Engineering - MRd Slab", layout="wide")

# ---------------------------------------------------------------------------
# 1. MITARBEITER-DATENBANK (Später an Supabase / Microsoft 365 Entra ID anbindbar)
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
# 2. SIDEBAR: LOGIN / BEARBEITER & PROJEKTDATEN (FÜR BERICHTERSTELLER)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 👤 Angemeldeter Bearbeiter")
    selected_user = st.selectbox("Mitarbeiter auswählen", list(EMPLOYEES.keys()), index=0)
    emp = EMPLOYEES[selected_user]
    user_email = st.text_input("E-Mail", value=emp["email"])
    user_phone = st.text_input("Telefon", value=emp["phone"])

    st.markdown("---")
    st.markdown("### 📁 Projekt & Kunde")
    project_name = st.text_input("Projekt / Bauvorhaben", value="BV Neubau Logistikhalle")
    position_name = st.text_input("Bauteil / Position", value="Bodenplatte Halle 1")
    client_name = st.text_input("Kunde / Auftraggeber", value="Musterbau GmbH")
    calc_date = st.date_input("Datum", value=datetime.date.today())

# ---------------------------------------------------------------------------
# 3. HAUPTBEREICH: EINGABEMASKE (LINKS) & ERGEBNISSE (RECHTS)
# ---------------------------------------------------------------------------
st.title("Moment Capacity for SFRC – Slabs (MRd)")
st.caption("Bemessung nach DAfStb-Richtlinie Stahlfaserbeton (06/2021) in Verbindung mit DIN EN 1992-1-1")

col_in, col_out = st.columns([1.15, 1.0], gap="large")

with col_in:
    st.subheader("1. Querschnitt & Baustoffe")
    c1, c2, c3 = st.columns(3)
    with c1:
        h_mm = st.number_input("Bauteilstärke h [mm]", min_value=70.0, max_value=1500.0, value=150.0, step=10.0)
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
    L1_val, L2_val = 0.9, 0.9
    f_R1m_val, f_R4m_val = 1.80, 1.70

    if fiber_concept == "Performance Class":
        fc1, fc2 = st.columns(2)
        with fc1:
            L1_val = st.selectbox("Leistungsklasse L1 (GZG / δ=0,5 mm)", PERFORMANCE_CLASS_VALUES[1:], index=3)
        with fc2:
            L2_val = st.selectbox("Leistungsklasse L2 (GZT / δ=3,5 mm)", PERFORMANCE_CLASS_VALUES[1:], index=2)
        if L2_val / L1_val < 0.7:
            st.warning("Hinweis: L2/L1 < 0,7 – nach DAfStb-Richtlinie R.3.6.3 (3) ist für GZT der Spannungsblock zu prüfen.")
    elif fiber_concept == "Residual Strength":
        rc1, rc2 = st.columns(2)
        with rc1:
            f_R1m_val = st.number_input("f_R1,m [N/mm²]", min_value=0.0, value=1.80, step=0.1)
        with rc2:
            f_R4m_val = st.number_input("f_R4,m [N/mm²]", min_value=0.0, value=1.70, step=0.1)

    st.subheader("3. Betonstahlbewehrung")
    rc_col1, rc_col2 = st.columns(2)
    with rc_col1:
        rebar_concept = st.selectbox(
            "Bewehrungskonzept",
            ["Fibers Only", "Top", "Center", "Bottom", "Top + Bottom"],
            index=0,
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
        cover_top = st.number_input("Betondeckung c_oben [mm]", min_value=15.0, max_value=150.0, value=35.0, step=5.0)
    with cov2:
        cover_bot = st.number_input("Betondeckung c_unten [mm]", min_value=15.0, max_value=150.0, value=35.0, step=5.0)

    simp_input = SimplifiedRebarInput()
    full_top_list = []
    full_bot_list = []

    if rebar_concept != "Fibers Only":
        if rebar_definition == "Simplified":
            st.markdown("##### Vereinfachte Bewehrungseingabe (Kreuznetz Lage 1 & 2)")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**Oben (Top)**")
                t_d = st.selectbox("Ø Oben [mm]", REBAR_DIAMETERS, index=2, key="s_td")
                t_s = st.number_input("Abstand s Oben [mm]", min_value=20.0, value=150.0, step=25.0, key="s_ts")
                dir_top = st.selectbox("Richtung 1. Lage Oben", ["r", "s"], index=0)
            with sc2:
                st.markdown("**Mitte (Center)**")
                c_d = st.selectbox("Ø Mitte [mm]", REBAR_DIAMETERS, index=2, key="s_cd")
                c_s = st.number_input("Abstand s Mitte [mm]", min_value=20.0, value=150.0, step=25.0, key="s_cs")
                dir_cen = st.selectbox("Richtung 1. Lage Mitte", ["r", "s"], index=0)
            with sc3:
                st.markdown("**Unten (Bottom)**")
                b_d = st.selectbox("Ø Unten [mm]", REBAR_DIAMETERS, index=2, key="s_bd")
                b_s = st.number_input("Abstand s Unten [mm]", min_value=20.0, value=150.0, step=25.0, key="s_bs")
                dir_bot = st.selectbox("Richtung 1. Lage Unten", ["r", "s"], index=0)

            simp_input = SimplifiedRebarInput(
                top_diam=float(t_d), top_spacing=float(t_s),
                center_diam=float(c_d), center_spacing=float(c_s),
                bottom_diam=float(b_d), bottom_spacing=float(b_s),
            )
        else:
            st.markdown("##### Volle Bewehrungsdefinition (inkl. Mischbewehrung Ø1+Ø2 & Stabbündel)")
            default_rows = [
                {"Lage": "1 (r)", "Richtung": "r", "Diam_1": 8, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "2 (s)", "Richtung": "s", "Diam_1": 8, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "3 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "4 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "5 (r)", "Richtung": "r", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
                {"Lage": "6 (s)", "Richtung": "s", "Diam_1": 0, "Diam_2": 0, "Abstand_s": 150, "Stabbuendel": False},
            ]
            if rebar_concept in ("Top", "Top + Bottom"):
                st.markdown("**Obere Bewehrungslagen (1 bis 6)**")
                df_top = st.data_editor(pd.DataFrame(default_rows), key="ed_top", use_container_width=True, hide_index=True)
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
                df_bot = st.data_editor(pd.DataFrame(default_rows), key="ed_bot", use_container_width=True, hide_index=True)
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
# 4. BERECHNUNG AUSFÜHREN & ERGEBNISSE ANZEIGEN
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
    st.subheader("📊 Ergebnisse Biegewiderstand")
    m1, m2 = st.columns(2)
    m1.metric("M_Rd⁻ (Zug oben)", f"{res.neg_moment.m_rd_knm_m:.2f} kNm/m", f"x = {res.neg_moment.x_mm:.1f} mm")
    m2.metric("M_Rd⁺ (Zug unten)", f"{res.pos_moment.m_rd_knm_m:.2f} kNm/m", f"x = {res.pos_moment.x_mm:.1f} mm")

    st.markdown("##### Stahlfaser-Bemessungswerte & Querschnittskräfte (Zug unten)")
    st.write(
        f"**f_ctd,L1** = `{res.f_ctd_L1:.3f} N/mm²` | "
        f"**f_ctd,L2** = `{res.f_ctd_L2:.3f} N/mm²` | "
        f"**f_ctd,u** = `{res.f_ctd_u:.3f} N/mm²` | "
        f"**β_L2** = `{res.beta_L2:.3f}`"
    )

    rows_disp = []
    for r in res.pos_moment.rows:
        if r.f_kn_m is not None:
            rows_disp.append({
                "Anteil": r.name,
                "z [mm]": round(r.z_mm, 1) if r.z_mm is not None else None,
                "ε [‰]": round(r.eps_permille, 3) if r.eps_permille is not None else None,
                "σ [N/mm²]": round(r.sigma_mpa, 2) if r.sigma_mpa is not None else None,
                "F [kN/m]": round(r.f_kn_m, 2),
            })
    st.dataframe(pd.DataFrame(rows_disp), use_container_width=True, hide_index=True)

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
