"""
mrd_pdf.py - Falconix Engineering GmbH
Generiert den 7-seitigen prüffähigen PDF-Nachweis "Biegewiderstand" (MRd)
im exakten Falconix-Originaldesign mit WeasyPrint.
"""
from __future__ import annotations
import base64
import io
import os
import math
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import weasyprint
from mrd_core import MRdResult, ActiveSteelLayer


def _fig_to_base64_svg(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    svg_bytes = buf.getvalue()
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _get_logo_data_uri(logo_path: str = "extracted_Image12.jpg") -> str:
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    return ""


def make_concrete_chart(res: MRdResult) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=150)
    eps = np.linspace(0.0, 3.5, 300)
    f_cd = res.f_cd
    sig = np.where(eps <= 2.0, f_cd * (1.0 - (1.0 - eps / 2.0) ** 2), f_cd)
    ax.plot(eps, sig, color="#3b6291", linewidth=2.0)
    ax.set_title("Beton", fontsize=12, color="#444444", pad=10)
    ax.set_xlabel("Stauchung [‰]", fontsize=9, color="#444444")
    ax.set_ylabel("Druckspannung [N/mm²]", fontsize=9, color="#444444")
    ax.set_xlim(0.0, 3.5)
    y_max = max(16.0, math.ceil(f_cd / 2.0) * 2.0 + 2.0)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#efefef", linewidth=0.4)
    ax.tick_params(labelsize=8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    return _fig_to_base64_svg(fig)


def make_fiber_chart(res: MRdResult) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=150)
    eps = [0.0, 0.1, 3.5, 25.0]
    sig = [0.0, res.f_ctd_L1, res.f_ctd_L1, res.f_ctd_L2]
    ax.plot(eps, sig, color="#3b6291", linewidth=2.0)
    ax.set_title("Stahlfaserbeton", fontsize=12, color="#444444", pad=10)
    ax.set_xlabel("Dehnung [‰]", fontsize=9, color="#444444")
    ax.set_ylabel("Spannung [N/mm²]", fontsize=9, color="#444444")
    ax.set_xlim(0.0, 25.0)
    y_max = max(0.5, round(max(sig) * 1.2 + 0.05, 1)) if max(sig) > 0 else 0.5
    ax.set_ylim(0.0, y_max)
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="#efefef", linewidth=0.4)
    ax.tick_params(labelsize=8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    return _fig_to_base64_svg(fig)


def make_beta_l2_mini_chart(res: MRdResult) -> str:
    fig, ax = plt.subplots(figsize=(2.4, 1.35), dpi=150)
    x_vals = [0.7, 1.0, 1.5]
    y_vals = [0.25, 0.35, 0.44]
    ax.plot(x_vals, y_vals, color="#222222", linewidth=1.5)
    ax.vlines([0.7, 1.0, 1.5], 0.0, y_vals, colors="#777777", linestyles="dashed", linewidth=0.8)
    ax.hlines(y_vals, 0.5, x_vals, colors="#777777", linestyles="dashed", linewidth=0.8)
    if res.f_cflk_L1 > 0:
        r = res.f_cflk_L2 / res.f_cflk_L1
        if 0.7 <= r <= 1.5:
            ax.plot([r], [res.beta_L2], marker="o", markersize=4, color="#c0392b")
    ax.set_xlim(0.5, 1.6)
    ax.set_ylim(0.0, 0.52)
    ax.set_xticks([0.7, 1.0, 1.5])
    ax.set_yticks([0.25, 0.35, 0.44])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("L2/L1", fontsize=7)
    ax.set_title("βL2", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _fig_to_base64_svg(fig)


def make_moment_strain_charts(res: MRdResult) -> Tuple[str, str]:
    fig1, ax1 = plt.subplots(figsize=(7.2, 3.1), dpi=150)
    ax1.plot(res.neg_moment.curve_eps_permille, res.neg_moment.curve_m_knm_m, color="#d37222", linewidth=2.0)
    ax1.set_title("Moment - Dehnung (Zug oben)", fontsize=12, color="#444444", pad=10)
    ax1.set_xlabel("Dehnung [‰]", fontsize=9, color="#444444")
    ax1.set_ylabel("Biegemoment [kNm/m]", fontsize=9, color="#444444")
    ax1.set_xlim(0.0, 25.0)
    m_min = min(res.neg_moment.curve_m_knm_m)
    ax1.set_ylim(0.0, min(-5.0, math.floor(m_min * 1.15)))
    ax1.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", color="#efefef", linewidth=0.4)
    ax1.tick_params(labelsize=8, colors="#444444")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#cccccc")

    fig2, ax2 = plt.subplots(figsize=(7.2, 3.1), dpi=150)
    ax2.plot(res.pos_moment.curve_eps_permille, res.pos_moment.curve_m_knm_m, color="#d37222", linewidth=2.0)
    ax2.set_title("Moment - Dehnung (Zug unten)", fontsize=12, color="#444444", pad=12)
    ax2.set_xlabel("Dehnung [‰]", fontsize=9, color="#444444")
    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()
    ax2.set_ylabel("Biegemoment [kNm/m]", fontsize=9, color="#444444")
    ax2.set_xlim(0.0, 25.0)
    m_max = max(res.pos_moment.curve_m_knm_m)
    ax2.set_ylim(max(5.0, math.ceil(m_max * 1.15)), 0.0)
    ax2.grid(True, which="major", color="#d9d9d9", linewidth=0.7)
    ax2.minorticks_on()
    ax2.grid(True, which="minor", color="#efefef", linewidth=0.4)
    ax2.tick_params(labelsize=8, colors="#444444")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#cccccc")

    return _fig_to_base64_svg(fig1), _fig_to_base64_svg(fig2)


def fmt_num(val: Optional[float], decimals: int = 3, na_str: str = "") -> str:
    if val is None:
        return na_str
    return f"{val:.{decimals}f}".replace(".", ",")


def _render_header(title: str, logo_uri: str) -> str:
    """Erzeugt eine 2-spaltige Kopftabelle: Titel linksbündig, Falconix-Logo fest rechtsbündig."""
    return f"""
    <table class="header-table">
      <tr>
        <td class="header-title-cell">{title}</td>
        <td class="header-logo-cell"><img class="header-logo" src="{logo_uri}" alt="FALCONIX ENGINEERING"></td>
      </tr>
    </table>
    """


def generate_mrd_pdf(res: MRdResult, output_pdf_path: str, logo_path: str = "extracted_Image12.jpg") -> str:
    logo_uri = _get_logo_data_uri(logo_path)
    chart_concrete = make_concrete_chart(res)
    chart_fiber = make_fiber_chart(res)
    chart_beta = make_beta_l2_mini_chart(res)
    chart_m_neg, chart_m_pos = make_moment_strain_charts(res)

    inp = res.inputs
    meta = res.metadata

    lk_str = f"L{fmt_num(inp.L1, 1)}/{fmt_num(inp.L2, 1)}" if inp.fiber_concept == "Performance Class" else "n.a."
    sf_type_str = inp.fiber_name if inp.fiber_concept == "Residual Strength" else "n.a."
    dosage_str = f"{fmt_num(inp.dosage_kg_m3, 1)} kg/m³" if (inp.fiber_concept == "Residual Strength" and inp.dosage_kg_m3) else "n.a."
    fr1_str = fmt_num(inp.f_R1m, 2) if inp.fiber_concept == "Residual Strength" else "n.a."
    fr4_str = fmt_num(inp.f_R4m, 2) if inp.fiber_concept == "Residual Strength" else "n.a."

    has_top_rebar = any(k in res.steel_layers for k in ("t1", "t2", "t3"))
    has_bot_rebar = any(k in res.steel_layers for k in ("b1", "b2", "b3"))
    cov_top_str = fmt_num(inp.cover_top_mm, 0) if has_top_rebar else "n.a."
    cov_bot_str = fmt_num(inp.cover_bottom_mm, 0) if has_bot_rebar else "n.a."

    rebar_rows_p2 = []
    for group_label, keys in [("Oben", ["t₁", "t₂", "t₃"]), ("Unten", ["b₃", "b₂", "b₁"])]:
        for idx, k_disp in enumerate(keys):
            k_internal = k_disp.replace("₁", "1").replace("₂", "2").replace("₃", "3")
            lay: Optional[ActiveSteelLayer] = res.steel_layers.get(k_internal)
            rebar_rows_p2.append({
                "group": group_label if idx == 0 else "",
                "label": k_disp,
                "d1": fmt_num(lay.diam_1, 0) if (lay and lay.diam_1 > 0) else "",
                "d2": fmt_num(lay.diam_2, 0) if (lay and lay.diam_2 > 0) else "",
                "s": fmt_num(lay.spacing, 0) if (lay and lay.as_cm2_m > 0) else "",
                "as": fmt_num(lay.as_cm2_m, 2) if (lay and lay.as_cm2_m > 0) else "",
                "c": fmt_num(lay.c_mm, 0) if (lay and lay.as_cm2_m > 0) else "",
                "z": fmt_num(lay.z_from_mid_mm, 1) if (lay and lay.as_cm2_m > 0) else "",
            })

    p5_rows = []
    label_map = ["Beton (c)", "SF (ct)", "Stahl (t₁)", "Stahl (t₂)", "Stahl (t₃)", "Stahl (b₃)", "Stahl (b₂)", "Stahl (b₁)"]
    for i in range(8):
        rn = res.neg_moment.rows[i]
        rp = res.pos_moment.rows[i]
        p5_rows.append({
            "name": label_map[i],
            "n_z": fmt_num(rn.z_mm, 1),
            "n_e": fmt_num(rn.eps_permille, 3) if i > 1 else "",
            "n_s": fmt_num(rn.sigma_mpa, 2) if i > 1 else "",
            "n_f": fmt_num(rn.f_kn_m, 2),
            "p_z": fmt_num(rp.z_mm, 1),
            "p_e": fmt_num(rp.eps_permille, 3) if i > 1 else "",
            "p_s": fmt_num(rp.sigma_mpa, 2) if i > 1 else "",
            "p_f": fmt_num(rp.f_kn_m, 2),
        })

    bew_konzept_de = "Reine Stahlfaserbewehrung" if not res.has_rebar else (
        "Kombinierte Bewehrung (Stahlfaserbeton + Betonstahl)" if res.has_fibers else "Reine Betonstahlbewehrung"
    )
    t1_lay = res.steel_layers.get("t1")
    b1_lay = res.steel_layers.get("b1")

    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 16mm 14mm 16mm;
    @bottom-right {{
      content: "Seite " counter(page) " von " counter(pages);
      font-size: 8pt;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #666;
    }}
    @bottom-left {{
      content: "Falconix Engineering GmbH — Biegewiderstand (MRd)";
      font-size: 8pt;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: #666;
    }}
  }}
  body {{
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
    font-size: 9.5pt;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
  }}
  .page {{
    page-break-after: always;
    position: relative;
    height: 267mm;
    box-sizing: border-box;
  }}
  .page:last-child {{
    page-break-after: auto;
  }}
  table.header-table {{
    width: 100%;
    border-collapse: collapse;
    border-bottom: 2.2px solid #222;
    margin-bottom: 14px;
  }}
  table.header-table td.header-title-cell {{
    font-size: 16pt;
    font-weight: 700;
    text-align: left;
    vertical-align: bottom;
    padding-bottom: 6px;
  }}
  table.header-table td.header-logo-cell {{
    text-align: right;
    vertical-align: bottom;
    padding-bottom: 6px;
  }}
  .header-logo {{
    height: 36px;
  }}
  .section-hdr {{
    font-weight: 700;
    font-size: 10pt;
    border-top: 1.5px solid #222;
    border-bottom: 1px solid #555;
    padding: 3px 4px;
    margin-bottom: 6px;
    background-color: #fafafa;
  }}
  table.kv-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
  }}
  table.kv-table td {{
    padding: 2.5px 4px;
    vertical-align: middle;
  }}
  .val-cell {{
    text-align: right;
    font-weight: 500;
  }}
  .unit-cell {{
    color: #333;
    padding-left: 4px;
    width: 48px;
  }}
  .formula-note {{
    color: #333;
    font-style: italic;
    font-size: 8.5pt;
  }}
  table.grid-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
  }}
  table.grid-table th {{
    border-top: 1.2px solid #222;
    border-bottom: 1px solid #222;
    padding: 3px 4px;
    text-align: center;
    font-weight: 600;
  }}
  table.grid-table td {{
    padding: 3px 4px;
    border-bottom: 0.5px solid #ddd;
    text-align: right;
  }}
  table.grid-table td.left {{
    text-align: left;
  }}
  table.grid-table td.center {{
    text-align: center;
  }}
  .two-col {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
  }}
  .col-half {{
    width: 48.5%;
  }}
  .chart-container {{
    text-align: center;
    margin-top: 12px;
    border: 0.8px solid #d0d0d0;
    padding: 8px;
  }}
  .mrd-highlight {{
    border-top: 1.5px solid #222;
    border-bottom: 2px double #222;
    font-weight: 700;
    font-size: 10pt;
    padding: 4px 2px;
    margin-top: 8px;
    display: flex;
    justify-content: space-between;
  }}
</style>
</head>
<body>

<!-- ==================== SEITE 1: DECKBLATT ==================== -->
<div class="page">
  {_render_header("Biegewiderstand", logo_uri)}

  <div class="two-col" style="margin-top: 25px;">
    <div class="col-half">
      <div class="section-hdr">Projekt</div>
      <div style="height: 85px; padding: 4px; font-size: 9.5pt;">
        <strong>{meta.project_name}</strong><br/>
        {meta.position_name}
      </div>
    </div>
    <div class="col-half">
      <div class="section-hdr">Kunde</div>
      <div style="height: 85px; padding: 4px; font-size: 9.5pt;">
        <strong>{meta.client_name}</strong>
      </div>
    </div>
  </div>

  <div class="section-hdr" style="margin-top: 15px;">Ersteller</div>
  <div class="two-col" style="padding: 6px 4px; line-height: 1.55;">
    <div class="col-half">
      <strong>{meta.user_name}</strong> ({meta.user_kuerzel})<br/>
      {meta.company}<br/>
      {meta.street}<br/>
      {meta.location}
    </div>
    <div class="col-half">
      <table class="kv-table">
        <tr><td>Datum</td><td class="val-cell">{meta.date_str}</td></tr>
        <tr><td>Version</td><td class="val-cell">{meta.version}</td></tr>
        <tr><td colspan="2" style="padding-top: 8px;">{meta.phone}</td></tr>
        <tr><td colspan="2">{meta.email}</td></tr>
      </table>
    </div>
  </div>
</div>

<!-- ==================== SEITE 2: EINGANGSWERTE ==================== -->
<div class="page">
  {_render_header("Eingangswerte", logo_uri)}

  <div class="two-col">
    <div class="col-half">
      <div class="section-hdr">Querschnitt</div>
      <table class="kv-table">
        <tr><td>Bauteil</td><td class="val-cell">{"Platte" if inp.element_type == "Slab" else "Wand"}</td><td class="unit-cell"></td></tr>
        <tr><td>Bauteilstärke</td><td class="val-cell">{fmt_num(inp.h_mm, 0)}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Druckfestigkeitsklasse</td><td class="val-cell">{inp.concrete_class}</td><td class="unit-cell"></td></tr>
        <tr><td>Betonstahl</td><td class="val-cell">{inp.steel_class}</td><td class="unit-cell"></td></tr>
        <tr><td>Leistungsklasse</td><td class="val-cell">{lk_str}</td><td class="unit-cell"></td></tr>
        <tr><td>Stahlfasern</td><td class="val-cell">{sf_type_str}</td><td class="unit-cell"></td></tr>
        <tr><td>Dosierung</td><td class="val-cell">{dosage_str}</td><td class="unit-cell"></td></tr>
      </table>
    </div>
    <div class="col-half">
      <div class="section-hdr">Nachrissbiegezugfestigkeit</div>
      <table class="kv-table">
        <tr><td>f<sub>R1,m</sub></td><td class="val-cell">{fr1_str}</td><td class="unit-cell">N/mm²</td></tr>
        <tr><td>f<sub>R4,m</sub></td><td class="val-cell">{fr4_str}</td><td class="unit-cell">N/mm²</td></tr>
      </table>
      <div class="section-hdr" style="margin-top: 16px;">Betondeckung</div>
      <table class="kv-table">
        <tr><td>Oben</td><td class="val-cell">{cov_top_str}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Unten</td><td class="val-cell">{cov_bot_str}</td><td class="unit-cell">mm</td></tr>
      </table>
    </div>
  </div>

  <div class="section-hdr" style="margin-top: 16px;">Bewehrung</div>
  <div class="two-col">
    <div style="width: 66%;">
      <table class="grid-table">
        <thead>
          <tr>
            <th colspan="2">Lage</th>
            <th>Ø<sub>1</sub><br/>[mm]</th>
            <th>Ø<sub>2</sub><br/>[mm]</th>
            <th>s<br/>[mm]</th>
            <th>A<sub>s</sub><br/>[cm²/m]</th>
            <th>c<br/>[mm]</th>
            <th>z<br/>[mm]</th>
          </tr>
        </thead>
        <tbody>
          {"".join(f"<tr><td class='left'><strong>{r['group']}</strong></td><td class='center'>{r['label']}</td><td>{r['d1']}</td><td>{r['d2']}</td><td>{r['s']}</td><td>{r['as']}</td><td>{r['c']}</td><td>{r['z']}</td></tr>" for r in rebar_rows_p2)}
        </tbody>
      </table>
    </div>
    <div style="width: 31%; font-size: 8.5pt; padding-top: 10px;">
      <div style="border: 1px solid #aaa; padding: 8px; text-align: center; background: #fcfcfc;">
        <strong>A<sub>s</sub> = (1 / s) · (π / 4) · (Ø<sub>1</sub>² + Ø<sub>2</sub>²)</strong>
        <hr style="border: 0; border-top: 1px dashed #ccc; margin: 8px 0;">
        <div style="font-size: 8pt; color: #444; text-align: left;">
          • <strong>z</strong> bezogen auf Bauteilmitte (h/2)<br/>
          • <strong>+z</strong>: oberhalb Mitte (t<sub>1</sub>..t<sub>3</sub>)<br/>
          • <strong>-z</strong>: unterhalb Mitte (b<sub>1</sub>..b<sub>3</sub>)<br/>
          • Berechnungsrichtung: <strong>{inp.tensile_direction}</strong>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ==================== SEITE 3: BAUSTOFFE ==================== -->
<div class="page">
  {_render_header("Baustoffe", logo_uri)}

  <div class="two-col">
    <div class="col-half">
      <div class="section-hdr">Beton</div>
      <table class="kv-table">
        <tr><td>f<sub>ck</sub></td><td class="val-cell">{fmt_num(res.f_ck, 3)}</td><td class="unit-cell">N/mm²</td></tr>
        <tr><td>α<sub>cc</sub></td><td class="val-cell">{fmt_num(inp.alpha_cc, 2)}</td><td class="unit-cell"></td></tr>
        <tr><td>γ<sub>c</sub></td><td class="val-cell">{fmt_num(inp.gamma_c, 2)}</td><td class="unit-cell"></td></tr>
        <tr><td><strong>f<sub>cd</sub></strong></td><td class="val-cell"><strong>{fmt_num(res.f_cd, 3)}</strong></td><td class="unit-cell">N/mm²</td></tr>
      </table>
    </div>
    <div class="col-half">
      <div class="section-hdr">Betonstahl</div>
      <table class="kv-table">
        <tr><td>f<sub>yk</sub></td><td class="val-cell">{fmt_num(res.f_yk, 3, "n.a.")}</td><td class="unit-cell">N/mm²</td></tr>
        <tr><td>γ<sub>s</sub></td><td class="val-cell">{fmt_num(inp.gamma_s, 2) if res.has_rebar else "n.a."}</td><td class="unit-cell"></td></tr>
        <tr><td>f<sub>yd</sub></td><td class="val-cell">{fmt_num(res.f_yd, 3, "n.a.")}</td><td class="unit-cell">N/mm²</td></tr>
        <tr><td>E<sub>s</sub></td><td class="val-cell">{fmt_num(res.E_s, 3, "n.a.")}</td><td class="unit-cell">kN/mm²</td></tr>
      </table>
    </div>
  </div>

  <div class="chart-container" style="margin-top: 24px;">
    <img src="{chart_concrete}" style="width: 100%; max-height: 230px;">
  </div>
</div>

<!-- ==================== SEITE 4: STAHLFASERN ==================== -->
<div class="page">
  {_render_header("Stahlfasern", logo_uri)}

  <div class="two-col">
    <div style="width: 64%;">
      <table class="kv-table">
        <tr><td>f<sub>R1,m</sub></td><td class="val-cell">{fmt_num(inp.f_R1m, 3) if inp.fiber_concept == "Residual Strength" else "n.a."}</td><td class="unit-cell">N/mm²</td><td class="formula-note"></td></tr>
        <tr><td>f<sub>R4,m</sub></td><td class="val-cell">{fmt_num(inp.f_R4m, 3) if inp.fiber_concept == "Residual Strength" else "n.a."}</td><td class="unit-cell">N/mm²</td><td class="formula-note"></td></tr>
        <tr><td>f<sup>f</sup><sub>cflk,L1</sub></td><td class="val-cell">{fmt_num(res.f_cflk_L1, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">L1 ≡ f<sup>f</sup><sub>cflk,L1</sub> = 0,51 · f<sub>R1,m</sub></td></tr>
        <tr><td>f<sup>f</sup><sub>cflk,L2</sub></td><td class="val-cell">{fmt_num(res.f_cflk_L2, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">L2 ≡ f<sup>f</sup><sub>cflk,L2</sub> = 0,51 · f<sub>R4,m</sub></td></tr>
        <tr><td colspan="4" style="height: 4px;"></td></tr>
        <tr><td>β<sub>L1</sub></td><td class="val-cell">{fmt_num(res.beta_L1, 3)}</td><td class="unit-cell"></td><td class="formula-note">β<sub>L1</sub> = 0,40 ⇔ L2/L1 ≥ 0,7</td></tr>
        <tr><td>β<sub>L2</sub></td><td class="val-cell">{fmt_num(res.beta_L2, 3)}</td><td class="unit-cell"></td><td class="formula-note">gemäß DAfStb-RL Anhang P (Bild P.1)</td></tr>
        <tr><td>β<sub>u</sub></td><td class="val-cell">{fmt_num(res.beta_u, 3)}</td><td class="unit-cell"></td><td class="formula-note">β<sub>u</sub> = 0,37</td></tr>
        <tr><td colspan="4" style="height: 4px;"></td></tr>
        <tr><td>f<sup>f</sup><sub>ct0,L1</sub></td><td class="val-cell">{fmt_num(res.f_ct0_L1, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ct0,L1</sub> = f<sup>f</sup><sub>cflk,L1</sub> · β<sub>L1</sub></td></tr>
        <tr><td>f<sup>f</sup><sub>ct0,L2</sub></td><td class="val-cell">{fmt_num(res.f_ct0_L2, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ct0,L2</sub> = f<sup>f</sup><sub>cflk,L2</sub> · β<sub>L2</sub></td></tr>
        <tr><td>f<sup>f</sup><sub>ct0,u</sub></td><td class="val-cell">{fmt_num(res.f_ct0_u, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ct0,u</sub> = f<sup>f</sup><sub>cflk,L2</sub> · β<sub>u</sub></td></tr>
        <tr><td colspan="4" style="height: 4px;"></td></tr>
        <tr><td>κ<sup>f</sup><sub>F</sub></td><td class="val-cell">{fmt_num(inp.kappa_F, 3)}</td><td class="unit-cell"></td><td class="formula-note">1,0 für ebene, liegend hergestellte Platten</td></tr>
        <tr><td>κ<sup>f</sup><sub>G</sub></td><td class="val-cell">{fmt_num(inp.kappa_G, 3)}</td><td class="unit-cell"></td><td class="formula-note">κ<sup>f</sup><sub>G</sub> = 1,0 + A<sup>f</sup><sub>ct</sub> · 0,5 ≤ 1,70</td></tr>
        <tr><td colspan="4" style="height: 4px;"></td></tr>
        <tr><td>f<sup>f</sup><sub>ctR,L1</sub></td><td class="val-cell">{fmt_num(res.f_ctR_L1, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctR,L1</sub> = κ<sup>f</sup><sub>F</sub> · κ<sup>f</sup><sub>G</sub> · f<sup>f</sup><sub>ct0,L1</sub></td></tr>
        <tr><td>f<sup>f</sup><sub>ctR,L2</sub></td><td class="val-cell">{fmt_num(res.f_ctR_L2, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctR,L2</sub> = κ<sup>f</sup><sub>F</sub> · κ<sup>f</sup><sub>G</sub> · f<sup>f</sup><sub>ct0,L2</sub></td></tr>
        <tr><td>f<sup>f</sup><sub>ctR,u</sub></td><td class="val-cell">{fmt_num(res.f_ctR_u, 3)}</td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctR,u</sub> = κ<sup>f</sup><sub>F</sub> · κ<sup>f</sup><sub>G</sub> · f<sup>f</sup><sub>ct0,u</sub></td></tr>
        <tr><td colspan="4" style="height: 4px;"></td></tr>
        <tr><td>α<sup>f</sup><sub>ct</sub></td><td class="val-cell">{fmt_num(inp.alpha_ct_f, 3)}</td><td class="unit-cell"></td><td class="formula-note"></td></tr>
        <tr><td>γ<sup>f</sup><sub>ct</sub></td><td class="val-cell">{fmt_num(inp.gamma_ct_f, 3)}</td><td class="unit-cell"></td><td class="formula-note"></td></tr>
        <tr><td><strong>f<sup>f</sup><sub>ctd,L1</sub></strong></td><td class="val-cell"><strong>{fmt_num(res.f_ctd_L1, 3)}</strong></td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctd,L1</sub> = α<sup>f</sup><sub>ct</sub> · f<sup>f</sup><sub>ctR,L1</sub> / γ<sup>f</sup><sub>ct</sub></td></tr>
        <tr><td><strong>f<sup>f</sup><sub>ctd,L2</sub></strong></td><td class="val-cell"><strong>{fmt_num(res.f_ctd_L2, 3)}</strong></td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctd,L2</sub> = α<sup>f</sup><sub>ct</sub> · f<sup>f</sup><sub>ctR,L2</sub> / γ<sup>f</sup><sub>ct</sub></td></tr>
        <tr><td><strong>f<sup>f</sup><sub>ctd,u</sub></strong></td><td class="val-cell"><strong>{fmt_num(res.f_ctd_u, 3)}</strong></td><td class="unit-cell">N/mm²</td><td class="formula-note">f<sup>f</sup><sub>ctd,u</sub> = α<sup>f</sup><sub>ct</sub> · f<sup>f</sup><sub>ctR,u</sub> / γ<sup>f</sup><sub>ct</sub></td></tr>
      </table>
    </div>
    <div style="width: 33%;">
      <div style="text-align: center; margin-bottom: 18px;">
        <img src="{chart_beta}" style="width: 100%;">
      </div>
      <table class="grid-table" style="margin-top: 35px;">
        <thead>
          <tr><th>ε<sup>f</sup><sub>ct</sub><br/>[‰]</th><th>σ<sup>f</sup><sub>ct</sub><br/>[N/mm²]</th></tr>
        </thead>
        <tbody>
          <tr><td>0,000</td><td>0,000</td></tr>
          <tr><td>0,100</td><td>{fmt_num(res.f_ctd_L1, 3)}</td></tr>
          <tr><td>3,500</td><td>{fmt_num(res.f_ctd_L1, 3)}</td></tr>
          <tr><td>25,000</td><td>{fmt_num(res.f_ctd_L2, 3)}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="chart-container" style="margin-top: 14px;">
    <img src="{chart_fiber}" style="width: 100%; max-height: 200px;">
  </div>
</div>

<!-- ==================== SEITE 5: BIEGEWIDERSTAND ERGEBNISSE ==================== -->
<div class="page">
  {_render_header("Biegewiderstand", logo_uri)}

  <div class="two-col">
    <div class="col-half">
      <div class="section-hdr">Querschnitt</div>
      <table class="kv-table">
        <tr><td>h</td><td class="val-cell">{fmt_num(inp.h_mm, 0)}</td><td class="unit-cell">mm</td></tr>
      </table>
    </div>
    <div class="col-half">
      <div class="section-hdr">Statische Höhen</div>
      <table class="kv-table">
        <tr><td>d<sub>Oben</sub></td><td class="val-cell">{fmt_num(res.d_top_mm, 1, "n.a.")}</td><td class="unit-cell">mm</td></tr>
        <tr><td>d<sub>Unten</sub></td><td class="val-cell">{fmt_num(res.d_bottom_mm, 1, "n.a.")}</td><td class="unit-cell">mm</td></tr>
      </table>
    </div>
  </div>

  <div class="section-hdr" style="margin-top: 10px;">Ergebnisse</div>
  <div class="two-col">
    <!-- ZUG OBEN -->
    <div class="col-half">
      <div style="font-weight: 700; border-bottom: 1px solid #333; padding-bottom: 3px; margin-bottom: 6px;">Zug oben</div>
      <table class="kv-table" style="margin-bottom: 8px;">
        <tr><td>ε<sub>s,oben</sub></td><td class="val-cell">{fmt_num(res.neg_moment.eps_top_permille, 3)}</td><td class="unit-cell">‰</td></tr>
        <tr><td>ε<sub>s,unten</sub></td><td class="val-cell">{fmt_num(res.neg_moment.eps_bottom_permille, 3)}</td><td class="unit-cell">‰</td></tr>
        <tr><td>x<sub>Druckzone</sub></td><td class="val-cell">{fmt_num(res.neg_moment.x_mm, 1)}</td><td class="unit-cell">mm</td></tr>
      </table>

      <table class="grid-table">
        <thead>
          <tr><th class="left">Anteil</th><th>z<br/>[mm]</th><th>ε<br/>[‰]</th><th>σ<br/>[N/mm²]</th><th>F<br/>[kN/m]</th></tr>
        </thead>
        <tbody>
          {"".join(f"<tr><td class='left'>{r['name']}</td><td>{r['n_z']}</td><td>{r['n_e']}</td><td>{r['n_s']}</td><td>{r['n_f']}</td></tr>" for r in p5_rows)}
        </tbody>
      </table>
      <div class="mrd-highlight">
        <span>M<sub>Rd</sub><sup>-</sup></span>
        <span>{fmt_num(res.neg_moment.m_rd_knm_m, 2)} kNm/m</span>
      </div>
    </div>

    <!-- ZUG UNTEN -->
    <div class="col-half">
      <div style="font-weight: 700; border-bottom: 1px solid #333; padding-bottom: 3px; margin-bottom: 6px;">Zug unten</div>
      <table class="kv-table" style="margin-bottom: 8px;">
        <tr><td>ε<sub>s,oben</sub></td><td class="val-cell">{fmt_num(res.pos_moment.eps_top_permille, 3)}</td><td class="unit-cell">‰</td></tr>
        <tr><td>ε<sub>s,unten</sub></td><td class="val-cell">{fmt_num(res.pos_moment.eps_bottom_permille, 3)}</td><td class="unit-cell">‰</td></tr>
        <tr><td>x<sub>Druckzone</sub></td><td class="val-cell">{fmt_num(res.pos_moment.x_mm, 1)}</td><td class="unit-cell">mm</td></tr>
      </table>

      <table class="grid-table">
        <thead>
          <tr><th class="left">Anteil</th><th>z<br/>[mm]</th><th>ε<br/>[‰]</th><th>σ<br/>[N/mm²]</th><th>F<br/>[kN/m]</th></tr>
        </thead>
        <tbody>
          {"".join(f"<tr><td class='left'>{r['name']}</td><td>{r['p_z']}</td><td>{r['p_e']}</td><td>{r['p_s']}</td><td>{r['p_f']}</td></tr>" for r in p5_rows)}
        </tbody>
      </table>
      <div class="mrd-highlight">
        <span>M<sub>Rd</sub><sup>+</sup></span>
        <span>{fmt_num(res.pos_moment.m_rd_knm_m, 2)} kNm/m</span>
      </div>
    </div>
  </div>
</div>

<!-- ==================== SEITE 6: MOMENT - DEHNUNGSKURVEN ==================== -->
<div class="page">
  {_render_header("Moment - Dehnungskurven", logo_uri)}

  <div class="chart-container" style="margin-top: 10px;">
    <img src="{chart_m_neg}" style="width: 100%; max-height: 290px;">
  </div>

  <div class="chart-container" style="margin-top: 18px;">
    <img src="{chart_m_pos}" style="width: 100%; max-height: 290px;">
  </div>
</div>

<!-- ==================== SEITE 7: ZUSAMMENFASSUNG ==================== -->
<div class="page">
  {_render_header("Zusammenfassung:", logo_uri)}

  <p style="font-size: 8.8pt; line-height: 1.45; margin-bottom: 14px;">
    Diese Berechnung dient als Einzelnachweis innerhalb einer Gesamtstatik.<br/>
    Sie bezieht sich auf die Biegetragfähigkeit eines einzelnen Bauteils und darf daher nur im Zusammenhang mit
    weiteren zugehörigen Nachweisen im Rahmen einer Gesamtstatik verwendet werden.
  </p>

  <div class="section-hdr">Querschnittsbemessung</div>
  <table class="kv-table" style="margin-bottom: 12px;">
    <tr><td style="width: 45%;">Berechnungsmethode</td><td>DAfStb Richtlinie Stahlfaserbeton</td></tr>
    <tr><td style="width: 55%;">Sicherheitskonzept</td><td>tragende Bauteile, Grenzzustand der Tragfähigkeit</td></tr>
    <tr><td>Bewehrung</td><td>{bew_konzept_de}</td></tr>
    <tr><td>Geometrie</td><td>plattenartiger Querschnitt &nbsp;&nbsp;&nbsp; <strong>h = {fmt_num(inp.h_mm, 0)} mm</strong></td></tr>
  </table>

  <div class="section-hdr">Stahlfaserbeton</div>
  <div class="two-col" style="margin-bottom: 12px;">
    <div class="col-half">
      <table class="kv-table">
        <tr><td>Betonfestigkeitsklasse</td><td class="val-cell">{inp.concrete_class}</td><td class="unit-cell"></td></tr>
        <tr><td>f<sup>f</sup><sub>cflk,L1</sub></td><td class="val-cell">{fmt_num(res.f_cflk_L1, 3)}</td><td class="unit-cell">N/mm²</td></tr>
        <tr><td>f<sup>f</sup><sub>cflk,L2</sub></td><td class="val-cell">{fmt_num(res.f_cflk_L2, 3)}</td><td class="unit-cell">N/mm²</td></tr>
      </table>
    </div>
    <div class="col-half">
      <table class="kv-table">
        <tr><td>Stahlfasern Leistungsklasse</td><td class="val-cell">{lk_str}</td></tr>
        <tr><td>Stahlfasern</td><td class="val-cell">{sf_type_str}</td></tr>
        <tr><td>Zugabeempfehlung</td><td class="val-cell">{dosage_str}</td></tr>
      </table>
    </div>
  </div>

  <div class="section-hdr">Bewehrung</div>
  <table class="kv-table" style="margin-bottom: 6px;">
    <tr><td style="width: 35%;">Streckgrenze f<sub>yk</sub></td><td class="val-cell" style="width: 13%;">{fmt_num(res.f_yk, 0, "n.a.")}</td><td class="unit-cell">N/mm²</td><td></td></tr>
  </table>
  <div class="two-col" style="margin-bottom: 14px;">
    <div class="col-half">
      <strong>Bewehrung A<sub>s,oben</sub></strong>
      <table class="kv-table">
        <tr><td>Stabdurchmesser d<sub>s</sub></td><td class="val-cell">{fmt_num(t1_lay.diam_1, 0) if t1_lay else "n.a."}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Stababstand s</td><td class="val-cell">{fmt_num(t1_lay.spacing, 0) if t1_lay else "n.a."}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Stahlquerschnitt A<sub>s</sub></td><td class="val-cell">{fmt_num(res.as_top_total_cm2_m, 2) if has_top_rebar else "n.a."}</td><td class="unit-cell">cm²/m</td></tr>
        <tr><td>Betondeckung c<sub>nom</sub></td><td class="val-cell">{cov_top_str}</td><td class="unit-cell">mm</td></tr>
      </table>
    </div>
    <div class="col-half">
      <strong>Bewehrung A<sub>s,unten</sub></strong>
      <table class="kv-table">
        <tr><td>Stabdurchmesser d<sub>s</sub></td><td class="val-cell">{fmt_num(b1_lay.diam_1, 0) if b1_lay else "n.a."}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Stababstand s</td><td class="val-cell">{fmt_num(b1_lay.spacing, 0) if b1_lay else "n.a."}</td><td class="unit-cell">mm</td></tr>
        <tr><td>Stahlquerschnitt A<sub>s</sub></td><td class="val-cell">{fmt_num(res.as_bottom_total_cm2_m, 2) if has_bot_rebar else "n.a."}</td><td class="unit-cell">cm²/m</td></tr>
        <tr><td>Betondeckung c<sub>nom</sub></td><td class="val-cell">{cov_bot_str}</td><td class="unit-cell">mm</td></tr>
      </table>
    </div>
  </div>

  <div class="section-hdr">Momententragfähigkeit</div>
  <table class="kv-table" style="font-size: 10.5pt; font-weight: 700; margin-bottom: 18px;">
    <tr><td>M<sub>Rd</sub><sup>-</sup> (Zug oben)</td><td class="val-cell">{fmt_num(res.neg_moment.m_rd_knm_m, 2)} kNm/m</td></tr>
    <tr><td>M<sub>Rd</sub><sup>+</sup> (Zug unten)</td><td class="val-cell">{fmt_num(res.pos_moment.m_rd_knm_m, 2)} kNm/m</td></tr>
  </table>

  <div class="section-hdr">Bemerkungen</div>
  <p style="font-size: 8pt; color: #333; line-height: 1.4;">
    Alle an Falconix übermittelten Eingabewerte werden als korrekt und daher außerhalb der Verantwortung von Falconix liegend
    angesehen. Von Falconix getroffene erforderliche Annahmen sind entscheidend für eine erfolgreiche Ausführung und daher vom
    verantwortlichen Planer zu prüfen.
  </p>
</div>

</body>
</html>
"""
    weasyprint.HTML(string=html_content).write_pdf(output_pdf_path)
    return output_pdf_path