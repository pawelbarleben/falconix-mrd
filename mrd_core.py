"""
mrd_core.py - Falconix Engineering GmbH
Rechenkern fuer Biegewiderstand (MRd) von Stahlfaserbeton- und Stahlbetonplatten
nach DAfStb-Richtlinie Stahlfaserbeton (Ausgabe Juni 2021) & DIN EN 1992-1-1 / NA.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Tuple
import math
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 1. NORMTABELLEN (DIN EN 1992-1-1 & DAfStb-Richtlinie Stahlfaserbeton)
# ---------------------------------------------------------------------------

CONCRETE_CLASSES: Dict[str, Dict[str, float]] = {
    "C20/25": {"f_ck": 20.0, "f_ctm": 2.2, "f_ctk_005": 1.5, "f_ctk_095": 2.9, "E_cm": 30000.0},
    "C25/30": {"f_ck": 25.0, "f_ctm": 2.6, "f_ctk_005": 1.8, "f_ctk_095": 3.3, "E_cm": 31000.0},
    "C28/35": {"f_ck": 28.0, "f_ctm": 2.8, "f_ctk_005": 1.9, "f_ctk_095": 3.6, "E_cm": 32000.0},
    "C30/37": {"f_ck": 30.0, "f_ctm": 2.9, "f_ctk_005": 2.0, "f_ctk_095": 3.8, "E_cm": 33000.0},
    "C32/40": {"f_ck": 32.0, "f_ctm": 3.0, "f_ctk_005": 2.1, "f_ctk_095": 3.9, "E_cm": 33500.0},
    "C35/45": {"f_ck": 35.0, "f_ctm": 3.2, "f_ctk_005": 2.2, "f_ctk_095": 4.2, "E_cm": 34000.0},
    "C40/50": {"f_ck": 40.0, "f_ctm": 3.5, "f_ctk_005": 2.5, "f_ctk_095": 4.6, "E_cm": 35000.0},
    "C45/55": {"f_ck": 45.0, "f_ctm": 3.8, "f_ctk_005": 2.7, "f_ctk_095": 4.9, "E_cm": 36000.0},
    "C50/60": {"f_ck": 50.0, "f_ctm": 4.1, "f_ctk_005": 2.9, "f_ctk_095": 5.3, "E_cm": 37000.0},
}

STEEL_CLASSES: Dict[str, Dict[str, float]] = {
    "S235": {"f_yk": 235.0, "E_s": 200.0}, # E_s in kN/mm²
    "S400": {"f_yk": 400.0, "E_s": 200.0},
    "S500": {"f_yk": 500.0, "E_s": 200.0},
}

PERFORMANCE_CLASS_VALUES: List[float] = [0.0, 0.4, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
REBAR_DIAMETERS: List[int] = [6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 50]

# ---------------------------------------------------------------------------
# 2. DATENMODELLE (INPUTS, METADATA, OUTPUTS)
# ---------------------------------------------------------------------------

@dataclass
class UserMetadata:
    company: str = "Falconix Engineering GmbH"
    street: str = "Siemensstrasse 24"
    location: str = "61267 Neu-Anspach"
    user_name: str = "Pavel Barleben"
    user_kuerzel: str = "PB"
    email: str = "barleben@falconixengineering.com"
    phone: str = "+49 151 0000 0000"
    project_name: str = ""
    client_name: str = ""
    position_name: str = "Bodenplatte"
    date_str: str = "20-10-2026"
    version: str = "V1.00.00"

@dataclass
class RebarLayerInput:
    """Definition einer einzelnen Bewehrungslage (1 bis 6) fuer Full-Modus"""
    layer_index: int              # 1..6 (ungerade = r-Richtung, gerade = s-Richtung, je nach Startrichtung)
    direction: Literal["r", "s"] = "r"
    diam_1: float = 0.0           # mm
    diam_2: float = 0.0           # mm (fuer Mischbewehrung / Stabbündel)
    spacing: float = 150.0        # mm
    bundled: bool = False         # No / Yes (Stabbündel nebeneinander oder uebereinander)

@dataclass
class SimplifiedRebarInput:
    """Vereinfachte Bewehrungsdefinition (je 1 Kreuznetz aus Lage 1 & 2)"""
    top_diam: float = 8.0
    top_spacing: float = 150.0
    center_diam: float = 8.0
    center_spacing: float = 150.0
    bottom_diam: float = 8.0
    bottom_spacing: float = 150.0

@dataclass
class MRdInput:
    # Geometrie & Schnitt
    h_mm: float = 150.0
    tensile_direction: Literal["r", "s"] = "r"
    element_type: Literal["Slab", "Wall"] = "Slab"
    n_ed_kn_m: float = 0.0        # Druckkraft positiv in Excel ('Compression Force' kN/m)

    # Baustoffe
    concrete_class: str = "C25/30"
    steel_class: str = "S500"
    alpha_cc: float = 0.85
    gamma_c: float = 1.50
    gamma_s: float = 1.15

    # Stahlfaser-Konzept
    fiber_concept: Literal["Performance Class", "Residual Strength", "No Steel Fibers"] = "Performance Class"
    L1: float = 0.9
    L2: float = 0.9
    f_R1m: float = 1.80
    f_R4m: float = 1.70
    fiber_name: str = "n.a."
    dosage_kg_m3: Optional[float] = None
    kappa_F: float = 1.00         # 1.0 fuer Platten b > 5h unter Biegung
    kappa_G: float = 1.70         # 1.70 pauschal fuer Platten
    alpha_ct_f: float = 0.85
    gamma_ct_f: float = 1.25

    # Bewehrungskonzept
    rebar_concept: Literal["Fibers Only", "Top", "Center", "Bottom", "Top + Bottom"] = "Fibers Only"
    rebar_definition: Literal["Simplified", "Full"] = "Simplified"
    cover_top_mm: float = 35.0
    cover_bottom_mm: float = 35.0
    dir_layer1_top: Literal["r", "s"] = "r"
    dir_layer1_center: Literal["r", "s"] = "r"
    dir_layer1_bottom: Literal["r", "s"] = "r"

    simplified_rebar: SimplifiedRebarInput = field(default_factory=SimplifiedRebarInput)
    full_top_layers: List[RebarLayerInput] = field(default_factory=list)
    full_bottom_layers: List[RebarLayerInput] = field(default_factory=list)

@dataclass
class ActiveSteelLayer:
    label: str            # 't1', 't2', 't3', 'b3', 'b2', 'b1'
    group: Literal["Top", "Bottom"]
    diam_1: float         # mm
    diam_2: float         # mm
    spacing: float        # mm
    bundled: bool
    as_cm2_m: float       # cm²/m
    c_mm: float           # Randabstand bis Bewehrungsrand (Betondeckung + Vorlagen) in mm
    z_from_top_mm: float  # Abstand Schwerpunkt von Oberkante in mm
    z_from_mid_mm: float  # Abstand Schwerpunkt zur Bauteilmitte (+ nach oben, - nach unten) in mm

@dataclass
class ForceRowResult:
    name: str             # 'Beton (c)', 'SF (ct)', 'Stahl (t1)', ...
    z_mm: Optional[float] # Hebelarm bezogen auf Querschnittsmitte h/2 [mm]
    eps_permille: Optional[float] # Dehnung [‰]
    sigma_mpa: Optional[float]    # Spannung [N/mm²]
    f_kn_m: Optional[float]       # Kraft [kN/m]

@dataclass
class BendingDirectionResult:
    direction_label: str  # 'Zug oben' oder 'Zug unten'
    eps_top_permille: float
    eps_bottom_permille: float
    x_mm: float
    d_eff_mm: float
    as_tensile_cm2_m: float
    m_rd_knm_m: float
    rows: List[ForceRowResult]
    curve_eps_permille: List[float]
    curve_m_knm_m: List[float]
    curve_x_mm: List[float]

@dataclass
class MRdResult:
    inputs: MRdInput
    metadata: UserMetadata
    # Beton & Stahl
    f_ck: float
    f_cd: float
    f_yk: Optional[float]
    f_yd: Optional[float]
    E_s: Optional[float]
    has_rebar: bool
    has_fibers: bool
    # Stahlfaser-Zwischenwerte
    f_cflk_L1: float
    f_cflk_L2: float
    beta_L1: float
    beta_L2: float
    beta_u: float
    f_ct0_L1: float
    f_ct0_L2: float
    f_ct0_u: float
    f_ctR_L1: float
    f_ctR_L2: float
    f_ctR_u: float
    f_ctd_L1: float
    f_ctd_L2: float
    f_ctd_u: float
    # Aktive Bewehrungslagen in Berechnungsrichtung (t1..t3, b3..b1)
    steel_layers: Dict[str, ActiveSteelLayer]
    d_top_mm: Optional[float]
    d_bottom_mm: Optional[float]
    as_top_total_cm2_m: float
    as_bottom_total_cm2_m: float
    # Ergebnisse Biegung
    neg_moment: BendingDirectionResult # Zug oben (MRd-)
    pos_moment: BendingDirectionResult # Zug unten (MRd+)


# ---------------------------------------------------------------------------
# 3. GEOMETRIE DER BEWEHRUNGSLAGEN (SIMPLIFIED & FULL INKL. STABBÜNDEL)
# ---------------------------------------------------------------------------

def build_active_steel_layers(inp: MRdInput) -> Dict[str, ActiveSteelLayer]:
    """
    Erzeugt die bis zu 6 wirksamen Bewehrungslagen (t1, t2, t3 oben; b3, b2, b1 unten)
    in der gewaehlten Zug-Berechnungsrichtung ('r' oder 's').
    Formel auf PDF S. 2: As = (1/s) * (pi/4) * (d1^2 + d2^2) [umgerechnet in cm²/m]
    """
    layers: Dict[str, ActiveSteelLayer] = {}
    if inp.rebar_concept == "Fibers Only":
        return layers

    h = inp.h_mm
    target_dir = inp.tensile_direction

    def compute_as_cm2_m(d1: float, d2: float, s: float) -> float:
        if s <= 0.0 or (d1 <= 0.0 and d2 <= 0.0):
            return 0.0
        # s in mm, d in mm -> As in mm²/m = (1000 / s) * (pi/4) * (d1^2 + d2^2) -> /100 fuer cm²/m
        return (10.0 / s) * (math.pi / 4.0) * (d1**2 + d2**2)

    if inp.rebar_definition == "Simplified":
        # Im Simplified-Modus gibt es maximal 1 Lage pro aktivierter Zone in Zielrichtung
        # Top
        if inp.rebar_concept in ("Top", "Top + Bottom"):
            d_t = inp.simplified_rebar.top_diam
            s_t = inp.simplified_rebar.top_spacing
            # Wenn Layer 1 in target_dir liegt -> c = cover_top, sonst liegt Layer 2 um d_t tiefer
            offset = 0.0 if inp.dir_layer1_top == target_dir else d_t
            c_val = inp.cover_top_mm + offset
            z_top = c_val + d_t / 2.0
            layers["t1"] = ActiveSteelLayer(
                label="t1", group="Top", diam_1=d_t, diam_2=0.0, spacing=s_t, bundled=False,
                as_cm2_m=compute_as_cm2_m(d_t, 0.0, s_t),
                c_mm=c_val, z_from_top_mm=z_top, z_from_mid_mm=(h / 2.0) - z_top
            )
        # Center (Mittelbewehrung wird als mittige Lage abgebildet)
        if inp.rebar_concept == "Center":
            d_c = inp.simplified_rebar.center_diam
            s_c = inp.simplified_rebar.center_spacing
            offset = -d_c / 2.0 if inp.dir_layer1_center == target_dir else d_c / 2.0
            z_top = (h / 2.0) + offset
            c_val = z_top - d_c / 2.0
            layers["b1"] = ActiveSteelLayer(
                label="b1", group="Bottom", diam_1=d_c, diam_2=0.0, spacing=s_c, bundled=False,
                as_cm2_m=compute_as_cm2_m(d_c, 0.0, s_c),
                c_mm=h - z_top - d_c / 2.0, z_from_top_mm=z_top, z_from_mid_mm=(h / 2.0) - z_top
            )
        # Bottom
        if inp.rebar_concept in ("Bottom", "Top + Bottom"):
            d_b = inp.simplified_rebar.bottom_diam
            s_b = inp.simplified_rebar.bottom_spacing
            offset = 0.0 if inp.dir_layer1_bottom == target_dir else d_b
            c_val = inp.cover_bottom_mm + offset
            z_bot = c_val + d_b / 2.0
            z_top = h - z_bot
            layers["b1"] = ActiveSteelLayer(
                label="b1", group="Bottom", diam_1=d_b, diam_2=0.0, spacing=s_b, bundled=False,
                as_cm2_m=compute_as_cm2_m(d_b, 0.0, s_b),
                c_mm=c_val, z_from_top_mm=z_top, z_from_mid_mm=(h / 2.0) - z_top
            )
    else:
        # Full Reinforcement Definition (6 Lagen oben: 1..6, 6 Lagen unten: 1..6 im Kreuzverbund)
        if inp.rebar_concept in ("Top", "Top + Bottom"):
            cum_offset = inp.cover_top_mm
            t_idx = 1
            for lay in inp.full_top_layers[:6]:
                d_eff_thick = max(lay.diam_1, lay.diam_2) if not lay.bundled else (lay.diam_1 + lay.diam_2)
                if lay.direction == target_dir and (lay.diam_1 > 0 or lay.diam_2 > 0) and t_idx <= 3:
                    # Schwerpunkt bei Einzelstab/Nebeneinander vs. Stabbündel
                    d_cg = max(lay.diam_1, lay.diam_2) / 2.0
                    z_top = cum_offset + d_cg
                    key = f"t{t_idx}"
                    layers[key] = ActiveSteelLayer(
                        label=key, group="Top", diam_1=lay.diam_1, diam_2=lay.diam_2,
                        spacing=lay.spacing, bundled=lay.bundled,
                        as_cm2_m=compute_as_cm2_m(lay.diam_1, lay.diam_2, lay.spacing),
                        c_mm=cum_offset, z_from_top_mm=z_top, z_from_mid_mm=(h / 2.0) - z_top
                    )
                    t_idx += 1
                cum_offset += max(lay.diam_1, lay.diam_2)

        if inp.rebar_concept in ("Bottom", "Top + Bottom"):
            cum_offset = inp.cover_bottom_mm
            b_idx = 1
            for lay in inp.full_bottom_layers[:6]:
                if lay.direction == target_dir and (lay.diam_1 > 0 or lay.diam_2 > 0) and b_idx <= 3:
                    d_cg = max(lay.diam_1, lay.diam_2) / 2.0
                    z_bot = cum_offset + d_cg
                    z_top = h - z_bot
                    key = f"b{b_idx}"
                    layers[key] = ActiveSteelLayer(
                        label=key, group="Bottom", diam_1=lay.diam_1, diam_2=lay.diam_2,
                        spacing=lay.spacing, bundled=lay.bundled,
                        as_cm2_m=compute_as_cm2_m(lay.diam_1, lay.diam_2, lay.spacing),
                        c_mm=cum_offset, z_from_top_mm=z_top, z_from_mid_mm=(h / 2.0) - z_top
                    )
                    b_idx += 1
                cum_offset += max(lay.diam_1, lay.diam_2)

    return layers


# ---------------------------------------------------------------------------
# 4. SPANNUNGS-DEHNUNGS-GESETZE & INTEGRALE
# ---------------------------------------------------------------------------

def concrete_force_and_lever(eps_c_edge_permille: float, x_mm: float, h_mm: float, b_mm: float, f_cd: float, is_top_compressed: bool) -> Tuple[float, float, float]:
    """
    Berechnet analytisch exakt die Betondruckkraft F_c [kN/m] (negativ als Druckkraft)
    und den Hebelarm z_c [mm] bezogen auf Querschnittsmitte h/2 fuer das Parabel-Rechteck-Diagramm
    nach DIN EN 1992-1-1, 3.1.7 (eps_c2 = 2.0 ‰, eps_cu2 = 3.5 ‰, n = 2).
    """
    e_abs = abs(eps_c_edge_permille)
    if e_abs <= 1e-12 or x_mm <= 1e-12:
        return 0.0, 0.0, 0.0

    # Parabelbereich von e = 0 bis min(e_abs, 2.0)
    e_p = min(e_abs, 2.0)
    u_p = e_p / 2.0
    # Integral sigma_c(e) de fuer Parabelteil: f_cd * 2.0 * (u_p^2 - u_p^3 / 3)
    int_sigma_p = f_cd * 2.0 * (u_p**2 - (u_p**3) / 3.0)
    # Erstes Moment Integral e * sigma_c(e) de fuer Parabelteil: f_cd * 4.0 * (2/3 * u_p^3 - 1/4 * u_p^4)
    int_e_sigma_p = f_cd * 4.0 * ((2.0 / 3.0) * (u_p**3) - 0.25 * (u_p**4))

    # Rechteckbereich falls e_abs > 2.0
    if e_abs > 2.0:
        de_r = e_abs - 2.0
        int_sigma_r = f_cd * de_r
        int_e_sigma_r = f_cd * 0.5 * (e_abs**2 - 4.0)
    else:
        int_sigma_r = 0.0
        int_e_sigma_r = 0.0

    int_sigma_total = int_sigma_p + int_sigma_r
    int_e_sigma_total = int_e_sigma_p + int_e_sigma_r

    # Umrechnung von Dehnungskoordinate e auf Querschnittskoordinate y (y = x * e / e_abs)
    f_c_abs_kn_m = b_mm * (x_mm / e_abs) * int_sigma_total / 1000.0
    e_centroid = int_e_sigma_total / int_sigma_total
    y_from_na_mm = (x_mm / e_abs) * e_centroid # Abstand von der Nulllinie zum Druckmittelpunkt
    dist_from_comp_edge_mm = x_mm - y_from_na_mm

    if is_top_compressed:
        # Druckrand ist oben -> Abstand zur Mitte (h/2) ist positiv (+ nach oben)
        z_mid_mm = (h_mm / 2.0) - dist_from_comp_edge_mm
    else:
        # Druckrand ist unten -> Abstand zur Mitte (h/2) ist negativ (- nach unten)
        z_mid_mm = -((h_mm / 2.0) - dist_from_comp_edge_mm)

    sigma_edge = f_cd * (1.0 - (1.0 - u_p)**2) if e_abs <= 2.0 else f_cd
    return -f_c_abs_kn_m, z_mid_mm, -sigma_edge


def fiber_force_and_lever(eps_t_edge_permille: float, h_t_mm: float, x_mm: float, h_mm: float, b_mm: float,
                          f_ctd_L1: float, f_ctd_L2: float, is_top_tensioned: bool) -> Tuple[float, float, float]:
    """
    Berechnet analytisch exakt die Stahlfaserzugkraft F_ct [kN/m] (positiv)
    und den Hebelarm z_ct [mm] bezogen auf Querschnittsmitte h/2 fuer die trilineare
    Spannungs-Dehnungslinie nach DAfStb-Richtlinie Bild R.2:
      0.0 ‰ -> 0
      0.1 ‰ -> f_ctd_L1
      3.5 ‰ -> f_ctd_L1
      25.0 ‰ -> f_ctd_L2
    """
    e_max = eps_t_edge_permille
    if e_max <= 1e-12 or h_t_mm <= 1e-12 or (f_ctd_L1 <= 0 and f_ctd_L2 <= 0):
        return 0.0, 0.0, 0.0

    def sigma_f(e: float) -> float:
        if e <= 0.0:
            return 0.0
        elif e <= 0.1:
            return f_ctd_L1 * (e / 0.1)
        elif e <= 3.5:
            return f_ctd_L1
        else:
            return f_ctd_L1 + (f_ctd_L2 - f_ctd_L1) * ((e - 3.5) / (25.0 - 3.5))

    # Stützstellen der trilinearen Funktion bis e_max
    knots = [0.0]
    for k in (0.1, 3.5, 25.0):
        if k < e_max:
            knots.append(k)
    knots.append(e_max)

    area_total = 0.0
    moment_total = 0.0
    for i in range(len(knots) - 1):
        e0, e1 = knots[i], knots[i+1]
        de = e1 - e0
        if de <= 1e-15:
            continue
        s0, s1 = sigma_f(e0), sigma_f(e1)
        area_seg = 0.5 * (s0 + s1) * de
        if s0 + s1 > 1e-15:
            e_c_seg = e0 + (de / 3.0) * (s0 + 2.0 * s1) / (s0 + s1)
        else:
            e_c_seg = 0.5 * (e0 + e1)
        area_total += area_seg
        moment_total += area_seg * e_c_seg

    if area_total <= 1e-15:
        return 0.0, 0.0, 0.0

    f_ct_kn_m = b_mm * (h_t_mm / e_max) * area_total / 1000.0
    e_centroid = moment_total / area_total
    y_from_na_mm = (h_t_mm / e_max) * e_centroid # Abstand von Nulllinie in Richtung Zugrand
    dist_from_comp_edge_mm = x_mm + y_from_na_mm

    if is_top_tensioned:
        # Zugrand ist oben, Druckrand ist unten -> Abstand von Unterkante ist dist_from_comp_edge_mm
        z_from_top_mm = h_mm - dist_from_comp_edge_mm
    else:
        # Zugrand ist unten, Druckrand ist oben -> Abstand von Oberkante ist dist_from_comp_edge_mm
        z_from_top_mm = dist_from_comp_edge_mm

    z_mid_mm = (h_mm / 2.0) - z_from_top_mm
    return f_ct_kn_m, z_mid_mm, sigma_f(e_max)


def steel_stress(eps_s_permille: float, f_yd: float, E_s_kn_mm2: float) -> float:
    """Bilineare Spannungs-Dehnungslinie fuer Betonstahl (horizontaler Ast ab f_yd)."""
    # E_s in kN/mm² = 1000 N/mm²; eps in ‰ = 1e-3 -> sigma [N/mm²] = eps [‰] * E_s [kN/mm²]
    sigma_el = eps_s_permille * E_s_kn_mm2
    if sigma_el > f_yd:
        return f_yd
    elif sigma_el < -f_yd:
        return -f_yd
    return sigma_el


# ---------------------------------------------------------------------------
# 5. HAUPTBERECHNUNG (QUERSCHNITTSGLEICHGEWICHT & MOMENTEN-DEHNUNGS-KURVE)
# ---------------------------------------------------------------------------

def solve_section_for_direction(
    inp: MRdInput,
    is_top_tensioned: bool,
    f_cd: float,
    f_yd: Optional[float],
    E_s: Optional[float],
    f_ctd_L1: float,
    f_ctd_L2: float,
    steel_layers: Dict[str, ActiveSteelLayer]
) -> BendingDirectionResult:
    """
    Löst das Querschnittsgleichgewicht fuer 'Zug oben' (is_top_tensioned=True, MRd-)
    oder 'Zug unten' (is_top_tensioned=False, MRd+).
    """
    h = inp.h_mm
    b = 1000.0 # 1m Plattenstreifen
    has_fibers = inp.fiber_concept != "No Steel Fibers" and (f_ctd_L1 > 0 or f_ctd_L2 > 0)

    # Bestimme statische Nutzhoehe d und Zugbewehrung fuer diese Biegerichtung
    tensile_group = "Top" if is_top_tensioned else "Bottom"
    tensile_layers = [lay for lay in steel_layers.values() if lay.group == tensile_group and lay.as_cm2_m > 0]
    as_tensile = sum(lay.as_cm2_m for lay in tensile_layers)

    if tensile_layers:
        if is_top_tensioned:
            # Zug oben -> Abstand zum unteren Druckrand = h - z_from_top
            d_eff = sum(lay.as_cm2_m * (h - lay.z_from_top_mm) for lay in tensile_layers) / as_tensile
        else:
            # Zug unten -> Abstand zum oberen Druckrand = z_from_top
            d_eff = sum(lay.as_cm2_m * lay.z_from_top_mm for lay in tensile_layers) / as_tensile
    else:
        # Ohne Stabbewehrung gilt laut DAfStb-RL 6.1 (2)P: statische Hoehe d = h
        d_eff = h

    def evaluate_given_x(x_mm: float, eps_t_target: float) -> Tuple[float, float, float, List[ForceRowResult]]:
        """
        Gegeben Druckzonenhoehe x_mm in (0, h) und Zugrand- bzw. Stahl-Dehnung eps_t_target > 0:
        Bestimmt das Dehnungsprofil (unter Beachtung von eps_c <= 3.5 ‰ und eps_ct <= 25.0 ‰),
        alle Teilkraefte und das resultierende Biegemoment.
        """
        # Bei reinem Stahlfaserbeton (oder mit Fasern) bezieht sich eps_t_target auf den gezogenen Bauteilrand (max 25 ‰)
        h_t = h - x_mm
        if h_t <= 1e-9:
            eps_c_edge = -3.5
            eps_t_edge = 0.0
        else:
            eps_t_edge = eps_t_target
            eps_c_edge = -eps_t_edge * (x_mm / h_t)
            # Begrenzung der Betondruckstauchung auf eps_cu2 = -3.5 ‰
            if abs(eps_c_edge) > 3.5:
                eps_c_edge = -3.5
                eps_t_edge = 3.5 * (h_t / x_mm)

        # 1. Betondruckkraft
        f_c, z_c, sig_c = concrete_force_and_lever(
            eps_c_edge, x_mm, h, b, f_cd, is_top_compressed=(not is_top_tensioned)
        )

        # 2. Stahlfaserzugkraft
        if has_fibers and h_t > 0:
            f_ct, z_ct, sig_ct = fiber_force_and_lever(
                eps_t_edge, h_t, x_mm, h, b, f_ctd_L1, f_ctd_L2, is_top_tensioned=is_top_tensioned
            )
        else:
            f_ct, z_ct, sig_ct = 0.0, 0.0, 0.0

        # 3. Betonstahllagen (t1, t2, t3, b3, b2, b1)
        row_list: List[ForceRowResult] = [
            ForceRowResult("Beton (c)", z_c, eps_c_edge, sig_c, f_c),
            ForceRowResult("SF (ct)", z_ct if has_fibers else None, eps_t_edge if has_fibers else None, sig_ct if has_fibers else None, f_ct if has_fibers else None),
        ]

        f_steel_total = 0.0
        m_steel_total_knm = 0.0

        eps_top = eps_t_edge if is_top_tensioned else eps_c_edge
        eps_bot = eps_c_edge if is_top_tensioned else eps_t_edge

        for key in ["t1", "t2", "t3", "b3", "b2", "b1"]:
            if key in steel_layers and f_yd is not None and E_s is not None:
                lay = steel_layers[key]
                # Lineare Dehnungsverteilung ueber Querschnittshoehe h
                eps_lay = eps_top + (eps_bot - eps_top) * (lay.z_from_top_mm / h)
                sig_lay = steel_stress(eps_lay, f_yd, E_s)
                # As in cm²/m = 100 mm²/m -> F_s [kN/m] = As [cm²/m] * 100 * sig [N/mm²] / 1000 = As * sig / 10.0
                f_lay = lay.as_cm2_m * sig_lay / 10.0
                f_steel_total += f_lay
                # Vorzeichenkonvention PDF S. 5: M = - sum(F_i * z_i) / 1000
                m_steel_total_knm += (-f_lay * lay.z_from_mid_mm) / 1000.0
                row_list.append(ForceRowResult(f"Stahl ({key})", lay.z_from_mid_mm, eps_lay, sig_lay, f_lay))
            else:
                row_list.append(ForceRowResult(f"Stahl ({key})", None, None, None, None))

        # Summe der inneren Normalkraefte + aeussere Druckkraft (inp.n_ed_kn_m)
        n_res = f_c + f_ct + f_steel_total + inp.n_ed_kn_m
        m_total_knm = (-f_c * z_c - f_ct * z_ct) / 1000.0 + m_steel_total_knm

        return n_res, m_total_knm, eps_c_edge, row_list

    def solve_equilibrium_at_strain(eps_t_target: float) -> Tuple[float, float, float, List[ForceRowResult]]:
        if eps_t_target <= 1e-7:
            return h / 2.0, 0.0, 0.0, []
        # Suche Nullstelle von n_res(x) fuer x in [1e-5, h - 1e-5]
        def obj(x_try: float) -> float:
            return evaluate_given_x(x_try, eps_t_target)[0]

        x_sol = brentq(obj, 1e-4, h - 1e-4, xtol=1e-12)
        n_res, m_val, eps_c_val, rows = evaluate_given_x(x_sol, eps_t_target)
        return x_sol, m_val, eps_c_val, rows

    # Berechne Momenten-Dehnungskurve (100 Stützpunkte von 0 bis 25 ‰ wie in Excel CX11:DB211)
    # In Excel wird ein geometrisch gestuftes Gitter von 0 bis 25 ‰ verwendet
    eps_grid = [0.0] + list(np.geomspace(0.01, 25.0, 99))
    curve_eps = []
    curve_m = []
    curve_x = []

    for e_val in eps_grid:
        if e_val == 0.0:
            curve_eps.append(0.0)
            curve_m.append(0.0)
            curve_x.append(h / 3.0)
        else:
            x_i, m_i, _, _ = solve_equilibrium_at_strain(float(e_val))
            curve_eps.append(float(e_val))
            curve_m.append(float(m_i))
            curve_x.append(float(x_i))

    # GZT-Bemessungszustand bei eps_ct,u = 25.0 ‰ (DAfStb-RL Abschnitt 6.1)
    x_ult, m_ult, eps_c_ult, rows_ult = solve_equilibrium_at_strain(25.0)

    eps_top_ult = 25.0 if is_top_tensioned else eps_c_ult
    eps_bot_ult = eps_c_ult if is_top_tensioned else 25.0

    return BendingDirectionResult(
        direction_label="Zug oben" if is_top_tensioned else "Zug unten",
        eps_top_permille=eps_top_ult,
        eps_bottom_permille=eps_bot_ult,
        x_mm=x_ult,
        d_eff_mm=d_eff if tensile_layers else h,
        as_tensile_cm2_m=as_tensile,
        m_rd_knm_m=m_ult,
        rows=rows_ult,
        curve_eps_permille=curve_eps,
        curve_m_knm_m=curve_m,
        curve_x_mm=curve_x,
    )


def calculate_mrd(inp: MRdInput, metadata: Optional[UserMetadata] = None) -> MRdResult:
    if metadata is None:
        metadata = UserMetadata()

    c_props = CONCRETE_CLASSES[inp.concrete_class]
    f_ck = c_props["f_ck"]
    f_cd = inp.alpha_cc * f_ck / inp.gamma_c

    has_rebar = inp.rebar_concept != "Fibers Only"
    s_props = STEEL_CLASSES[inp.steel_class]
    f_yk = s_props["f_yk"] if has_rebar else None
    f_yd = (f_yk / inp.gamma_s) if (has_rebar and f_yk is not None) else None
    E_s = s_props["E_s"] if has_rebar else None

    # Stahlfaser-Kennwerte nach DAfStb-Richtlinie Stahlfaserbeton (Juni 2021)
    has_fibers = inp.fiber_concept != "No Steel Fibers"
    if not has_fibers:
        f_cflk_L1 = f_cflk_L2 = 0.0
    elif inp.fiber_concept == "Performance Class":
        f_cflk_L1 = float(inp.L1)
        f_cflk_L2 = float(inp.L2)
    else: # Residual Strength (f_R1m, f_R4m)
        f_cflk_L1 = 0.51 * inp.f_R1m
        f_cflk_L2 = 0.51 * inp.f_R4m

    if has_fibers and f_cflk_L1 > 0:
        ratio = f_cflk_L2 / f_cflk_L1
        beta_L1 = 0.40 if ratio >= 0.7 else 0.0
        beta_u = 0.37
        if ratio < 0.7:
            beta_L2 = 0.0
        elif ratio <= 1.0:
            # Exakte lineare Interpolation nach Bild P.1 zwischen (0.7, 0.25) und (1.0, 0.35)
            beta_L2 = 0.25 + (0.35 - 0.25) / (1.0 - 0.7) * (ratio - 0.7)
        elif ratio <= 1.5:
            beta_L2 = 0.35 + (0.44 - 0.35) / (1.5 - 1.0) * (ratio - 1.0)
        else:
            beta_L2 = 0.44
    else:
        beta_L1 = beta_L2 = beta_u = 0.0

    f_ct0_L1 = f_cflk_L1 * beta_L1
    f_ct0_L2 = f_cflk_L2 * beta_L2
    f_ct0_u  = f_cflk_L2 * beta_u

    f_ctR_L1 = inp.kappa_F * inp.kappa_G * f_ct0_L1
    f_ctR_L2 = inp.kappa_F * inp.kappa_G * f_ct0_L2
    f_ctR_u  = inp.kappa_F * inp.kappa_G * f_ct0_u

    f_ctd_L1 = inp.alpha_ct_f * f_ctR_L1 / inp.gamma_ct_f
    f_ctd_L2 = inp.alpha_ct_f * f_ctR_L2 / inp.gamma_ct_f
    f_ctd_u  = inp.alpha_ct_f * f_ctR_u  / inp.gamma_ct_f

    steel_layers = build_active_steel_layers(inp)

    top_layers = [l for l in steel_layers.values() if l.group == "Top" and l.as_cm2_m > 0]
    bot_layers = [l for l in steel_layers.values() if l.group == "Bottom" and l.as_cm2_m > 0]

    as_top_total = sum(l.as_cm2_m for l in top_layers)
    as_bot_total = sum(l.as_cm2_m for l in bot_layers)

    d_top = (sum(l.as_cm2_m * (inp.h_mm - l.z_from_top_mm) for l in top_layers) / as_top_total) if as_top_total > 0 else None
    d_bot = (sum(l.as_cm2_m * l.z_from_top_mm for l in bot_layers) / as_bot_total) if as_bot_total > 0 else None

    neg_res = solve_section_for_direction(inp, True,  f_cd, f_yd, E_s, f_ctd_L1, f_ctd_L2, steel_layers)
    pos_res = solve_section_for_direction(inp, False, f_cd, f_yd, E_s, f_ctd_L1, f_ctd_L2, steel_layers)

    return MRdResult(
        inputs=inp,
        metadata=metadata,
        f_ck=f_ck,
        f_cd=f_cd,
        f_yk=f_yk,
        f_yd=f_yd,
        E_s=E_s,
        has_rebar=has_rebar,
        has_fibers=has_fibers,
        f_cflk_L1=f_cflk_L1,
        f_cflk_L2=f_cflk_L2,
        beta_L1=beta_L1,
        beta_L2=beta_L2,
        beta_u=beta_u,
        f_ct0_L1=f_ct0_L1,
        f_ct0_L2=f_ct0_L2,
        f_ct0_u=f_ct0_u,
        f_ctR_L1=f_ctR_L1,
        f_ctR_L2=f_ctR_L2,
        f_ctR_u=f_ctR_u,
        f_ctd_L1=f_ctd_L1,
        f_ctd_L2=f_ctd_L2,
        f_ctd_u=f_ctd_u,
        steel_layers=steel_layers,
        d_top_mm=d_top,
        d_bottom_mm=d_bot,
        as_top_total_cm2_m=as_top_total,
        as_bottom_total_cm2_m=as_bot_total,
        neg_moment=neg_res,
        pos_moment=pos_res,
    )
