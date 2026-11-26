"""Step 12 - Report (PDF): meters, devices and data sources for a real deployment, with the measured results
(13 houses; one house; one house + the two reference inputs).
Output: reports/EnergyMind_meters_and_data_sources.pdf
Usage : python 12_data_sources_report.py
"""
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

import report_style  # noqa: F401,E402  (registers the fonts)
from common import MODELS, REPORTS  # noqa: E402

OUT = str(REPORTS / "EnergyMind_meters_and_data_sources.pdf")
_SH = MODELS / "single_house_refs"                      # final: one house + transformer + reference inputs
_BEST = json.loads((_SH / "best_model.json").read_text())["model"]
_R1 = json.loads((_SH / f"{_BEST}_results.json").read_text())
_R0 = json.loads((MODELS / "single_house" / f"{_BEST}_results.json").read_text())      # one house, no references
_R13 = json.loads((MODELS / "single" / f"{_BEST}_results.json").read_text())
_ALGO = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "random_forest": "Random Forest", "extra_trees": "Extra Trees",
         "hist_gb": "HistGradientBoosting", "mlp": "MLP"}[_BEST]

NAVY = colors.HexColor("#1F3864"); GREY = colors.HexColor("#595959"); LIGHT = colors.HexColor("#EEF2F8")
ZEBRA = colors.HexColor("#F3F5F9"); LINE = colors.HexColor("#BFC7D5")
C = {"STEG METER": "#C62828", "STEG TRAFO": "#C62828", "STEG PV/BMS": "#C62828", "COMPUTED": "#1565C0",
     "ESTIMATED": "#6D4C41", "ADDED DEVICE": "#EF6C00", "NOT AVAILABLE": "#757575"}

s_title = ParagraphStyle("t", fontName="AB", fontSize=20, leading=24, textColor=NAVY, spaceAfter=4)
s_sub = ParagraphStyle("s", fontName="A", fontSize=10.5, leading=14, textColor=GREY, spaceAfter=10)
s_h = ParagraphStyle("h", fontName="AB", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=10, spaceAfter=5, keepWithNext=1)
s_b = ParagraphStyle("b", fontName="A", fontSize=9.3, leading=12.5)
s_c = ParagraphStyle("c", fontName="A", fontSize=8.3, leading=10.5)
s_ch = ParagraphStyle("ch", fontName="AB", fontSize=8.3, leading=10.5, textColor=colors.white)
s_bul = ParagraphStyle("bu", parent=s_b, leftIndent=12, bulletIndent=2, spaceAfter=2)
s_src = ParagraphStyle("src", fontName="A", fontSize=7.6, leading=9.6, leftIndent=8, firstLineIndent=-8, spaceAfter=1.5)


def P(t, st=s_c):
    return Paragraph(t, st)


def src(t):
    return f'<font name="AB" color="{C[t]}">{t}</font>'


def table(rows, widths, head=True):
    data = [[P(c, s_ch) for c in rows[0]]] + [[c if not isinstance(c, str) else P(c) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(2, len(data), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st))
    return t


def box(text, st, bg=LIGHT, border=NAVY):
    t = Table([[Paragraph(text, st)]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, border), ("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def diagram():
    W, H = 170 * mm, 82 * mm
    d = Drawing(W, H)

    def node(x, y, w, h, title, lines, fill, stroke):
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor(stroke), strokeWidth=0.9))
        d.add(String(x + w / 2, y + h - 12, title, fontName="AB", fontSize=8.2, textAnchor="middle", fillColor=NAVY))
        for i, l in enumerate(lines):
            d.add(String(x + w / 2, y + h - 23 - i * 9.5, l, fontName="A", fontSize=7, textAnchor="middle"))

    def arrow(x1, y1, x2, y2, label=None):
        d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=0.9))
        import math
        a = math.atan2(y2 - y1, x2 - x1); L = 5
        d.add(Polygon([x2, y2, x2 - L * math.cos(a - 0.4), y2 - L * math.sin(a - 0.4),
                       x2 - L * math.cos(a + 0.4), y2 - L * math.sin(a + 0.4)], fillColor=GREY, strokeColor=GREY))
        if label:
            d.add(String((x1 + x2) / 2, (y1 + y2) / 2 + 3, label, fontName="AI", fontSize=6.5, textAnchor="middle", fillColor=GREY))

    G, Gs = "#E8F5E9", "#2E7D32"
    # left column: STEG data sources
    node(0, 168, 150, 56, "STEG smart meters", ["every house on the transformer", "U, I, P, Q per phase", "15-min load profile (to confirm)"], G, Gs)
    node(0, 92, 150, 56, "STEG transformer supervision", ["Smart Grid equipment", "loading %, P, Q", "LV-side voltage"], G, Gs)
    node(0, 6, 150, 66, "PV + battery supervision", ["hybrid inverter (STEG list)", "+ LFP battery with BMS", "PV, SOC, V, I, temperature,", "status, backup, transfer switch"], G, Gs)
    # centre: platform
    node(205, 50, 150, 130, "EnergyMind platform", ["1. align all data at 15 min", "2. compute derived values", "   (own history, meter ratio,", "   PV vs theory)", "3. anomaly detection model", "4. AI agent: flag / estimate /", "   quarantine / notify", "5. human approval for actions"], "#E3F2FD", "#1F3864")
    # right: outputs
    node(395, 120, 87, 60, "Grid operators", ["alerts, category,", "suggested action"], "#FFEBEE", "#C62828")
    node(395, 30, 87, 60, "Added devices", ["second meter (kWh)", "roof sensor: irradiance,", "module temperature"], "#FFF3E0", "#EF6C00")
    for y in (196, 120, 39):
        arrow(150, y, 205, 115 if y != 120 else 115)
    arrow(355, 140, 395, 150)
    arrow(395, 60, 355, 85)
    k = 0.78
    d.scale(k, k); d.width, d.height = W * k, H * k
    d.hAlign = 'CENTER'
    return d


def footer(c, doc):
    c.saveState()
    c.setFont("A", 7.5); c.setFillColor(GREY)
    c.drawString(20 * mm, 10 * mm, "EnergyMind Tunisia · Real deployment: meters and data sources")
    c.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


S = []
S.append(Paragraph("EnergyMind Tunisia: from simulation to a real deployment", s_title))
S.append(Paragraph("Which data STEG already has, which two inexpensive devices are added, and how every model input is obtained", s_sub))
S.append(box("<b>In one sentence.</b> The model uses data that STEG already collects (the smart meter of the house, the "
             "transformer supervision of the Smart Grid project, the PV inverter and battery data required by the STEG 2026 reference) "
             "plus <b>two inexpensive devices</b>: a <b>second meter</b> on each house and, on PV houses, a <b>roof reference cell</b> "
             "(irradiance and module temperature). They give the independent references that a single meter cannot: without them a "
             "meter bias, a slow drift and PV shading are invisible (section 6). The model never uses appliance-level data.", s_b))

S.append(Paragraph("1. The two house types in our dataset", s_h))
S.append(table([["House type", "Equipment", "Data available"],
                ["<b>Grid-only</b> (no PV)", "STEG smart meter + <b>second meter</b>", "Grid-side electrical values, and the energy seen by the second meter"],
                ["<b>PV + battery</b>", "STEG smart meter + <b>second meter</b> + hybrid inverter (SBU priority, zero export) + LFP battery with BMS + <b>roof reference cell</b>",
                 "+ solar production, battery state and temperature, inverter status, backup mode, transfer switch, irradiance, module temperature"]],
               [32 * mm, 70 * mm, 68 * mm]))

S.append(Paragraph("2. Data sources", s_h))
S.append(table([
    ["Source", "Why we use it", "What it gives"],
    ["<b>STEG smart meter</b> (Smart Grid project, supplied by Sagemcom)", "Installed by STEG in every house.",
     "Per-phase voltage, current, active and reactive power. Exact channels and recording interval (15 min expected) to confirm with STEG."],
    ["<b>STEG transformer supervision</b> (Smart Grid equipment)", "Shows what happens on the whole low-voltage network.",
     "Transformer loading %, P, Q, LV-side voltage and the power of the other feeders."],
    ["<b>Hybrid inverter</b> (PV + battery houses; a model on the STEG list of accepted inverters)",
     "The 2026 reference requires integration with supervision systems.",
     "PV power, AC output, status and alarm codes, backup mode, transfer switch; usually internal temperature."],
    ["<b>LFP battery with BMS</b> (PV + battery houses)",
     "The 2026 reference requires a BMS that monitors voltage, current, temperature, SOC and SOH (IEC 62619 / 62620 / 61427).",
     "SOC, battery voltage and current, cell temperature. Read through the inverter."],
    ["<b>Second meter</b> (added device, every house)",
     "An independent measurement of the same energy: the only way to see a bias, a drift or under-reporting of the house meter.",
     "Energy per 15 minutes (kWh). Class 1 (about ±1 %). Options in section 5."],
    ["<b>Roof reference cell</b> (added device, PV houses)",
     "Tells whether the sun was there: separates PV shading from a cloudy day. Costs about 50 to 200 EUR, Modbus output.",
     "Irradiance in the plane of the panels (about ±3 %) and module temperature (about ±0.5 °C)."],
    ["<b>Appliance sub-metering</b>", "Used in the simulation only.",
     "<b>Not available</b> from STEG and <b>not used</b> by the final model (removed after the leakage audit)."]],
    [42 * mm, 58 * mm, 70 * mm]))

S.append(KeepTogether([Paragraph("3. Data flow", s_h), diagram(), Spacer(1, 3),
                       Paragraph("STEG systems plus two devices added in the house. The model works on 15-minute values.", s_b)]))

S.append(Paragraph("4. Where each model input comes from", s_h))
S.append(Paragraph("Legend: " + " · ".join(src(k) for k in ["STEG METER", "STEG TRAFO", "STEG PV/BMS", "COMPUTED",
                                                          "ESTIMATED", "ADDED DEVICE", "NOT AVAILABLE"]) +
                   ". Column names are those of our dataset.", s_b))
S.append(Spacer(1, 4))
rows = [["Parameter (dataset columns)", "Source type", "Device / input", "How we obtain it"],
        ["<b>U_1..3</b>, <b>I_1..3</b>, <b>P_1..3</b>, <b>Q_1..3</b>", src("STEG METER"), "Smart meter", "Per-phase registers (P signed: import +)"],
        ["<b>S_1..3</b>, <b>PF_1..3</b>", src("COMPUTED"), "Smart meter", "S = U × I; PF = P / S"],
        ["<b>P_TOT, Q_TOT, S_TOT, PF_TOT</b>", src("COMPUTED"), "Smart meter", "Sum of phases; PF_TOT = P_TOT / S_TOT"],
        ["<b>grid_available</b>", src("COMPUTED"), "Smart meter / inverter", "1 if voltage above about 0.5 pu, else 0; confirmed by the inverter grid status"],
        ["<b>transformer_loading_percent</b>, <b>transformer_P_kw</b>, <b>transformer_Q_kvar</b>", src("STEG TRAFO"), "Transformer supervision", "Direct"],
        ["<b>transformer_U_pu</b>", src("STEG TRAFO"), "Transformer supervision",
         "LV-side voltage. Needed to tell a real undervoltage (transformer low too) from a faulty voltage sensor (only the house low)"],
        ["<b>other_feeders_P_kw</b>", src("STEG TRAFO"), "Transformer supervision / feeder-head meters", "Power of the customers outside the monitored group"],
        ["<b>group_P_kw</b>, <b>rest_of_group_kw</b>, <b>house_share_of_group</b>", src("COMPUTED"), "Transformer + house meter",
         "Transformer power minus other feeders; minus the house; house share of it"],
        ["<b>check_meter_energy_kwh</b>", src("ADDED DEVICE"), "Second meter", "Energy per 15 min, independent of the house meter"],
        ["<b>irradiance_w_m2</b>", src("ADDED DEVICE"), "Roof reference cell", "In the plane of the panels"],
        ["<b>module_temp_c</b>", src("ADDED DEVICE"), "Roof reference cell", "Sensor on the back of the module"],
        ["<b>ref_meter_lr_*</b>", src("COMPUTED"), "House meter + second meter",
         "Log ratio of the two energies over 1 h, 24 h and 7 days, its jump and its trend over 1, 3 and 7 days"],
        ["<b>ref_pv_ratio_*</b>", src("COMPUTED"), "Inverter + roof cell",
         "Measured PV power / theoretical PV power from irradiance, module temperature and panel size (only when the battery can absorb the PV)"],
        ["<b>hist_*</b>", src("COMPUTED"), "House meter history", "Power vs its usual level at the same time of day over the previous 14 days, and its trend"],
        ["<b>P_PV_W</b>, <b>E_PV_kWh</b>", src("STEG PV/BMS"), "Inverter", "Direct register; energy = sum of P_PV × Δt or inverter counter"],
        ["<b>battery_soc_percent</b>, <b>battery_voltage_v</b>, <b>battery_current_a</b>", src("STEG PV/BMS"), "BMS via inverter", "Direct (required by the STEG reference)"],
        ["<b>battery_charge_power_w / discharge_power_w</b>", src("COMPUTED"), "BMS", "V × I, split by the sign of I"],
        ["<b>inverter_status</b>, <b>backup_mode_active</b>, <b>transfer_switch_status</b>, <b>inverter_ac_output_w</b>", src("STEG PV/BMS"), "Hybrid inverter", "Status and power registers"],
        ["<b>backup_load_served_w</b>", src("COMPUTED"), "Inverter", "Inverter AC output while in backup mode"],
        ["<b>backup_load_demand_w</b>, <b>unmet_backup_demand_w</b>", src("ESTIMATED"), "History of the house",
         "Demand that is not served leaves no measurement: usual demand at the same hour and day type"],
        ["<b>hour_sin</b>, <b>hour_cos</b>, rolling statistics", src("COMPUTED"), "Time stamp + all channels",
         "Time of day only. No month or day-of-year feature, so the model cannot learn the season of the injected faults"],
        ["<b>app_*_W</b> (fridge, oven, water heater…)", src("NOT AVAILABLE"), "—", "Simulation only; not used by the final model"],
        ["Battery and inverter temperature", src("STEG PV/BMS"), "BMS / inverter", "Available; not used by the current model (future feature)"]]
S.append(table(rows, [44 * mm, 25 * mm, 32 * mm, 69 * mm]))

S.append(Paragraph("5. Detecting meter faults: why a second meter", s_h))
S.append(Paragraph("A meter that has a gain error (wrong current transformer ratio after maintenance, ageing) multiplies current, power and "
                   "energy by the <b>same factor</b>. The meter stays consistent with itself (U × I = S, same power factor), and the change "
                   "is smaller than the normal day-to-day variation of a house (guests, air conditioning, travel), so it cannot be seen from "
                   "that meter alone. We measured it: with one house and its transformer only, bias, drift and shading were not "
                   "detectable (validation PR-AUC 0.05 to 0.31) and were disabled. A second, independent meter gives the missing reference.", s_b))
S.append(Spacer(1, 4))
S.append(box("<b>Second meter: log ratio of the two energies, over 1 h, 24 h and 7 days</b><br/>"
             "≈ 0: normal (two class-1 meters differ by about ±2 %) · <b>jump</b>: current bias · <b>slow ramp</b>: drift · "
             "<b>large drop</b>: under-reporting or a disconnected current transformer",
             ParagraphStyle("bx", parent=s_b, alignment=TA_CENTER)))
S.append(Spacer(1, 4))
S.append(table([
    ["Where to put the second meter", "Independent of the house meter?", "Note"],
    ["Check meter in the panel, with its <b>own current transformers</b>", "Yes", "Simplest; Modbus or Wi-Fi output; a bypass upstream of both meters fools both"],
    ["The <b>hybrid inverter's own grid-input measurement</b> (PV houses with an SBU inverter)", "Yes (separate device)", "No extra hardware; PV houses only"],
    ["Meter <b>outside the house</b> (pole or street cabinet), out of the customer's reach", "Yes, even against fraud", "Used by utilities against bypass and tampering"],
    ["Transformer energy balance (transformer − sum of all meters)", "Yes, at group level", "Finds that energy is lost, not which house"]],
    [76 * mm, 38 * mm, 56 * mm]))
S.append(Spacer(1, 4))
S.append(Paragraph("<b>Roof reference cell.</b> Production is compared with the theoretical production from the measured irradiance and "
                   "module temperature. Sun present and production low: shading. Sun low and production low: clouds (the ratio stays "
                   "normal). The cell must sit in the plane of the panels and out of the shadows to detect, and it gets dirty like the "
                   "panels, so it must be cleaned more often than they are (soiling it shares would be hidden).", s_b))

S.append(Paragraph("6. Measured results", s_h))
S.append(Paragraph(f"All models are {_ALGO}, trained and tested on our simulated Tunisian dataset (13 houses, one 160 kVA transformer, "
                   "2018, 15 min, 14 anomaly types). Test set = 4 two-week blocks never used for training.", s_b))
S.append(Spacer(1, 4))


def _f1(r, l):
    return "not detected" if l in r.get("disabled_labels", []) else f"{r['per_label'][l]['f1']:.2f}"


def _o(r, k, fmt="{:.3f}"):
    return fmt.format(r["overall"][k])


S.append(table([
    ["", "13 houses (neighbours)", "One house + transformer", "One house + transformer + reference inputs (final)"],
    ["Inputs", "All houses of the transformer", "One house + transformer data", "One house + transformer data + second meter + roof sensor"],
    ["Macro F1 over the 14 anomaly types", _o(_R13, "macro_f1"), _o(_R0, "macro_f1"), "<b>" + _o(_R1, "macro_f1") + "</b>"],
    ["Detection recall (anomalous steps found)", _o(_R13, "detect_recall"), _o(_R0, "detect_recall"), "<b>" + _o(_R1, "detect_recall") + "</b>"],
    ["Event recall", _o(_R13, "event_recall_macro"), _o(_R0, "event_recall_macro"), "<b>" + _o(_R1, "event_recall_macro") + "</b>"],
    ["False alarms per house-day", _o(_R13, "false_alarms_per_house_day"), _o(_R0, "false_alarms_per_house_day"), _o(_R1, "false_alarms_per_house_day")],
    ["F1 PV partial shading", _f1(_R13, "pv_partial_shading"), _f1(_R0, "pv_partial_shading"), "<b>" + _f1(_R1, "pv_partial_shading") + "</b>"],
    ["F1 meter drift", _f1(_R13, "meter_drift"), _f1(_R0, "meter_drift"), "<b>" + _f1(_R1, "meter_drift") + "</b>"],
    ["F1 meter current bias", _f1(_R13, "meter_bias_current"), _f1(_R0, "meter_bias_current"), "<b>" + _f1(_R1, "meter_bias_current") + "</b>"],
    ["F1 falsified under-reporting", _f1(_R13, "falsified_under_report"), _f1(_R0, "falsified_under_report"), "<b>" + _f1(_R1, "falsified_under_report") + "</b>"]],
    [48 * mm, 32 * mm, 38 * mm, 52 * mm]))
S.append(Spacer(1, 5))
S.append(box("<b>Important.</b> These scores come from simulated data. The two reference inputs are simulated too: computed from the true "
             "values the simulator stored and degraded to an assumed accuracy (second meter ±1 %, irradiance about ±3 %, temperature "
             "±0.5 °C). The scores for bias, drift and shading hold only if real devices reach this accuracy, are independent of the house "
             "meter and are read at the same 15-minute timestamps. The 13-house column still uses information from the simulator "
             "(exact appliance sum) that the leakage audit removed from the one-house models.",
             s_b, bg=colors.HexColor("#FFF5F5"), border=colors.HexColor("#C62828")))

S.append(Paragraph("7. What changes in our model", s_h))
S.append(table([
    ["Simulation (today)", "Real deployment", "Status"],
    ["Transformer loading, P, Q, LV voltage and other feeders simulated", "Read from STEG transformer supervision", "<b>Ready</b>: the model already uses them"],
    ["One house analysed alone", "Single-house mode", "<b>Done</b>"],
    ["app_*_W exact, used as reference", "Not available", "<b>Done</b>: removed after the leakage audit; replaced by the house's own history, its share of the transformer group and the two reference inputs"],
    ["Second meter and roof sensor simulated from true values (±1 %, ±3 %, ±0.5 °C)", "Real devices to choose, install and read at 15 min", "<b>To do</b>: select the devices and validate their accuracy and independence"],
    ["Meter noise of a class 0.5 meter (≈0.3 V, ≈0.2 % on power)", "Accuracy class of the STEG meters", "<b>To do</b>: set the noise to the real meter class and retrain"],
    ["15-minute data", "STEG load-profile interval", "<b>To confirm</b> with STEG (model built for 15 min)"],
    ["No battery or inverter temperature", "Available from BMS and inverter", "<b>Future</b>: features for heat-related battery derating"]],
    [55 * mm, 52 * mm, 63 * mm]))

S.append(Paragraph("8. Limits and points to confirm", s_h))
for t in ["Which channels the smart meters record (per-phase U, I, P, Q or energy only) and at which interval.",
          "Whether the transformer supervision includes the LV-side voltage, not only the loading.",
          "How the PV inverter and BMS data reach STEG (supervision platform, interval, delay).",
          "Data access for EnergyMind: agreement and anonymisation rules for household data.",
          "The second meter must be a different device from the house meter. A bypass upstream of both meters fools both; "
          "a meter outside the house avoids it.",
          "The roof cell must be in the plane of the panels, out of the shadows to detect, and kept clean; it is one more device "
          "that can fail or drift and must itself be monitored.",
          "The accuracy of the two added devices is an assumption of the simulation (±1 %, ±3 %, ±0.5 °C) to validate on real devices.",
          "The inverter model must be checked against the current STEG list (lists have validity dates)."]:
    S.append(Paragraph(t, s_bul, bulletText="•"))

S.append(Paragraph("Sources", s_h))
for t in ["STEG, <i>Référentiel technique des installations photovoltaïques</i> with battery storage (July 2026), hosted by La Presse: "
          "lapresse.tn/wp-content/uploads/2026/07/Referentiel_technique_des_installations_PV.pdf",
          "STEG, <i>Liste des onduleurs acceptés par la STEG</i>: steg.com.tn/system/files/pdf/Liste_des_onduleurs_acceptes_par_la_STEG.pdf",
          "STEG, <i>Liste des micro-onduleurs acceptés</i> (V12, May 2026): steg.com.tn/system/files/pdf/Liste_des_micro_onduleurs_acceptes_par_la_STEG.pdf",
          "French Treasury, <i>Projet Smart-Grid de la STEG financé par l'AFD</i> (Feb 2025): tresor.economie.gouv.fr",
          "EnergyMind v3 results: models/single_house_refs/ (final), models/single_house/ (without references), models/single/ (13 houses); "
          "data/05_references/ (reference inputs)"]:
    S.append(Paragraph("– " + t, s_src))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                        title="EnergyMind Tunisia: real deployment", author="EnergyMind")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("ok", OUT)
