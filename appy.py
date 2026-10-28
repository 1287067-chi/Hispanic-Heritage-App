# Replace this section in app.py:
def main():
    st.set_page_config(
        page_title="Mexico & Migration Historical Data Explorer",
        page_icon="🇲🇽",
        layout="wide"
    )

    # ADD YOUR NAME HERE
    YOUR_NAME = Chinomso Ikwecheghe"  # <-- Replace with your actual name

    st.title("🇲🇽 Hispanic Heritage Data Lab: Mexico & Migration Focus")
    st.caption(f"Developed by **{Chinomso Ikwecheghe}** | Authentic World Bank Historical Data & Predictive Modeling")
import datetime
import io
import math
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

# ReportLab imports for PDF export
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------------------------------------------------------
# 1. CONSTANTS AND METADATA
# -----------------------------------------------------------------------------

CURRENT_YEAR = datetime.datetime.now().year
DEFAULT_START_YEAR = CURRENT_YEAR - 70
DEFAULT_END_YEAR = CURRENT_YEAR - 1

COUNTRIES = {
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "CO": "Colombia",
    "CL": "Chile",
    "PE": "Peru",
    "VE": "Venezuela",
    "EC": "Ecuador",
    "UY": "Uruguay",
}

INDICATORS = {
    "SP.DYN.LE00.IN": {
        "name": "Life Expectancy at Birth",
        "units": "Years",
        "min": 0,
        "max": 120,
        "definition": "Life expectancy at birth indicates the number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life.",
    },
    "SL.UEM.TOTL.NE.ZS": {
        "name": "Unemployment, Total (% of total labor force)",
        "units": "% of total labor force (national estimate)",
        "min": 0,
        "max": 100,
        "definition": "Unemployment refers to the share of the labor force that is without work but available for and seeking employment (national estimate).",
    },
    "NY.GDP.PCAP.CD": {
        "name": "GDP per Capita",
        "units": "Current US$",
        "min": 0,
        "max": 1000000,
        "definition": "Gross Domestic Product per capita is gross domestic product divided by midyear population. Note: This measures total economic output per person, not average personal wealth or household income.",
    },
    "SP.DYN.CBRT.IN": {
        "name": "Crude Birth Rate",
        "units": "Births per 1,000 people",
        "min": 0,
        "max": 100,
        "definition": "Crude birth rate indicates the number of live births occurring during the year, per 1,000 population at midyear. Note: This is a rate, not total birth count.",
    },
    "SM.POP.NETM": {
        "name": "Net Migration",
        "units": "Number of people",
        "min": -50000000,
        "max": 50000000,
        "definition": "Net migration is the net total of migrants during the period (immigrants minus emigrants). Note: This is net flow, not an emigration count alone.",
    },
    "VC.IHR.PSRC.P5": {
        "name": "Intentional Homicide Rate",
        "units": "Per 100,000 people",
        "min": 0,
        "max": 500,
        "definition": "Intentional homicides are estimates of unlawful deaths purposefully inflicted on a person by another person, expressed per 100,000 people.",
    },
    "FP.CPI.TOTL.ZG": {
        "name": "Consumer Price Inflation",
        "units": "Annual %",
        "min": -100,
        "max": 100000,
        "definition": "Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services.",
    },
    "GC.DOD.TOTL.GD.ZS": {
        "name": "Central Government Debt",
        "units": "% of GDP",
        "min": 0,
        "max": 1000,
        "definition": "Central government gross debt consists of all liabilities that require payment or payments of interest and/or principal by the debtor to the creditor at a date or dates in the future.",
    },
    "FI.RES.TOTL.CD": {
        "name": "Total Reserves (includes gold)",
        "units": "Current US$",
        "min": 0,
        "max": 10e13,
        "definition": "Total reserves comprise special drawing rights, reserves of IMF members, and holdings of foreign exchange under the control of monetary authorities, plus gold.",
    },
    "NE.RSB.GNFS.ZS": {
        "name": "External Balance on Goods and Services",
        "units": "% of GDP",
        "min": -100,
        "max": 100,
        "definition": "External balance on goods and services (formerly resource balance) equals exports of goods and services minus imports of goods and services, as a percentage of GDP.",
    },
}

# -----------------------------------------------------------------------------
# 2. DATA RETRIEVAL (WORLD BANK API)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_world_bank_data(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 1000
    }
    retrieval_date = datetime.date.today().strftime("%Y-%m-%d")
    source_url = f"https://data.worldbank.org/indicator/{indicator_code}?locations={country_code}"
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        
        if not json_data or len(json_data) < 2 or json_data[1] is None:
            return pd.DataFrame(columns=["Year", "Value"]), source_url, retrieval_date, "No data returned from API."
        
        records = []
        for entry in json_data[1]:
            val = entry.get("value")
            yr = entry.get("date")
            if val is not None and yr is not None:
                try:
                    records.append({"Year": int(yr), "Value": float(val)})
                except ValueError:
                    continue
                    
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("Year").reset_index(drop=True)
        return df, source_url, retrieval_date, None
    except Exception as e:
        return pd.DataFrame(columns=["Year", "Value"]), source_url, retrieval_date, f"API Error: {str(e)}"

# -----------------------------------------------------------------------------
# 3. MATHEMATICAL & POLYNOMIAL MODELING HELPERS
# -----------------------------------------------------------------------------

def fit_polynomial_model(df: pd.DataFrame, degree: int, holdout_size: int = 0):
    if df.empty or len(df) < degree + 1:
        return None
    
    df_sorted = df.sort_values("Year").copy()
    
    if holdout_size > 0 and len(df_sorted) > degree + 1 + holdout_size:
        train_df = df_sorted.iloc[:-holdout_size].copy()
        test_df = df_sorted.iloc[-holdout_size:].copy()
    else:
        train_df = df_sorted.copy()
        test_df = pd.DataFrame(columns=["Year", "Value"])

    # Centered time
    base_year = float(train_df["Year"].min())
    t_train = train_df["Year"].values - base_year
    y_train = train_df["Value"].values
    
    # Fit coefficients (highest degree first)
    coeffs = np.polyfit(t_train, y_train, degree)
    poly = np.poly1d(coeffs)
    
    # In-sample metrics
    y_train_pred = poly(t_train)
    r2_train = r2_score(y_train, y_train_pred) if len(y_train) > 1 else 1.0
    mae_train = mean_absolute_error(y_train, y_train_pred)
    residuals_train = y_train - y_train_pred
    
    # Holdout metrics
    holdout_r2, holdout_mae = None, None
    if not test_df.empty:
        t_test = test_df["Year"].values - base_year
        y_test = test_df["Value"].values
        y_test_pred = poly(t_test)
        holdout_mae = mean_absolute_error(y_test, y_test_pred)
        if len(y_test) > 1:
            holdout_r2 = r2_score(y_test, y_test_pred)

    return {
        "coeffs": coeffs,
        "poly": poly,
        "base_year": base_year,
        "degree": degree,
        "r2_train": r2_train,
        "mae_train": mae_train,
        "residuals_train": residuals_train,
        "train_df": train_df,
        "test_df": test_df,
        "holdout_r2": holdout_r2,
        "holdout_mae": holdout_mae,
        "obs_count": len(train_df),
        "min_year": train_df["Year"].min(),
        "max_year": train_df["Year"].max(),
        "mean_val": y_train.mean(),
    }

def predict_year(model, year: float):
    t = year - model["base_year"]
    est = float(model["poly"](t))
    min_y = model["min_year"]
    max_y = model["max_year"]
    
    is_observed_year = year in model["train_df"]["Year"].values
    is_in_range = (min_y <= year <= max_y)
    
    if is_in_range:
        classification = "Within Range (Interpolation)"
    else:
        classification = "Outside Range (Extrapolation)"
        
    return {
        "year": year,
        "estimate": est,
        "classification": classification,
        "is_observed_year": is_observed_year,
        "in_range": is_in_range
    }

def format_poly_equation(coeffs, base_year):
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        c_str = f"{c:+.6e}"
        if power == 0:
            terms.append(f"{c_str}")
        elif power == 1:
            terms.append(f"{c_str}*(t)")
        else:
            terms.append(f"{c_str}*(t)^{power}")
    eq_str = "y(t) = " + " ".join(terms) if terms else "y(t) = 0"
    eq_str += f", where t = Year - {int(base_year)}"
    return eq_str

# -----------------------------------------------------------------------------
# 4. REPORTLAB PDF GENERATION
# -----------------------------------------------------------------------------

def generate_pdf_report(country_name, indicator_meta, model, fig_buf, edited_flag, df_data, calc_results):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#1E3A8A'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=9, leading=12)
    meta_style = ParagraphStyle('Meta', parent=styles['Italic'], fontSize=8, leading=11, textColor=colors.gray)

    elements = []
    
    # Header
    elements.append(Paragraph("Hispanic Heritage Data Lab Report", title_style))
    elements.append(Paragraph(f"<b>Country:</b> {country_name} | <b>Indicator:</b> {indicator_meta['name']}", body_style))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | <b>Scenario:</b> {'User-Modified' if edited_flag else 'Official Source Data'}", meta_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1E3A8A'), spaceBefore=6, spaceAfter=10))

    # Model Summary
    elements.append(Paragraph("1. Model Specification & Metrics", h2_style))
    eq_str = format_poly_equation(model["coeffs"], model["base_year"])
    metrics_data = [
        ["Polynomial Degree", str(model["degree"]), "In-Sample R²", f"{model['r2_train']:.4f}"],
        ["Base Year (t=0)", str(int(model["base_year"])), "In-Sample MAE", f"{model['mae_train']:.4f} {indicator_meta['units']}"],
        ["Observations Count", str(model["obs_count"]), "Historical Mean", f"{model['mean_val']:.4f}"],
        ["Historical Range", f"{model['min_year']} - {model['max_year']}", "Holdout MAE", f"{model['holdout_mae']:.4f}" if model['holdout_mae'] is not None else "N/A"]
    ]
    t_metrics = Table(metrics_data, colWidths=[130, 130, 130, 130])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_metrics)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(f"<b>Fitted Equation:</b> <font name='Courier'>{eq_str}</font>", body_style))

    # Visualization
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("2. Visualization", h2_style))
    if fig_buf:
        img = Image(fig_buf, width=520, height=260)
        elements.append(img)

    # Educational Findings & Limitations
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("3. Analytical Findings & Limitations", h2_style))
    findings = f"""
    <b>Indicator Definition:</b> {indicator_meta['definition']}<br/>
    <b>Polynomial Modeling Behavior:</b> Fits centered time <i>t = Year - {int(model['base_year'])}</i> up to degree {model['degree']}. 
    Note that high-degree polynomials can display steep end-behavior polynomial divergence beyond observed data bounds. 
    Extrapolations outside {model['min_year']}–{model['max_year']} must be interpreted strictly as mathematical extensions, not demographic predictions.
    """
    elements.append(Paragraph(findings, body_style))

    # Single/Batch Calculated Results Table
    if calc_results:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("4. Calculated Year Estimates", h2_style))
        calc_rows = [["Year", f"Estimated Value ({indicator_meta['units']})", "Classification", "Observed Point?"]]
        for r in calc_results[:15]: # Limit to top 15 rows for neatness
            calc_rows.append([
                str(int(r["year"])),
                f"{r['estimate']:.4f}",
                r["classification"],
                "Yes" if r["is_observed_year"] else "No"
            ])
        t_calc = Table(calc_rows, colWidths=[80, 160, 180, 100])
        t_calc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        elements.append(t_calc)

    # Data Preview
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("5. Working Data Sample", h2_style))
    data_rows = [["Year", f"Value ({indicator_meta['units']})"]]
    for idx, row in df_data.head(10).iterrows():
        data_rows.append([str(int(row["Year"])), f"{row['Value']:.4f}"])
    t_data = Table(data_rows, colWidths=[150, 250])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E5E7EB')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('PADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_data)

    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer

# -----------------------------------------------------------------------------
# 5. MAIN STREAMLIT APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Hispanic Heritage Historical Data Explorer",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Hispanic Heritage Historical Data Explorer")
    st.caption("Authentic World Bank Data Analysis & Mathematical Polynomial Modeling")

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    st.sidebar.header("🕹️ Controls & Settings")
    
    selected_country_code = st.sidebar.selectbox(
        "Primary Country",
        options=list(COUNTRIES.keys()),
        format_func=lambda c: f"{COUNTRIES[c]} ({c})"
    )
    
    selected_indicator_code = st.sidebar.selectbox(
        "Indicator",
        options=list(INDICATORS.keys()),
        format_func=lambda i: f"{INDICATORS[i]['name']}"
    )

    indicator_meta = INDICATORS[selected_indicator_code]
    
    comp_country_code = st.sidebar.selectbox(
        "Comparison Country (Optional)",
        options=["None"] + [c for c in COUNTRIES.keys() if c != selected_country_code],
        format_func=lambda c: "None" if c == "None" else f"{COUNTRIES[c]} ({c})"
    )

    st.sidebar.markdown("---")
    poly_degree = st.sidebar.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3, step=1)
    graph_increment = st.sidebar.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1, step=1)
    forward_horizon = st.sidebar.slider("Forward Display Horizon (Years)", min_value=0, max_value=50, value=10, step=1)
    
    holdout_years = st.sidebar.number_input("Chronological Holdout Count (Validation)", min_value=0, max_value=15, value=0, step=1)
    
    printer_mode = st.sidebar.checkbox("🖨️ Printer-Friendly View Mode")

    # -------------------------------------------------------------------------
    # STATE MANAGEMENT & DATA RETRIEVAL
    # -------------------------------------------------------------------------
    state_key = f"{selected_country_code}_{selected_indicator_code}"
    
    if "current_state_key" not in st.session_state or st.session_state["current_state_key"] != state_key:
        st.session_state["current_state_key"] = state_key
        st.session_state["edited_data"] = None
        st.session_state["calc_results"] = None

    # Fetch official data
    raw_df, source_url, retrieval_date, fetch_error = fetch_world_bank_data(
        selected_country_code, selected_indicator_code, DEFAULT_START_YEAR, DEFAULT_END_YEAR
    )

    # Reset button handling
    if st.sidebar.button("🔄 Reset to Official Source Data"):
        st.session_state["edited_data"] = None
        st.session_state["calc_results"] = None
        st.rerun()

    # Determine working data
    if st.session_state["edited_data"] is not None:
        working_df = st.session_state["edited_data"].copy()
        is_user_modified = True
    else:
        working_df = raw_df.copy()
        is_user_modified = False

    # -------------------------------------------------------------------------
    # HEADER METADATA & DEFINITIONS
    # -------------------------------------------------------------------------
    if printer_mode:
        st.info("🖨️ **Printer-Friendly View Active**: Layout simplified for clean printing/exporting.")

    col_meta1, col_meta2 = st.columns([2, 1])
    with col_meta1:
        st.subheader(f"Country: {COUNTRIES[selected_country_code]} | Indicator: {indicator_meta['name']}")
        st.write(f"**Definition:** {indicator_meta['definition']}")
        st.write(f"**Units:** {indicator_meta['units']}")
    with col_meta2:
        st.write(f"**Source:** [World Bank API Link]({source_url})")
        st.write(f"**Retrieval Date:** {retrieval_date}")
        st.write(f"**Data Status:** {'⚠️ User-Modified Scenario' if is_user_modified else '✅ Authentic Source Data'}")
        st.write(f"**Actual Coverage:** {len(working_df)} valid observations ({DEFAULT_START_YEAR} to {DEFAULT_END_YEAR})")

    if fetch_error:
        st.error(f"Data Retrieval Notice: {fetch_error}")

    # -------------------------------------------------------------------------
    # EDITABLE DATA TABLE & VALIDATION
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Data Management & Validation Table")
    st.caption("You can edit values or years below. Modified tables will be flagged as a user-modified scenario.")

    edited_df_input = st.data_editor(
        working_df,
        num_rows="dynamic",
        key=f"editor_{state_key}",
        use_container_width=True
    )

    # Validate Edited Data
    validation_errors = []
    clean_df = pd.DataFrame(columns=["Year", "Value"])

    if edited_df_input is not None and not edited_df_input.empty:
        # Check non-empty
        try:
            temp_df = edited_df_input.dropna(subset=["Year", "Value"]).copy()
            # Validate types
            temp_df["Year"] = temp_df["Year"].astype(int)
            temp_df["Value"] = temp_df["Value"].astype(float)
            
            # Check duplicates
            if temp_df["Year"].duplicated().any():
                dups = temp_df[temp_df["Year"].duplicated()]["Year"].tolist()
                validation_errors.append(f"Duplicate years found: {dups}")
            
            # Check constraints
            out_of_bounds = temp_df[(temp_df["Value"] < indicator_meta["min"]) | (temp_df["Value"] > indicator_meta["max"])]
            if not out_of_bounds.empty:
                validation_errors.append(f"Values present outside valid indicator range [{indicator_meta['min']}, {indicator_meta['max']}]")
            
            if not validation_errors:
                clean_df = temp_df.sort_values("Year").reset_index(drop=True)
                # Check if changed from raw
                if not clean_df.equals(raw_df):
                    st.session_state["edited_data"] = clean_df
                    is_user_modified = True
        except Exception as e:
            validation_errors.append(f"Data format error: {str(e)}")

    if validation_errors:
        for err in validation_errors:
            st.error(f"❌ Validation Error: {err}")

    # -------------------------------------------------------------------------
    # MODEL FITTING
    # -------------------------------------------------------------------------
    model = None
    if len(clean_df) >= poly_degree + 1:
        model = fit_polynomial_model(clean_df, poly_degree, holdout_size=holdout_years)
    else:
        st.warning(f"Insufficient observations ({len(clean_df)}) to fit a polynomial of degree {poly_degree}. Requires at least {poly_degree + 1} distinct data points.")

    # -------------------------------------------------------------------------
    # MODEL METRICS & MATHEMATICAL ANALYSIS
    # -------------------------------------------------------------------------
    if model:
        st.markdown("---")
        st.subheader("📐 Model Fit & In-Sample Metrics")
        
        eq_text = format_poly_equation(model["coeffs"], model["base_year"])
        st.code(eq_text, language="text")

        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Degree (d)", model["degree"])
        m_col2.metric("In-Sample R²", f"{model['r2_train']:.4f}")
        m_col3.metric("In-Sample MAE", f"{model['mae_train']:.4f}")
        m_col4.metric("Observations", model["obs_count"])
        m_col5.metric("Historical Mean", f"{model['mean_val']:.2f}")

        if model["holdout_mae"] is not None:
            st.info(f"📊 **Chronological Holdout Validation ({holdout_years} years):** Holdout MAE = {model['holdout_mae']:.4f}" + (f" | Holdout R² = {model['holdout_r2']:.4f}" if model['holdout_r2'] is not None else ""))

        # Curricular Tabs
        tab_alg2, tab_precalc, tab_calc = st.tabs(["Algebra 2 Focus", "Precalculus Focus", "Calculus Focus"])

        with tab_alg2:
            st.markdown("#### Algebra 2 Concepts")
            st.write(f"- **Domain (Historical):** [{model['min_year']}, {model['max_year']}]")
            st.write(f"- **Range (In-Sample Fitted):** [{min(model['poly'](model['train_df']['Year']-model['base_year'])):.2f}, {max(model['poly'](model['train_df']['Year']-model['base_year'])):.2f}]")
            
            # End behavior
            lead_coeff = model["coeffs"][0]
            deg_even = (model["degree"] % 2 == 0)
            if deg_even:
                eb = "y → +∞ as t → ±∞" if lead_coeff > 0 else "y → -∞ as t → ±∞"
            else:
                eb = "y → +∞ as t → +∞, y → -∞ as t → -∞" if lead_coeff > 0 else "y → -∞ as t → +∞, y → +∞ as t → -∞"
            st.write(f"- **End Behavior:** Leading coefficient = `{lead_coeff:.3e}` (Degree {model['degree']}). As {eb}")
            
            # Average rate of change
            first_y = model["poly"](model["min_year"] - model["base_year"])
            last_y = model["poly"](model["max_year"] - model["base_year"])
            avg_rate = (last_y - first_y) / (model["max_year"] - model["min_year"])
            st.write(f"- **Overall Average Rate of Change:** {avg_rate:.4f} {indicator_meta['units']}/year across historical range.")

        with tab_precalc:
            st.markdown("#### Precalculus Concepts")
            p_deriv = np.polyder(model["poly"])
            roots = np.roots(p_deriv)
            # Real roots in domain
            real_roots = [r.real for r in roots if np.isreal(r)]
            real_turning = [r + model["base_year"] for r in real_roots if model["min_year"] <= (r + model["base_year"]) <= model["max_year"]]
            st.write(f"- **Degree & Max Turning Points:** Degree {model['degree']} polynomial allows up to {model['degree'] - 1} turning points.")
            st.write(f"- **In-Domain Turning Points (Years):** {', '.join([f'{r:.1f}' for r in real_turning]) if real_turning else 'None within historical range'}")
            
            # Concavity changes
            p_deriv2 = np.polyder(model["poly"], 2)
            roots2 = np.roots(p_deriv2)
            real_roots2 = [r.real for r in roots2 if np.isreal(r)]
            inflections = [r + model["base_year"] for r in real_roots2 if model["min_year"] <= (r + model["base_year"]) <= model["max_year"]]
            st.write(f"- **In-Domain Inflection Point Years:** {', '.join([f'{r:.1f}' for r in inflections]) if inflections else 'None within historical range'}")

        with tab_calc:
            st.markdown("#### Calculus Concepts")
            p_d1 = np.polyder(model["poly"], 1)
            p_d2 = np.polyder(model["poly"], 2)
            
            st.write("**First Derivative:** $y'(t)$ governs velocity/rate of change.")
            st.write("**Second Derivative:** $y''(t)$ governs acceleration/concavity.")
            
            # Extrema classification
            crit_pts = [r.real for r in np.roots(p_d1) if np.isreal(r)]
            crit_in_domain = [r for r in crit_pts if 0 <= r <= (model["max_year"] - model["base_year"])]
            
            if crit_in_domain:
                st.write("**Classified Critical Points in Domain:**")
                for cp in crit_in_domain:
                    val2 = p_d2(cp)
                    c_type = "Local Minimum" if val2 > 0 else ("Local Maximum" if val2 < 0 else "Inconclusive")
                    st.write(f"  • Year {cp + model['base_year']:.2f}: {c_type} (f''(t) = {val2:.3e})")
            else:
                st.write("No critical points found inside the historical domain.")

    # -------------------------------------------------------------------------
    # SINGLE & BATCH CALCULATED ESTIMATES
    # -------------------------------------------------------------------------
    calc_list = []
    if model:
        st.markdown("---")
        st.subheader("🔢 Model Estimator (Single & Batch Calculation)")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            st.markdown("##### Single-Year Input")
            single_year_val = st.number_input("Target Year (1950 - 2100)", min_value=1950, max_value=2100, value=2025, step=1)
            if st.button("Calculate Value"):
                res = predict_year(model, single_year_val)
                st.session_state["calc_results"] = [res]

        with calc_col2:
            st.markdown("##### Batch-Year Input")
            batch_mode = st.radio("Batch Method", ["Range", "Custom List"], horizontal=True)
            
            batch_years = []
            if batch_mode == "Range":
                r_col1, r_col2, r_col3 = st.columns(3)
                b_start = r_col1.number_input("Start Year", 1950, 2100, 2020)
                b_end = r_col2.number_input("End Year", 1950, 2100, 2030)
                b_step = r_col3.slider("Step", 1, 10, 1)
                if b_start <= b_end:
                    batch_years = list(range(int(b_start), int(b_end) + 1, int(b_step)))
                else:
                    st.error("Start Year must be <= End Year.")
            else:
                custom_str = st.text_input("Comma-Separated Years", "1990, 2000, 2010, 2025, 2030")
                try:
                    batch_years = [int(y.strip()) for y in custom_str.split(",") if y.strip().isdigit()]
                except Exception:
                    st.error("Invalid comma-separated list of years.")
            
            if st.button("Calculate Batch Values"):
                res_list = [predict_year(model, y) for y in batch_years]
                st.session_state["calc_results"] = res_list

        # Display Calculated Results Table
        if st.session_state.get("calc_results"):
            calc_list = st.session_state["calc_results"]
            st.markdown("##### Calculation Results Table")
            calc_df = pd.DataFrame(calc_list)
            
            # Format display
            calc_disp = pd.DataFrame({
                "Year": calc_df["year"].astype(int),
                f"Estimated Value ({indicator_meta['units']})": calc_df["estimate"].map(lambda x: f"{x:.4f}"),
                "Classification": calc_df["classification"],
                "Is Observed Historical Point?": calc_df["is_observed_year"].map(lambda x: "Yes" if x else "No")
            })
            st.dataframe(calc_disp, use_container_width=True)
            
            # Implausibility Flag Check
            implausible = [r for r in calc_list if r["estimate"] < indicator_meta["min"] or r["estimate"] > indicator_meta["max"]]
            if implausible:
                st.warning(f"⚠️ **Polynomial Extrapolation Alert:** {len(implausible)} calculated estimate(s) exceed physical/realistic bounds [{indicator_meta['min']}, {indicator_meta['max']}]. Higher-degree polynomials diverge rapidly beyond historical bounds.")

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    fig_buf = None
    if model:
        st.markdown("---")
        st.subheader("📈 Interactive Visualization & Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot train data
        sample_train = model["train_df"].iloc[::graph_increment]
        ax.scatter(sample_train["Year"], sample_train["Value"], color="#1E3A8A", label=f"Observed ({COUNTRIES[selected_country_code]})", zorder=4, s=35)
        
        # Plot smooth fitted curve
        min_plot_yr = model["min_year"]
        max_plot_yr = model["max_year"] + forward_horizon
        plot_years = np.linspace(min_plot_yr, max_plot_yr, 300)
        plot_t = plot_years - model["base_year"]
        plot_preds = model["poly"](plot_t)
        
        # Split historical vs extrapolation for line style
        hist_mask = plot_years <= model["max_year"]
        ax.plot(plot_years[hist_mask], plot_preds[hist_mask], color="#2563EB", linewidth=2, label="Fitted Polynomial (Historical)")
        if forward_horizon > 0:
            ax.plot(plot_years[~hist_mask], plot_preds[~hist_mask], color="#DC2626", linestyle="--", linewidth=2, label="Extrapolated Segment")
            
        # Draw historical boundary
        ax.axvline(model["max_year"], color="gray", linestyle=":", alpha=0.7, label="Historical Boundary")
        
        # Plot Calculated Single/Batch Estimates
        if calc_list:
            c_years = [r["year"] for r in calc_list]
            c_ests = [r["estimate"] for r in calc_list]
            ax.scatter(c_years, c_ests, color="#D97706", marker="X", s=70, label="Calculated Point Estimates", zorder=5)

        # Plot Comparison Country Data if selected
        if comp_country_code != "None":
            comp_df, _, _, _ = fetch_world_bank_data(comp_country_code, selected_indicator_code, DEFAULT_START_YEAR, DEFAULT_END_YEAR)
            if not comp_df.empty:
                ax.plot(comp_df["Year"], comp_df["Value"], color="#059669", linestyle="-.", label=f"Comparison: {COUNTRIES[comp_country_code]}")

        ax.set_title(f"{indicator_meta['name']} - {COUNTRIES[selected_country_code]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(indicator_meta["units"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        
        st.pyplot(fig)
        
        # Save figure buffer for PDF report
        fig_buf = io.BytesIO()
        fig.savefig(fig_buf, format="png", bbox_inches="tight", dpi=150)
        fig_buf.seek(0)

    # -------------------------------------------------------------------------
    # EXPORTS (CSV, EXCEL, PDF)
    # -------------------------------------------------------------------------
    if model:
        st.markdown("---")
        st.subheader("📥 Export & Download Center")
        
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        
        # 1. CSV Export
        csv_data = clean_df.to_csv(index=False).encode('utf-8')
        ex_col1.download_button(
            label="📄 Download Working Data (CSV)",
            data=csv_data,
            file_name=f"{selected_country_code}_{selected_indicator_code}_data.csv",
            mime="text/csv"
        )
        
        # 2. Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([
                {"Metric": "Country", "Value": COUNTRIES[selected_country_code]},
                {"Metric": "Indicator", "Value": indicator_meta["name"]},
                {"Metric": "Polynomial Degree", "Value": model["degree"]},
                {"Metric": "Base Year", "Value": int(model["base_year"])},
                {"Metric": "In-Sample R2", "Value": model["r2_train"]},
                {"Metric": "In-Sample MAE", "Value": model["mae_train"]},
                {"Metric": "Data Scenario", "Value": "User-Modified" if is_user_modified else "Official World Bank Source"}
            ])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            # Data sheet
            clean_df.to_excel(writer, sheet_name="Working Data", index=False)
            
            # Results sheet
            if calc_list:
                pd.DataFrame(calc_list).to_excel(writer, sheet_name="Calculated Results", index=False)
                
        excel_buffer.seek(0)
        ex_col2.download_button(
            label="📊 Download Full Report (Excel .xlsx)",
            data=excel_buffer,
            file_name=f"{selected_country_code}_{selected_indicator_code}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 3. PDF Export
        pdf_buf = generate_pdf_report(
            COUNTRIES[selected_country_code],
            indicator_meta,
            model,
            fig_buf,
            is_user_modified,
            clean_df,
            calc_list
        )
        ex_col3.download_button(
            label="📑 Download PDF Summary Report",
            data=pdf_buf,
            file_name=f"{selected_country_code}_{selected_indicator_code}_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
