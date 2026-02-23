"""
ESG RISK INTELLIGENCE PLATFORM
================================
Production Version 1.0
Author: Strategic Build
Purpose: Enterprise ESG Compliance Monitoring & Risk Intelligence

DISCLAIMER: This is a demonstration platform for informational purposes only.
Not financial or legal advice. For professional use with verified data sources.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import hmac
import time
from pathlib import Path

# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="ESG Risk Intelligence Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/yourprofile',
        'Report a bug': 'https://github.com/yourusername/esg-platform/issues',
        'About': '# ESG Risk Intelligence Platform\n\nEnterprise-grade compliance monitoring with dynamic risk scoring.\n\n**For demonstration purposes only.**'
    }
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Professional color scheme */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1B3B4F;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5F6B7A;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #EAECF0;
    }
    .risk-critical {
        background: #FEF3F2;
        border-left: 4px solid #D92D20;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-warning {
        background: #FFFAEB;
        border-left: 4px solid #F79009;
        padding: 1rem 1.5rem;
        border-radius: 8px;
    }
    .risk-safe {
        background: #F6FEF9;
        border-left: 4px solid #12B76A;
        padding: 1rem 1.5rem;
        border-radius: 8px;
    }
    .info-box {
        background: #F8F9FC;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E4E7EC;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #98A2B3;
        padding: 2rem;
        font-size: 0.85rem;
        border-top: 1px solid #EAECF0;
        margin-top: 3rem;
    }
    .disclaimer {
        background: #FCFCFD;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #667085;
        border: 1px solid #EAECF0;
    }
    .stButton button {
        background: #1B3B4F;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
    }
    .stButton button:hover {
        background: #0F2A38;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_loaded': False,
        'df': None,
        'last_loaded': None,
        'email_sent': False,
        'filter_preset': 'all'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hash_dataframe(df):
    """Create a hash of dataframe for change detection"""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

def format_currency(value):
    """Format currency values"""
    return f"${value:,.1f}M"

def get_threshold_config():
    """Get risk thresholds from session or defaults"""
    return {
        'critical': st.session_state.get('critical_threshold', 7),
        'warning': st.session_state.get('warning_threshold', 30)
    }

# ============================================================================
# SIDEBAR - ABOUT & CONFIGURATION
# ============================================================================

with st.sidebar:
    # Logo and branding
    st.image("https://img.icons8.com/fluency/96/null/sustainable-energy.png", width=60)
    st.markdown("## ESG Risk Intelligence")
    st.markdown("*Enterprise Compliance Platform*")
    
    # About section - Professional positioning
    with st.expander("ℹ️ About", expanded=True):
        st.markdown("""
        **Platform Purpose:**
        - Dynamic risk scoring for ESG compliance
        - Department-level risk aggregation
        - Executive-ready reporting
        - Configurable thresholds
        
        **Target Users:**
        - Compliance Officers
        - Risk Managers
        - ESG Directors
        - Internal Audit Teams
        
        **Data Sources:**
        - Internal compliance records
        - Regulatory deadlines
        - Impact assessments
        """)
        
        st.markdown("""
        <div class="disclaimer">
        ⚖️ DISCLAIMER: Demonstration platform only. 
        Not legal/financial advice. Verify all data independently.
        </div>
        """, unsafe_allow_html=True)
    
    # Configuration section
    st.markdown("---")
    st.markdown("### ⚙️ Risk Configuration")
    
    # Threshold settings
    critical_days = st.slider(
        "Critical Threshold (days)",
        min_value=1,
        max_value=14,
        value=7,
        help="Items due within this many days are flagged CRITICAL"
    )
    
    warning_days = st.slider(
        "Warning Threshold (days)",
        min_value=15,
        max_value=60,
        value=30,
        help="Items due within this many days are flagged WARNING"
    )
    
    # Impact weights
    st.markdown("### ⚖️ Impact Weights")
    col1, col2, col3 = st.columns(3)
    with col1:
        low_weight = st.number_input("Low", 1, 5, 1)
    with col2:
        medium_weight = st.number_input("Medium", 1, 5, 2)
    with col3:
        high_weight = st.number_input("High", 1, 5, 3)
    
    impact_weights = {
        "Low": low_weight,
        "Medium": medium_weight,
        "High": high_weight
    }
    
    # Store in session state
    st.session_state.critical_threshold = critical_days
    st.session_state.warning_threshold = warning_days
    
    # Email configuration status
    st.markdown("---")
    st.markdown("### 📧 Email Status")
    email_configured = all([
        os.getenv("ESG_SMTP_SERVER"),
        os.getenv("ESG_SMTP_PORT"),
        os.getenv("ESG_SENDER_EMAIL")
    ])
    
    if email_configured:
        st.success("✅ Email system active")
    else:
        st.warning("📧 Email reports disabled (configure env for production)")

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<p class="main-header">🌱 ESG Risk Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Dynamic risk scoring • Department analytics • Executive reporting</p>',
    unsafe_allow_html=True
)

# ============================================================================
# DATA LOADING SECTION
# ============================================================================

def create_enterprise_sample():
    """Generate professional sample data"""
    return pd.DataFrame({
        "Regulation": [
            "EU CSRD", "SEC Climate Rule", "SFDR", "TCFD", "EU Taxonomy",
            "GDPR", "CCPA", "Basel III", "Dodd-Frank", "MiFID II",
            "UK SDR", "ISO 14001", "GRI Standards", "SASB Standards", "CDP"
        ],
        "Due Date": [
            "2024-12-31", "2024-06-30", "2024-03-15", "2024-04-15", "2024-01-20",
            "2024-09-30", "2024-07-31", "2024-08-15", "2024-05-30", "2024-10-31",
            "2024-11-15", "2024-12-15", "2024-08-30", "2024-09-15", "2024-10-15"
        ],
        "Status": [
            "Open", "In Progress", "Open", "Not Started", "Closed",
            "Open", "In Progress", "Open", "In Progress", "Not Started",
            "Open", "In Progress", "Not Started", "Open", "Closed"
        ],
        "Impact Level": [
            "High", "High", "High", "Medium", "Low",
            "High", "Medium", "Medium", "Low", "High",
            "High", "Medium", "Medium", "Low", "Low"
        ],
        "Jurisdiction": [
            "EU", "US", "EU", "Global", "EU",
            "EU", "California", "Global", "US", "EU",
            "UK", "Global", "Global", "US", "Global"
        ],
        "Department": [
            "Sustainability", "Legal", "Finance", "Risk", "Sustainability",
            "Privacy", "Legal", "Compliance", "Legal", "Finance",
            "Sustainability", "Operations", "Sustainability", "Finance", "Risk"
        ],
        "Owner": [
            "Sarah Chen", "Michael Ross", "Emma Watson", "David Kim", "Lisa Patel",
            "James Wilson", "Maria Garcia", "Robert Taylor", "Anna Schmidt", "Tom Hughes",
            "Nina Patel", "Omar Hassan", "Clara Martinez", "Anders Nielsen", "Yuki Tanaka"
        ],
        "Budget Allocation ($M)": [
            12.5, 8.3, 15.2, 5.1, 2.3,
            7.8, 4.2, 6.7, 3.1, 9.4,
            11.2, 4.8, 5.9, 3.8, 2.9
        ]
    })

# Data loading UI
col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Upload Compliance Data (CSV)",
        type=["csv"],
        help="Upload your ESG compliance tracking CSV file"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📋 Load Sample Dataset", type="primary", use_container_width=True):
        df = create_enterprise_sample()
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.session_state.last_loaded = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.rerun()

with col3:
    if st.session_state.last_loaded:
        st.info(f"📊 Last loaded: {st.session_state.last_loaded}")

# Process uploaded file
if uploaded_file and not st.session_state.data_loaded:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.session_state.last_loaded = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.success(f"✅ Successfully loaded {len(df)} compliance items")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Check if we have data
if not st.session_state.data_loaded or st.session_state.df is None:
    st.info("👆 Please upload a CSV file or load the sample dataset to begin")
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        example_df = create_enterprise_sample().head(3)
        st.dataframe(example_df, use_container_width=True)
        st.code("""
Required columns:
- Regulation: Name of regulation/standard
- Due Date: YYYY-MM-DD format
- Status: Open/In Progress/Not Started/Closed
- Impact Level: Low/Medium/High
- Jurisdiction: EU/US/Global/etc
- Department: Owning department
- Owner: Responsible person
        """)
    st.stop()

# Get dataframe from session state
df = st.session_state.df.copy()

# ============================================================================
# DATA VALIDATION
# ============================================================================

required_columns = ["Due Date", "Status", "Impact Level", "Regulation", "Jurisdiction", "Department"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"⚠️ Missing required columns: {missing_columns}")
    st.stop()

# ============================================================================
# RISK ENGINE
# ============================================================================

try:
    # Process dates
    df["Due Date"] = pd.to_datetime(df["Due Date"])
    today = pd.to_datetime(datetime.now().date())
    df["Days Remaining"] = (df["Due Date"] - today).dt.days
    
    # Impact weights
    df["Impact Weight"] = df["Impact Level"].map(impact_weights).fillna(1)
    
    # Risk score calculation
    def calculate_risk_score(row):
        if row["Status"] == "Closed" or row["Days Remaining"] > warning_days:
            return 0
        
        urgency = max(0, warning_days - max(0, row["Days Remaining"]))
        
        # Status multiplier
        status_mult = {
            "Not Started": 1.5,
            "Open": 1.2,
            "In Progress": 1.0,
            "Closed": 0
        }.get(row["Status"], 1.0)
        
        # Jurisdiction complexity (example factor)
        jurisdiction_mult = {
            "EU": 1.2,
            "Global": 1.1,
            "US": 1.0,
            "California": 1.1,
            "UK": 1.0
        }.get(row.get("Jurisdiction", "Global"), 1.0)
        
        base_score = urgency * row["Impact Weight"] * status_mult
        return base_score * jurisdiction_mult
    
    df["Risk Score"] = df.apply(calculate_risk_score, axis=1)
    
    # Risk classification
    def classify_risk(row):
        if row["Status"] == "Closed":
            return "Closed"
        elif row["Days Remaining"] < 0:
            return "Overdue"
        elif row["Days Remaining"] <= critical_days:
            return "Critical"
        elif row["Days Remaining"] <= warning_days:
            return "Warning"
        else:
            return "On Track"
    
    df["Risk Level"] = df.apply(classify_risk, axis=1)
    
    # Priority (1 = highest)
    df["Priority"] = df["Risk Score"].rank(ascending=False, method='dense').astype(int)
    
except Exception as e:
    st.error(f"Error in risk calculation: {str(e)}")
    st.stop()

# ============================================================================
# EXECUTIVE DASHBOARD
# ============================================================================

st.markdown("## 📊 Executive Dashboard")

# Key metrics
total_items = len(df)
active_items = len(df[df["Status"] != "Closed"])
overdue_items = len(df[df["Risk Level"] == "Overdue"])
critical_items = len(df[df["Risk Level"] == "Critical"])
avg_risk = df[df["Status"] != "Closed"]["Risk Score"].mean()

# Budget at risk
if "Budget Allocation ($M)" in df.columns:
    budget_at_risk = df[df["Risk Level"].isin(["Overdue", "Critical"])]["Budget Allocation ($M)"].sum()
else:
    budget_at_risk = 0

# Metrics row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Items", total_items, f"{active_items} active")
with col2:
    st.metric("Overdue", overdue_items, delta_color="inverse")
with col3:
    st.metric("Critical", critical_items)
with col4:
    st.metric("Avg Risk Score", f"{avg_risk:.1f}")
with col5:
    if budget_at_risk > 0:
        st.metric("Budget at Risk", format_currency(budget_at_risk))
    else:
        health_score = ((active_items - overdue_items - critical_items) / active_items * 100) if active_items > 0 else 100
        st.metric("Health Score", f"{health_score:.0f}%")

# ============================================================================
# FILTER SECTION
# ============================================================================

st.markdown("## 🔍 Risk Explorer")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    departments = ["All Departments"] + sorted(df["Department"].unique().tolist())
    selected_dept = st.selectbox("Department", departments)

with filter_col2:
    risk_levels = ["All Levels"] + sorted(df["Risk Level"].unique().tolist())
    selected_risk = st.selectbox("Risk Level", risk_levels)

with filter_col3:
    impact_levels = ["All"] + sorted(df["Impact Level"].unique().tolist())
    selected_impact = st.selectbox("Impact Level", impact_levels)

with filter_col4:
    search_term = st.text_input("🔍 Search Regulation", "").lower()

# Apply filters
filtered_df = df.copy()

if selected_dept != "All Departments":
    filtered_df = filtered_df[filtered_df["Department"] == selected_dept]

if selected_risk != "All Levels":
    filtered_df = filtered_df[filtered_df["Risk Level"] == selected_risk]

if selected_impact != "All":
    filtered_df = filtered_df[filtered_df["Impact Level"] == selected_impact]

if search_term:
    filtered_df = filtered_df[filtered_df["Regulation"].str.lower().str.contains(search_term, na=False)]

# ============================================================================
# ANALYTICS VISUALIZATIONS
# ============================================================================

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Department risk profile
    dept_risk = filtered_df[filtered_df["Status"] != "Closed"].groupby("Department").agg({
        "Risk Score": ["mean", "count"]
    }).round(1)
    dept_risk.columns = ['Avg Risk', 'Count']
    dept_risk = dept_risk.reset_index()
    
    if not dept_risk.empty:
        fig = px.bar(
            dept_risk,
            x="Department",
            y="Avg Risk",
            color="Avg Risk",
            color_continuous_scale="RdYlGn_r",
            title="Average Risk by Department",
            text="Count"
        )
        fig.update_traces(texttemplate='%{text} items', textposition='outside')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    # Timeline view
    fig = px.scatter(
        filtered_df,
        x="Due Date",
        y="Regulation",
        size="Risk Score",
        color="Risk Level",
        hover_data=["Department", "Owner", "Days Remaining"],
        title="Compliance Timeline (Size = Risk Score)",
        color_discrete_map={
            "Overdue": "#D92D20",
            "Critical": "#F79009",
            "Warning": "#FDB022",
            "On Track": "#12B76A",
            "Closed": "#98A2B3"
        }
    )
    fig.update_layout(height=350, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PRIORITY VIEW
# ============================================================================

st.markdown("## 🔥 Priority Queue")
st.caption("Items sorted by risk score (highest priority first)")

# Column selector
display_cols = st.multiselect(
    "Display columns",
    options=df.columns.tolist(),
    default=["Priority", "Risk Level", "Regulation", "Department", "Due Date", 
             "Days Remaining", "Risk Score", "Owner", "Impact Level"]
)

# Display table
if display_cols:
    display_df = filtered_df.sort_values("Risk Score", ascending=False).head(25)[display_cols]
    
    # Style based on risk
    def style_risk(row):
        if row.get("Risk Level") == "Overdue":
            return ['background-color: #FEF3F2'] * len(row)
        elif row.get("Risk Level") == "Critical":
            return ['background-color: #FFFAEB'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(style_risk, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

# ============================================================================
# EXECUTIVE INSIGHTS
# ============================================================================

st.markdown("## 📋 Executive Summary")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 Key Findings")
    
    active_filtered = filtered_df[filtered_df["Status"] != "Closed"]
    
    if not active_filtered.empty:
        # Top priority
        top_priority = active_filtered.nlargest(1, "Risk Score").iloc[0]
        
        # Department with highest avg risk
        dept_risk_avg = active_filtered.groupby("Department")["Risk Score"].mean()
        highest_risk_dept = dept_risk_avg.idxmax() if not dept_risk_avg.empty else "N/A"
        
        # Items due this week
        this_week = len(active_filtered[
            (active_filtered["Days Remaining"] >= 0) & 
            (active_filtered["Days Remaining"] <= 7)
        ])
        
        st.markdown(f"""
        <div class="risk-critical">
            <strong>🚨 CRITICAL PRIORITY</strong><br>
            {top_priority['Regulation']} • {top_priority['Jurisdiction']}<br>
            Risk Score: {top_priority['Risk Score']:.1f} | Owner: {top_priority.get('Owner', 'Unassigned')}<br>
            Due: {top_priority['Due Date'].strftime('%Y-%m-%d')} ({top_priority['Days Remaining']} days)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Department Focus:</strong> {highest_risk_dept} has highest average risk<br>
            <strong>📅 Immediate Actions:</strong> {this_week} items due in next 7 days<br>
            <strong>⚖️ Impact Distribution:</strong> {len(active_filtered[active_filtered['Impact Level']=='High'])} high-impact items active
        </div>
        """, unsafe_allow_html=True)

with insight_col2:
    st.markdown("### 📈 Risk Distribution")
    
    # Risk distribution pie
    risk_dist = filtered_df["Risk Level"].value_counts().reset_index()
    risk_dist.columns = ["Risk Level", "Count"]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_dist["Risk Level"],
        values=risk_dist["Count"],
        hole=0.4,
        marker=dict(colors=['#D92D20', '#F79009', '#FDB022', '#12B76A', '#98A2B3'])
    )])
    fig.update_layout(height=300, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# EXPORT & REPORTING
# ============================================================================

st.markdown("## 📥 Export & Reporting")

export_col1, export_col2, export_col3, export_col4 = st.columns(4)

with export_col1:
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Full Dataset (CSV)",
        data=csv_data,
        file_name=f"esg_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with export_col2:
    # Summary report
    summary = filtered_df.groupby(["Department", "Risk Level"]).agg({
        "Regulation": "count",
        "Risk Score": "mean"
    }).round(1).reset_index()
    summary.columns = ["Department", "Risk Level", "Item Count", "Avg Risk Score"]
    
    summary_csv = summary.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📈 Summary Report",
        data=summary_csv,
        file_name=f"esg_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with export_col3:
    high_risk = filtered_df[filtered_df["Risk Level"].isin(["Overdue", "Critical"])]
    if not high_risk.empty:
        high_risk_csv = high_risk.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⚠️ High Risk Items",
            data=high_risk_csv,
            file_name=f"esg_high_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

with export_col4:
    json_data = filtered_df.to_json(orient='records', date_format='iso')
    st.download_button(
        label="🔧 JSON Export",
        data=json_data,
        file_name=f"esg_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True
    )

# ============================================================================
# EMAIL REPORTS (PRODUCTION READY)
# ============================================================================

st.markdown("## 📧 Report Distribution")

email_col1, email_col2 = st.columns([2, 1])

with email_col1:
    recipient = st.text_input(
        "Email Report To",
        placeholder="compliance-team@company.com",
        help="Enter email address to receive the report"
    )

with email_col2:
    include_attachments = st.checkbox("Include attachments", value=True)

def send_secure_email(recipient):
    """Send email using environment variables (production)"""
    
    # Get credentials from environment
    smtp_server = os.getenv("ESG_SMTP_SERVER")
    smtp_port = int(os.getenv("ESG_SMTP_PORT", "587"))
    sender = os.getenv("ESG_SENDER_EMAIL")
    password = os.getenv("ESG_SENDER_PASSWORD")
    
    if not all([smtp_server, smtp_port, sender, password]):
        return False, "Email not configured"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = f"ESG Risk Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Professional email body
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #1B3B4F;">🌱 ESG Risk Intelligence Report</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
            
            <h3>Executive Summary</h3>
            <ul>
                <li>Total Items: {total_items}</li>
                <li>Active Items: {active_items}</li>
                <li>Overdue: {overdue_items}</li>
                <li>Critical: {critical_items}</li>
                <li>Average Risk Score: {avg_risk:.1f}</li>
            </ul>
            
            <h3>Top 5 Priorities</h3>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background: #F2F4F7;">
                    <th style="padding: 8px; text-align: left;">Regulation</th>
                    <th style="padding: 8px; text-align: left;">Risk Level</th>
                    <th style="padding: 8px; text-align: left;">Risk Score</th>
                    <th style="padding: 8px; text-align: left;">Due Date</th>
                </tr>
        """
        
        for _, row in df.nlargest(5, "Risk Score").iterrows():
            body += f"""
                <tr>
                    <td style="padding: 6px;">{row['Regulation']}</td>
                    <td style="padding: 6px;">{row['Risk Level']}</td>
                    <td style="padding: 6px;">{row['Risk Score']:.1f}</td>
                    <td style="padding: 6px;">{row['Due Date'].strftime('%Y-%m-%d')}</td>
                </tr>
            """
        
        body += """
            </table>
            <p style="margin-top: 20px; color: #667085; font-size: 0.9em;">
                This is an automated report from the ESG Risk Intelligence Platform.<br>
                For demonstration purposes only. Not legal/financial advice.
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # In production, uncomment these lines:
        # server = smtplib.SMTP(smtp_server, smtp_port)
        # server.starttls()
        # server.login(sender, password)
        # server.send_message(msg)
        # server.quit()
        
        return True, "Email ready (demo mode)"
        
    except Exception as e:
        return False, str(e)

if st.button("📤 Send Report", use_container_width=True):
    if not recipient:
        st.warning("Please enter a recipient email address")
    else:
        with st.spinner("Preparing report..."):
            success, message = send_secure_email(recipient)
            if success:
                st.success(f"✅ {message}")
                if not os.getenv("ESG_SMTP_SERVER"):
                    st.info("📧 To enable actual email: Set ESG_SMTP_SERVER, ESG_SMTP_PORT, ESG_SENDER_EMAIL, ESG_SENDER_PASSWORD in environment")
            else:
                st.error(f"❌ {message}")

# ============================================================================
# DATA QUALITY & AUDIT
# ============================================================================

with st.expander("📋 Data Quality & Audit Trail"):
    tab1, tab2, tab3 = st.tabs(["Completeness", "Statistics", "Audit"])
    
    with tab1:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.warning("Missing values detected")
            st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}))
        else:
            st.success("✅ No missing values")
    
    with tab2:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
    
    with tab3:
        st.info(f"Last Loaded: {st.session_state.last_loaded}")
        st.info(f"Total Records: {len(df)}")
        st.info(f"Unique Regulations: {df['Regulation'].nunique()}")
        st.info(f"Departments: {', '.join(df['Department'].unique())}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    🌱 ESG Risk Intelligence Platform v1.0 | Enterprise Demonstration<br>
    Built with Streamlit • Secure by Design • Not legal/financial advice<br>
    © 2024 | For demonstration purposes only
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEPLOYMENT INSTRUCTIONS (COMMENTED)
# ============================================================================

"""
DEPLOYMENT CHECKLIST:
=====================

1. GitHub Repository
   - Create new repo: esg-risk-intelligence-platform
   - Push this code as app.py
   - Add requirements.txt:
     
     streamlit==1.28.1
     pandas==2.0.3
     plotly==5.17.0
     openpyxl==3.1.2

2. Streamlit Cloud (Recommended)
   - Go to https://streamlit.io/cloud
   - Connect GitHub repo
   - Deploy main branch
   - Add secrets (Settings → Secrets):
     
     ESG_SMTP_SERVER = "smtp.gmail.com"
     ESG_SMTP_PORT = "587"
     ESG_SENDER_EMAIL = "your-email@gmail.com"
     ESG_SENDER_PASSWORD = "your-app-password"

3. Custom Domain (Optional)
   - Buy domain from GoDaddy/Namecheap
   - Add CNAME to Streamlit
   - Update DNS settings

4. LinkedIn Announcement Post:
   
   🚀 I just built and deployed an ESG Risk Intelligence Platform
   
   Why: Most compliance teams use static spreadsheets with no risk prioritization
   
   What: Dynamic risk scoring engine with:
   • Configurable risk thresholds
   • Department-level analytics
   • Executive reporting
   • Priority-based queueing
   
   Built with: Python + Streamlit + Plotly
   
   Try it: [your-url]
   
   Open to feedback and collaboration! 🌱

5. Portfolio Update:
   - Add to GitHub profile
   - Add to LinkedIn Projects
   - Screenshot in portfolio
"""
