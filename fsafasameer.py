import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# Error handling
try:
    from textstat import textstat
    from textblob import TextBlob
except ImportError:
    st.error("Missing libraries! Run: pip install textstat textblob")
    st.stop()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic Intelligence Suite", page_icon="⚖️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .positive-trend { color: #00cc96; font-weight: bold; }
    .negative-trend { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. HELPERS: LOADERS & EXTRACTORS ---
@st.cache_data
def load_red_flags():
    # Fallback dictionary if file not uploaded
    default_data = {
        "Word": ["contingent", "estimate", "litigation", "claim", "uncertainty", "material", "adverse", "going concern", "restatement", "impairment", "write-off", "related party"],
        "Category": ["Uncertainty", "Uncertainty", "Legal", "Legal", "Uncertainty", "Materiality", "Risk", "Viability", "Accounting", "Loss", "Loss", "Governance"]
    }
    return pd.DataFrame(default_data)

@st.cache_data
def extract_text_fast(file, start_p=1, end_p=None):
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        # Scan Loop
        for i in range(start_p - 1, end_p):
            page_text = reader.pages[i].extract_text()
            if page_text: text += page_text + "\n"
        return text
    except: return ""

# --- 2. ANALYTICS ENGINE ---
def analyze_metrics(text, red_flag_df):
    if not text: return None
    
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # Metrics
    try: fog = textstat.gunning_fog(text)
    except: fog = 0
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Passive Voice Detection (Heuristic: "was/were" + verb-ed)
    passive_matches = re.findall(r'\b(was|were|been|being)\b\s+\w+ed\b', text.lower())
    passive_pct = (len(passive_matches) / total_words) * 1000 # Per 1k words
    
    # Red Flag Matching
    red_flag_set = set(red_flag_df['Word'].str.lower().unique())
    matched_words = [w for w in words if w in red_flag_set]
    
    # Complex Words %
    complex_count = sum(1 for w in words if textstat.syllable_count(w) >= 3)
    complex_pct = (complex_count / total_words) * 100

    return {
        "fog": fog,
        "sentiment": sentiment,
        "passive_score": passive_pct,
        "complex_pct": complex_pct,
        "matched_words": matched_words,
        "total_words": total_words
    }

def calculate_similarity(text1, text2):
    # Cosine Similarity
    try:
        # Truncate for speed
        t1 = " ".join(text1.split()[:5000])
        t2 = " ".join(text2.split()[:5000])
        vectorizer = CountVectorizer().fit_transform([t1, t2])
        return cosine_similarity(vectorizer.toarray())[0][1]
    except: return 0

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Inputs")
    file_curr = st.file_uploader("Current Year Report (PDF)", type="pdf")
    file_prev = st.file_uploader("Previous Year Report (Optional)", type="pdf")
    
    st.markdown("---")
    st.header("🔢 Financial 'Smoke Test'")
    st.caption("Check for Earnings Manipulation (Accruals)")
    net_income = st.number_input("Net Income (Cr)", value=0.0)
    cash_flow = st.number_input("Cash Flow from Ops (Cr)", value=0.0)
    
    # Settings
    use_all = st.checkbox("Scan Full Doc", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start", 1, value=50)
        end_p = c2.number_input("End", 1, value=100)
    else: end_p = None

# --- MAIN APP ---
if file_curr:
    # Load Data
    red_flags = load_red_flags()
    
    # Process Current
    text_curr = extract_text_fast(file_curr, start_p, end_p)
    metrics_curr = analyze_metrics(text_curr, red_flags)
    
    # Process Previous (If exists)
    metrics_prev = None
    similarity = None
    if file_prev:
        text_prev = extract_text_fast(file_prev, start_p, end_p)
        metrics_prev = analyze_metrics(text_prev, red_flags)
        similarity = calculate_similarity(text_curr, text_prev)

    # --- DASHBOARD ---
    st.title("Forensic Analysis Dashboard")
    
    # 1. FINANCIAL CHECK (Top Alert)
    if net_income > 0 and cash_flow > 0:
        if net_income > (cash_flow * 1.5):
            st.warning(f"⚠️ **Accruals Warning:** Net Income is significantly higher than Cash Flow ({net_income} vs {cash_flow}). This suggests low Quality of Earnings.")
    
    # 2. COMPARATIVE METRICS
    st.subheader("📊 Year-over-Year Linguistic Shift")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Fog Index Delta
    delta_fog = (metrics_curr['fog'] - metrics_prev['fog']) if metrics_prev else 0
    c1.metric("Fog Index", f"{metrics_curr['fog']:.2f}", f"{delta_fog:.2f}", delta_color="inverse")
    
    # Sentiment Delta
    delta_sent = (metrics_curr['sentiment'] - metrics_prev['sentiment']) if metrics_prev else 0
    c2.metric("Sentiment Score", f"{metrics_curr['sentiment']:.2f}", f"{delta_sent:.2f}")
    
    # Similarity Score
    sim_val = f"{similarity*100:.1f}%" if similarity else "N/A"
    c3.metric("Consistency (YoY)", sim_val, "Target: >90%")
    
    # Passive Voice Delta
    delta_pas = (metrics_curr['passive_score'] - metrics_prev['passive_score']) if metrics_prev else 0
    c4.metric("Passive Voice / 1k", f"{metrics_curr['passive_score']:.1f}", f"{delta_pas:.1f}", delta_color="inverse")

    st.markdown("---")

    # 3. SENTIMENT & COMPLEXITY COMPARISON CHART
    if metrics_prev:
        st.subheader("📈 Trend Visualization")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Side-by-Side Bar Chart for Sentiment
            fig_sent = go.Figure(data=[
                go.Bar(name='Previous Year', x=['Sentiment'], y=[metrics_prev['sentiment']], marker_color='#95a5a6'),
                go.Bar(name='Current Year', x=['Sentiment'], y=[metrics_curr['sentiment']], marker_color='#3498db')
            ])
            fig_sent.update_layout(title="Sentiment Shift (Tone Analysis)", barmode='group')
            st.plotly_chart(fig_sent, use_container_width=True)
            st.caption("A significant drop in sentiment often precedes bad news disclosure.")
            
        with col_chart2:
             # Side-by-Side Bar Chart for Complexity
            fig_fog = go.Figure(data=[
                go.Bar(name='Previous Year', x=['Fog Index'], y=[metrics_prev['fog']], marker_color='#95a5a6'),
                go.Bar(name='Current Year', x=['Fog Index'], y=[metrics_curr['fog']], marker_color='#e74c3c')
            ])
            fig_fog.update_layout(title="Complexity Spike (Obfuscation Check)", barmode='group')
            st.plotly_chart(fig_fog, use_container_width=True)
            st.caption("If Complexity rises while Sentiment falls, risk is VERY HIGH.")

    st.markdown("---")
    
    # 4. RED FLAG CLOUD (Custom List)
    st.subheader("🚩 Anomalous Word Cloud")
    c_cloud, c_stats = st.columns([2, 1])
    
    with c_cloud:
        if metrics_curr['matched_words']:
            wc = WordCloud(background_color="white", colormap="Reds", height=300).generate(" ".join(metrics_curr['matched_words']))
            fig_wc, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No anomalies found in current document.")
            
    with c_stats:
        if metrics_curr['matched_words']:
            st.write("**Top Red Flags**")
            counts = Counter(metrics_curr['matched_words']).most_common(10)
            st.dataframe(pd.DataFrame(counts, columns=["Word", "Freq"]), hide_index=True, use_container_width=True)
    
    # 5. REGRESSION (Restored)
    st.markdown("---")
    st.subheader("📉 Industry Benchmark")
    
    # Dummy Benchmark Data
    np.random.seed(42)
    bench_fog = np.random.normal(16, 3, 50)
    bench_risk = (bench_fog * 2.0) + np.random.normal(0, 5, 50)
    df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
    
    model = LinearRegression()
    model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
    curr_pred = model.predict([[metrics_curr['fog']]])[0]
    
    fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.3, title="Regression: Complexity vs. Fraud Risk")
    line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
    fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Trend'))
    fig_reg.add_traces(go.Scatter(x=[metrics_curr['fog']], y=[curr_pred], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='Your File'))
    
    st.plotly_chart(fig_reg, use_container_width=True)

else:
    st.info("Upload Current Year PDF to start.")
