import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# Error handling for libraries
try:
    from textstat import textstat
    from textblob import TextBlob
except ImportError:
    st.error("Please install missing libraries: pip install textstat textblob")
    st.stop()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Analyst", page_icon="🕵️‍♂️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #4e8cff; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    .metric-label { font-size: 14px; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #222; }
    .metric-sub { font-size: 12px; color: #888; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADER (RED FLAGS) ---
@st.cache_data
def load_red_flag_dictionary():
    """
    Loads the custom anomaly dictionary. 
    Assumes columns: 'Word', 'Category'
    """
    # Replace 'red_flags.csv' with your actual filename if different
    # If using the Excel file you uploaded, ensure it is saved as CSV or use pd.read_excel
    try:
        # Trying to read CSV first
        df = pd.read_csv("Annual_Report_Red_Flags.csv") # UPDATE THIS FILENAME
    except:
        try:
            # Fallback to creating a dummy dictionary if file missing (so app doesn't crash)
            data = {
                "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "pending", "unresolved", "material", "adverse", "risk", "doubt", "going concern", "restatement", "write-off", "impairment"],
                "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Legal", "Legal", "Materiality", "Negative", "Risk", "Viability", "Viability", "Accounting", "Loss", "Loss"]
            }
            df = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Could not load dictionary: {e}")
            return pd.DataFrame()
            
    # Normalize to lowercase for matching
    df['Word'] = df['Word'].str.lower().str.strip()
    return df

# --- 2. TEXT EXTRACTION ---
@st.cache_data
def extract_text_fast(file, start_p=1, end_p=None):
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        # Progress bar
        my_bar = st.progress(0, text="Scanning pages...")
        for i in range(start_p - 1, end_p):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
            my_bar.progress(min(int(((i - start_p + 1) / (end_p - start_p + 1)) * 100), 100))
        my_bar.empty()
        return text, total_pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", 0

# --- 3. METRICS ENGINE ---
def analyze_metrics(text, red_flag_df):
    if not text: return None
    
    # A. Basic Counts
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # B. Readability
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    
    # C. Sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity # -1 to 1
    
    # D. Complex Words
    # Get set of complex words
    unique_words = set(words)
    complex_words_list = [w for w in unique_words if textstat.syllable_count(w) >= 3]
    # Count occurrences of these complex words in the full text
    complex_count = sum(1 for w in words if w in complex_words_list)
    complex_pct = (complex_count / total_words) * 100
    
    # E. Custom Red Flag Analysis (Using CSV)
    # Filter: Keep only words that exist in our Red Flag CSV
    red_flag_set = set(red_flag_df['Word'].unique())
    
    # Find matches in document
    matched_words = [w for w in words if w in red_flag_set]
    
    # Map back to categories
    # Create a lookup dict: word -> category
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict()
    matched_categories = [word_to_cat.get(w, "Unknown") for w in matched_words]
    
    return {
        "fog_index": fog_index,
        "sentiment": sentiment,
        "complex_pct": complex_pct,
        "total_words": total_words,
        "matched_words": matched_words, # List of red flag words found
        "matched_categories": matched_categories # Corresponding categories
    }

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    # Allow user to upload their own dictionary optionally
    uploaded_dict = st.file_uploader("Upload Red Flag List (CSV/Excel)", type=["csv", "xlsx"])
    
    st.markdown("---")
    use_all = st.checkbox("Scan Entire Document", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start Page", 1, value=50)
        end_p = c2.number_input("End Page", 1, value=100)
    else: end_p = None

# --- MAIN APP ---
if uploaded_file:
    # Load Dictionary
    if uploaded_dict:
        if uploaded_dict.name.endswith('.csv'):
            red_flags_df = pd.read_csv(uploaded_dict)
        else:
            red_flags_df = pd.read_excel(uploaded_dict)
        # Ensure standard columns
        if 'Word' not in red_flags_df.columns: 
            st.error("Uploaded dictionary must have a 'Word' column.")
            st.stop()
        red_flags_df['Word'] = red_flags_df['Word'].str.lower().str.strip()
    else:
        red_flags_df = load_red_flag_dictionary()

    # Process Text
    text, _ = extract_text_fast(uploaded_file, start_p, end_p)
    
    if text:
        metrics = analyze_metrics(text, red_flags_df)
        
        st.title("Forensic Analysis Dashboard")
        st.caption(f"Analyzing {metrics['total_words']:,} words against {len(red_flags_df)} Red Flags.")

        # --- ROW 1: FLASHCARDS ---
        col1, col2, col3 = st.columns(3)
        
        # 1. Fog Index
        with col1:
            risk = "High" if metrics['fog_index'] > 18 else "Low"
            color = "high-risk" if metrics['fog_index'] > 18 else "good-metric"
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">Fog Index (Complexity)</div>
                <div class="metric-value">{metrics['fog_index']:.1f}</div>
                <div class="metric-sub">{risk} Complexity Level</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Sentiment (With Explanation)
        with col2:
            # Logic: -1 (Negative) to +1 (Positive)
            sent = metrics['sentiment']
            if sent > 0.2: s_label = "Overly Positive"
            elif sent < -0.1: s_label = "Negative/Concering"
            else: s_label = "Neutral/Balanced"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Score</div>
                <div class="metric-value">{sent:.2f}</div>
                <div class="metric-sub">{s_label}</div>
                <div style="font-size:10px; color:#666; margin-top:5px;">
                <i>Range: -1 (Negative) to +1 (Positive).<br>High positive scores in poor financial years indicate obfuscation.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Complex Words %
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Complex Word Density</div>
                <div class="metric-value">{metrics['complex_pct']:.1f}%</div>
                <div class="metric-sub">Percentage of words with 3+ syllables</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: CUSTOM WORD CLOUD & STATS ---
        st.subheader("🚩 Red Flag Analysis (Based on Uploaded List)")
        
        c_cloud, c_stats = st.columns([2, 1])
        
        with c_cloud:
            # Generate Cloud ONLY for matched words
            if metrics['matched_words']:
                # Join the list of matched words into a string
                text_for_cloud = " ".join(metrics['matched_words'])
                
                wc = WordCloud(
                    background_color="white", 
                    colormap="Reds", 
                    width=600, 
                    height=350,
                    collocations=False
                ).generate(text_for_cloud)
                
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No Red Flag words found in this document.")

        with c_stats:
            if metrics['matched_words']:
                # 1. Most Frequent Red Flags
                st.markdown("**Top Anomalous Words Found**")
                counts = Counter(metrics['matched_words'])
                df_counts = pd.DataFrame(counts.most_common(10), columns=["Word", "Count"])
                st.dataframe(df_counts, hide_index=True, use_container_width=True, height=150)
                
                # 2. Category Analysis
                st.markdown("**Primary Anomaly Category**")
                cat_counts = Counter(metrics['matched_categories'])
                top_cat = cat_counts.most_common(1)[0] # (Category, Count)
                
                st.info(f"🚨 **{top_cat[0]}**")
                st.caption(f"This category appeared {top_cat[1]} times.")
                
                # Show distribution
                fig_cat = px.pie(names=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.4)
                fig_cat.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150, showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # --- ROW 3: REGRESSION BENCHMARK (Restored) ---
        st.subheader("📈 Fraud Risk Regression Model")
        st.write("Comparing your document's Fog Index against an Industry Benchmark dataset.")
        
        col_reg_desc, col_reg_chart = st.columns([1, 2])
        
        with col_reg_desc:
            st.markdown("""
            **How this works:**
            We simulated a dataset of 50 companies.
            * **X-Axis:** Fog Index (Complexity)
            * **Y-Axis:** Fraud Risk Score
            
            The **Red X** represents the uploaded document. If it falls high on the line, the complexity suggests higher risk.
            """)
            
        with col_reg_chart:
            # 1. Generate Dummy Data
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50) # Industry Avg Fog ~16
            # Assume Risk correlates with Fog
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            # 2. Train Model
            model = LinearRegression()
            X = df_bench[["Fog Index"]]
            y = df_bench["Risk Score"]
            model.fit(X, y)
            
            # 3. Predict for Current File
            curr_fog = metrics['fog_index']
            pred_risk = model.predict([[curr_fog]])[0]
            
            # 4. Plot
            fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4, title="Industry Benchmark Analysis")
            
            # Add Trend Line
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            line_y = model.predict(line_x)
            fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=line_y, mode='lines', name='Industry Trend', line=dict(color='gray', dash='dash')))
            
            # Add Current File Marker
            fig_reg.add_traces(go.Scatter(
                x=[curr_fog], 
                y=[pred_risk], 
                mode='markers+text', 
                marker=dict(color='red', size=15, symbol='x'),
                name='Your File',
                text=["YOU"],
                textposition="top center"
            ))
            
            st.plotly_chart(fig_reg, use_container_width=True)

else:
    st.info("Upload a PDF to begin analysis.")
