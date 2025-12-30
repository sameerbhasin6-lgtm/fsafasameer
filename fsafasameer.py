import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# Try-except block to handle missing libraries gracefully
try:
    from textstat import textstat
    from textblob import TextBlob
except ImportError:
    st.error("Missing Libraries! Please run: pip install textstat textblob")
    st.stop()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Commander", page_icon="🕵️‍♂️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #4e8cff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. ADVANCED EXTRACTION (Page-wise) ---
@st.cache_data
def extract_text_by_page(file, start_p=1, end_p=None):
    """Extracts text page-by-page using pypdf (Fast)."""
    pages_data = []
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        # Validate ranges
        if start_p < 1: start_p = 1
        if end_p is None or end_p > total_pages: end_p = total_pages
        if start_p > total_pages: start_p = 1
        
        # Progress bar
        my_bar = st.progress(0, text="Scanning pages...")
        
        for i in range(start_p - 1, end_p):
            try:
                text = reader.pages[i].extract_text()
                if text:
                    pages_data.append({"page": i + 1, "text": text})
            except:
                continue # Skip bad pages
            
            # Update bar
            percent = int(((i - (start_p - 1)) / ((end_p - (start_p - 1)) + 1)) * 100)
            my_bar.progress(min(percent, 100))
            
        my_bar.empty()
        return pages_data, total_pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return [], 0

# --- 2. ANALYTICS ENGINE ---
def analyze_page_metrics(pages_data):
    results = []
    full_text = ""
    
    for p in pages_data:
        txt = p['text']
        full_text += txt + " "
        
        # Safe Metric Calculation
        try: fog = textstat.gunning_fog(txt)
        except: fog = 0
        
        try: 
            blob = TextBlob(txt)
            sentiment = blob.sentiment.polarity
        except: 
            sentiment = 0
        
        results.append({"Page": p['page'], "Fog Index": fog, "Sentiment": sentiment})
        
    return pd.DataFrame(results), full_text

def perform_topic_modeling(text, n_topics=3):
    """LDA Topic Modeling."""
    if len(text.split()) < 50: return {} # Not enough text
    
    stopwords = list(STOPWORDS) + ['company', 'year', 'financial', 'notes', 'december', 'march', 'ended', 'amount', 'value', 'crore', 'lakhs']
    
    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words=stopwords)
        dtm = vectorizer.fit_transform([text])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        
        topics = {}
        feature_names = vectorizer.get_feature_names_out()
        for index, topic in enumerate(lda.components_):
            topics[f"Topic {index+1}"] = [feature_names[i] for i in topic.argsort()[-5:]]
        return topics
    except:
        return {}

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    uploaded_file = st.file_uploader("Upload Annual Report (PDF)", type="pdf")
    
    st.markdown("---")
    st.write("⚙️ **Scan Scope**")
    use_all = st.checkbox("Scan Entire Document", value=False)
    
    start_p, end_p = 1, 50
    if not use_all:
        col1, col2 = st.columns(2)
        start_p = col1.number_input("Start Page", 1, value=50, step=10)
        end_p = col2.number_input("End Page", 1, value=100, step=10)
    else:
        end_p = None

# --- MAIN APP ---
if uploaded_file:
    # 1. Extraction
    pages_data, total_pgs = extract_text_by_page(uploaded_file, start_p, end_p)
    
    if pages_data:
        # 2. Analysis
        df_trends, full_text = analyze_page_metrics(pages_data)
        
        # Global Metrics
        avg_fog = df_trends["Fog Index"].mean()
        avg_sent = df_trends["Sentiment"].mean()
        
        # --- DASHBOARD ---
        st.title("Forensic Commander Dashboard")
        st.caption(f"Analyzing {len(pages_data)} pages | Total Words: {len(full_text.split()):,}")
        
        # ROW 1: METRICS
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            risk = "High" if avg_fog > 18 else "Low"
            color = "high-risk" if avg_fog > 18 else "good-metric"
            st.markdown(f'<div class="metric-card {color}"><h3>Avg Fog Index</h3><h2>{avg_fog:.1f}</h2><p>{risk} Complexity</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h3>Sentiment</h3><h2>{avg_sent:.2f}</h2><p>-1 (Neg) to +1 (Pos)</p></div>', unsafe_allow_html=True)
        with c3:
             # Complex Words Count
            complex_words = [w for w in full_text.split() if textstat.syllable_count(w) >= 3]
            st.markdown(f'<div class="metric-card"><h3>Complex Words</h3><h2>{len(complex_words)}</h2><p>Total Count</p></div>', unsafe_allow_html=True)
        with c4:
             st.markdown(f'<div class="metric-card"><h3>Pages Scanned</h3><h2>{len(pages_data)}</h2><p>of {total_pgs} Total</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROW 2: HEARTBEAT CHART
        st.subheader("📈 Risk Heartbeat (Complexity Flow)")
        fig_trend = px.line(df_trends, x="Page", y="Fog Index", markers=True)
        fig_trend.add_hline(y=18, line_dash="dash", line_color="red")
        st.plotly_chart(fig_trend, use_container_width=True)

        # ROW 3: TABS
        tab1, tab2 = st.tabs(["☁️ Word Cloud", "🧠 Hidden Topics"])
        
        with tab1:
            if len(complex_words) > 0:
                wc = WordCloud(background_color="white", colormap="Reds", height=300).generate(" ".join(complex_words))
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Not enough complex words found.")
        
        with tab2:
            topics = perform_topic_modeling(full_text)
            if topics:
                cols = st.columns(3)
                for i, (topic, words) in enumerate(topics.items()):
                    with cols[i % 3]:
                        st.info(f"**{topic}**")
                        st.write(", ".join(words))
            else:
                st.warning("Not enough text data for Topic Modeling.")

    else:
        st.warning("No text extracted. Try a different page range (e.g., avoid image-heavy pages).")

else:
    st.info("Upload a PDF to start.")