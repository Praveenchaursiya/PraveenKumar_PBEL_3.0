# app.py
import os
import time
import re
import math
import json
import random
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup

# Optional: better extraction where allowed
try:
    from newspaper import Article
    NEWSPAPER_OK = True
except Exception:
    NEWSPAPER_OK = False

# -------------
# Gemini setup (robust: don't crash if secrets.toml is broken)
# -------------
GEMINI_MODE = "gemini-1.5-flash"
try:
    # Try reading Streamlit secrets safely
    GEMINI_API_KEY = None
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")  # may raise if secrets file is malformed
    except Exception:
        # fallback to env var if secrets parsing failed or file missing
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        # warn user only in-app (not fatal)
        st.warning("Could not read .streamlit/secrets.toml (or it is malformed). Falling back to environment variable for GEMINI_API_KEY if set.")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to import/configure google.generativeai: {e}")
        genai = None
else:
    genai = None

# -------------
# Streamlit UI Configuration
# -------------
st.set_page_config(
    page_title="FactCheckAI - Fake News Detector with Evidence & Sources",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 45%, #f7f7fb 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero-panel {
        background: linear-gradient(135deg, #0f172a 0%, #172554 50%, #2563eb 100%);
        color: white;
        padding: 1.6rem 1.8rem;
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(37, 99, 235, 0.18);
        margin-bottom: 1.2rem;
    }
    .main-header {
        font-size: 2.8rem;
        color: white;
        margin: 0.2rem 0 0.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #dbeafe;
        margin-bottom: 0.8rem;
        max-width: 720px;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem 0.35rem 0.25rem 0;
    }
    .neutral-pill {
        background: rgba(255,255,255,0.14);
    }
    .success-pill {
        background: rgba(16, 185, 129, 0.22);
        border-color: rgba(16, 185, 129, 0.35);
    }
    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .hero-stat {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 16px;
        padding: 0.8rem 0.9rem;
        font-size: 0.95rem;
    }
    .panel-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .sidebar-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 18px;
        padding: 0.9rem;
        backdrop-filter: blur(6px);
    }
    .verdict-card {
        font-weight: 700;
        font-size: 1.55rem;
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border-left: 6px solid;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        margin: 0.8rem 0 1rem;
    }
    .verdict-true {
        color: #166534;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #22c55e;
    }
    .verdict-false {
        color: #b91c1c;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-color: #ef4444;
    }
    .verdict-uncertain {
        color: #b45309;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-color: #f59e0b;
    }
    .evidence-card {
        border: 1px solid #e5e7eb;
        padding: 1rem;
        margin: 0.7rem 0;
        border-radius: 14px;
        background: #fcfdff;
        transition: all 0.2s ease;
    }
    .evidence-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
        background: #ffffff;
    }
    .source-badge {
        background: #eff6ff;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.84rem;
        color: #1d4ed8;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
    }
    .confidence-bar {
        height: 9px;
        background: #e2e8f0;
        border-radius: 999px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1s ease-in-out;
    }
    .progress-container {
        margin: 2rem 0;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    .share-buttons {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .educational-tip {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid #d1d5db;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #0f172a;
        font-weight: 600;
        padding: 0.45rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #60a5fa;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.16);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }
    @media (max-width: 768px) {
        .main-header { font-size: 2.2rem; }
        .hero-stats { grid-template-columns: 1fr; }
        .verdict-card { font-size: 1.3rem; padding: 1rem; }
    }
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

# Educational tips
EDUCATIONAL_TIPS = [
    "💡 Always verify information with multiple reliable sources before sharing",
    "🔍 Check the publication date - older information may be outdated or inaccurate",
    "🏛️ Government (.gov) and educational (.edu) sources tend to be more reliable",
    "📊 Look for original research and primary sources rather than interpretations",
    "⏰ Be wary of breaking news - initial reports often contain errors",
    "🌐 Consider the source's reputation and potential biases",
    "📝 Check if other reputable news organizations are reporting the same information",
    "🔎 Look for supporting evidence like data, studies, or expert opinions"
]

# Credible domains for source scoring
CREDIBLE_DOMAINS = {
    'reuters.com': 0.95, 'ap.org': 0.95, 'bbc.com': 0.9, 'bbc.co.uk': 0.9,
    'nytimes.com': 0.9, 'theguardian.com': 0.9, 'wsj.com': 0.9,
    '.gov': 0.95, '.edu': 0.9, '.ac.uk': 0.9, '.edu.au': 0.9,
    'who.int': 0.95, 'un.org': 0.95, 'nasa.gov': 0.95, 'nih.gov': 0.95
}

# -------------
# Helper Functions
# -------------
@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def fetch_google_news(query: str, region: str):
    """
    Fetch Google News RSS results, try a couple of RSS URL variations and return (results, used_url)
    """
    # Primary (region-specific) and fallback simple search
    primary = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-{region}&gl={region}&ceid={region}:en"
    fallback = f"https://news.google.com/rss/search?q={quote_plus(query)}"

    for rss_url in (primary, fallback):
        feed = feedparser.parse(rss_url)
        # If parser had entries, return them (even if published missing)
        if getattr(feed, "entries", None):
            results = []
            for entry in feed.entries:
                results.append({
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "published": getattr(entry, "published", None),
                    "source": getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else None,
                })
            return results, rss_url

    # nothing found - return empty with last attempted URL
    return [], fallback

@st.cache_data(show_spinner=False, ttl=1800, max_entries=50)
def extract_article_text(url: str, timeout: int = 8) -> str:
    """Extract article text with improved error handling"""
    try:
        if NEWSPAPER_OK:
            try:
                art = Article(url)
                art.download()
                art.parse()
                text = art.text.strip()
                if text and len(text.split()) > 50:
                    return text
            except Exception:
                pass

        headers = {"User-Agent": "Mozilla/5.0 (compatible; FactCheckAI/1.0; +https://github.com/yourusername/factcheck-ai)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        selectors = ["article", "main", "[itemprop='articleBody']", ".article-content", ".post-content", ".story-content"]
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                text = " ".join([elem.get_text(" ", strip=True) for elem in elements])
                if len(text.split()) > 50:
                    return text
        
        # fallback whole page
        return soup.get_text(" ", strip=True)
        
    except Exception:
        return ""  # return empty string on failure rather than an error message

def parse_pubdate_safe(date_str):
    """Try feedparser._parse_date first, fallback to None"""
    try:
        parsed = feedparser._parse_date(date_str)
        if parsed:
            return datetime(*parsed[:6])
    except Exception:
        pass
    return None

def rate_source_credibility(url: str, content: str) -> float:
    """Rate source credibility score (0-1)"""
    credibility = 0.5
    url_l = (url or "").lower()
    
    for domain_pattern, score in CREDIBLE_DOMAINS.items():
        if domain_pattern in url_l:
            credibility = max(credibility, score)
    
    # Content quality indicators
    if len((content or "").split()) > 200:
        credibility = min(credibility + 0.1, 1.0)
    if any(keyword in (content or "").lower() for keyword in ['study', 'research', 'data', 'according to', 'experts say']):
        credibility = min(credibility + 0.05, 1.0)
    
    return credibility

def trim_text(text: str, max_chars: int = 4000) -> str:
    """Trim text at sentence boundary"""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    m = re.findall(r"^(.+?[.!?])\s", cut, flags=re.S)
    return m[-1] if m else cut

def make_prompt_for_gemini(claim: str, evidence_items: list[str]) -> str:
    """Create an optimized prompt that requests JSON with verdict, confidence, rationale[], cited_sources[]"""
    bullets = "\n\n".join([f"[Source {i+1}]\n{trim_text(txt, 1200)}" for i, txt in enumerate(evidence_items)])
    return f"""
You are an expert fact-checker. Use the EVIDENCE below to evaluate the CLAIM.

CLAIM:
\"\"\"{claim}\"\"\"

EVIDENCE (snippets from multiple sources):
{bullets}

Task:
- Determine a verdict: one of "Likely True", "Likely False", or "Uncertain".
- Assign a confidence score between 0.0 and 1.0.
- Provide 3 short bullet points as rationale citing the most relevant sources.
- Provide up to 3 cited_sources objects with fields: idx (source index), quote_or_summary (short excerpt), relevance (low|med|high).

Return valid JSON only, using these keys: verdict, confidence, rationale (array of strings), cited_sources (array of objects).
"""

# def extract_json_from_text(text: str):
#     """Try to extract the first JSON object from model text output."""
#     if not text:
#         return None
#     # remove triple backticks if present
#     text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
#     # find substring that looks like JSON object
#     m = re.search(r"(\{[\s\S]*\})", text)
#     if m:
#         try:
#             return json.loads(m.group(1))
#         except Exception:
#             # try to fix common issues: replace single quotes with double (risky)
#             try:
#                 fixed = m.group(1).replace("'", "\"")
#                 return json.loads(fixed)
#             except Exception:
#                 return None
#     # final fallback: try to parse whole text
#     try:
#         return json.loads(text)
#     except Exception:
#         return None
def extract_json_from_text(text: str):
    """
    Robust JSON extraction from LLM output.
    Handles:
    - Markdown fences (```json)
    - Extra text before/after JSON
    - Multiple JSON-like blocks
    - Trailing commas
    - Comments (// ...)
    - Smart quotes
    - Single quotes
    """

    if not text or not isinstance(text, str):
        return None

    # Remove markdown fences
    text = re.sub(r"```json|```", "", text, flags=re.I).strip()

    # Normalize smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Remove line comments
    text = re.sub(r"//.*", "", text)

    # Extract all { ... } blocks
    blocks = re.findall(r"\{[\s\S]*?\}", text)
    if not blocks:
        return None

    # Try each candidate, largest first
    blocks = sorted(blocks, key=len, reverse=True)

    for block in blocks:
        clean = block.strip()

        # Fix trailing commas before } or ]
        clean = re.sub(r",\s*([}\]])", r"\1", clean)

        # Attempt standard parse
        try:
            return json.loads(clean)
        except Exception:
            pass

        # Attempt with single→double quote conversion
        try:
            converted = clean.replace("'", '"')
            return json.loads(converted)
        except Exception:
            pass

    # Final fallback: try extraction from first { to last }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        fragment = text[start:end]

        fragment = re.sub(r",\s*([}\]])", r"\1", fragment)
        fragment = fragment.replace("'", '"')

        return json.loads(fragment)
    except Exception:
        return None


def fallback_rule_based_analysis(claim: str, docs: list[dict]):
    """A simple deterministic fallback analysis when Gemini is unavailable or parsing fails."""
    keywords = [w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3]
    if not keywords:
        keywords = [w.lower() for w in claim.split() if len(w) > 3]
    total = len(docs)
    support = 0
    contradict = 0
    top_sources = []
    scores = []
    for d in docs:
        txt = (d.get("title", "") + " " + d.get("text", "")).lower()
        # count how many keywords appear
        kcount = sum(1 for k in keywords if k in txt)
        if kcount > 0:
            support += 1
        # naive contradiction detection
        if any(phrase in txt for phrase in ["no evidence", "not true", "debunk", "false", "denied", "not found", "refute"]):
            contradict += 1
        scores.append(d.get("credibility", 0.5))
        # pick short summary
        excerpt = (d.get("text") or d.get("title") or "")[:280]
        top_sources.append({"idx": d["idx"], "quote_or_summary": excerpt, "relevance": "med"})
    avg_cred = sum(scores) / max(1, len(scores))
    # Decide
    if support >= max(1, math.ceil(total * 0.6)) and avg_cred > 0.6:
        verdict = "Likely True"
        confidence = min(0.9, avg_cred)
        rationale = [
            f"{support}/{total} sources mention the claim or related keywords.",
            f"Average source credibility is {avg_cred:.2f}. Top sources support the claim.",
            "No strong contradictory language found in majority of sources."
        ]
    elif contradict >= max(1, math.ceil(total * 0.5)):
        verdict = "Likely False"
        confidence = min(0.85, 0.5 + (avg_cred / 2))
        rationale = [
            f"{contradict}/{total} sources contain language indicating the claim was refuted or denied.",
            f"Average source credibility is {avg_cred:.2f}.",
            "Contradictory phrasing suggests claim is likely false or misrepresented."
        ]
    else:
        verdict = "Uncertain"
        confidence = min(0.6, avg_cred)
        rationale = [
            "Evidence is mixed or insufficient to make a confident call.",
            f"{support}/{total} sources mention claim keywords; {contradict} show refutation-like language.",
            "Consider more specific search terms or primary sources."
        ]
    cited = top_sources[:3]
    return {"verdict": verdict, "confidence": confidence, "rationale": rationale, "cited_sources": cited}

@st.cache_data(show_spinner=False, ttl=600, max_entries=20)
def reason_with_gemini(claim: str, docs: list[dict], temperature: float = 0.3):
    """Enhanced Gemini reasoning with robust parsing and deterministic fallback."""
    # If genai not configured, use fallback rule-based analysis
    if not genai:
        return fallback_rule_based_analysis(claim, docs)
    try:
        texts = [d.get("text", "") or d.get("title", "") for d in docs]
        prompt = make_prompt_for_gemini(claim, texts)
        model = genai.GenerativeModel(GEMINI_MODE)
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        output = resp.text.strip()
        # Try to extract JSON
        parsed = extract_json_from_text(output)
        if parsed:
            # Normalize structure to expected keys
            parsed.setdefault("verdict", parsed.get("verdict", "Uncertain"))
            parsed.setdefault("confidence", float(parsed.get("confidence", 0.5)))
            # ensure rationale is list
            rat = parsed.get("rationale", parsed.get("reasoning", []))
            if isinstance(rat, str):
                rat = [r.strip() for r in re.split(r"\n|-{1,}\s*", rat) if r.strip()]
            parsed["rationale"] = rat
            parsed.setdefault("cited_sources", parsed.get("cited_sources", []))
            return {
                "verdict": parsed["verdict"],
                "confidence": float(parsed["confidence"]),
                "rationale": parsed["rationale"],
                "cited_sources": parsed["cited_sources"],
            }
        else:
            # if we couldn't parse JSON, try to salvage with regex extraction for verdict & confidence
            v_match = re.search(r'(?i)(likely true|likely false|uncertain)', output)
            verdict = v_match.group(0).title() if v_match else "Uncertain"
            c_match = re.search(r'(\d?\.\d+|\d+)%', output)
            if c_match:
                # if % present
                conf = float(c_match.group(1).replace("%", "")) / 100.0
            else:
                # fallback numeric
                num_match = re.search(r'confidence[:\s]*([0-1](?:\.\d+)?)', output, re.I)
                conf = float(num_match.group(1)) if num_match else 0.5
            # Take first 3 lines as rationale
            lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
            rationale = lines[:3] if lines else ["Model returned text but not parseable JSON."]
            # include a short excerpt as cited_sources
            cited = []
            for i, d in enumerate(docs[:3]):
                cited.append({"idx": d["idx"], "quote_or_summary": (d.get("text") or d.get("title"))[:280], "relevance": "med"})
            return {"verdict": verdict, "confidence": conf, "rationale": rationale, "cited_sources": cited}
    except Exception:
        # Final fallback to deterministic rule-based analysis
        return fallback_rule_based_analysis(claim, docs)

def create_shareable_report(claim: str, result: dict, sources: list) -> str:
    """Create shareable report text"""
    return f"""
🔍 FactCheckAI Analysis Report
──────────────────────────────

Claim: "{claim}"

Verdict: {result.get('verdict', 'Uncertain')}
Confidence: {result.get('confidence', 0.0) * 100:.0f}%

Key Findings:
{chr(10).join(f'• {point}' for point in result.get('rationale', [])[:3])}

Sources Analyzed: {len(sources)}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

───
Generated by FactCheckAI - Transparent fact-checking with evidence
""".strip()

# -------------
# Main Application
# -------------
def main():
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_tip' not in st.session_state:
        st.session_state.current_tip = random.choice(EDUCATIONAL_TIPS)
    if 'last_request' not in st.session_state:
        st.session_state.last_request = 0
    if 'pre_filled' not in st.session_state:
        st.session_state.pre_filled = ""

    # Header
    st.markdown("""
    <div class="hero-panel">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap;">
            <div>
                <div class="pill">🧠 evidence-first analysis</div>
                <h1 class="main-header">🔍 FactCheckAI</h1>
                <p class="sub-header">Verify claims with transparent evidence, source scoring, and a clear verdict in seconds.</p>
            </div>
            <div class="pill">⚡ Fast • Accurate • Transparent</div>
        </div>
        <div class="hero-stats">
            <div class="hero-stat"><strong>Live evidence</strong><br/>News + article context</div>
            <div class="hero-stat"><strong>Source scoring</strong><br/>Credibility-aware results</div>
            <div class="hero-stat"><strong>Shareable output</strong><br/>Reports and downloads</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        region = st.selectbox(
            "News Region",
            ["US", "GB", "IN", "AU", "CA", "DE", "FR", "SG"],
            index=0,
            help="Select region for news sources"
        )
        freshness_hours = st.slider("Freshness (hours)", 6, 168, 48)
        max_articles = st.slider("Max Articles", 3, 15, 8)
        temperature = st.slider("Analysis Creativity", 0.0, 1.0, 0.3, 0.1)
        
        st.divider()
        st.header("📚 Recent Checks")
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[:5]):
                with st.expander(f"{item['claim'][:40]}...", expanded=i==0):
                    st.caption(f"**Verdict:** {item['result']['verdict']}")
                    st.caption(f"**Confidence:** {item['result']['confidence']:.0%}")
                    st.caption(f"**Sources:** {item['sources_count']}")
                    if st.button("🔍 Review", key=f"review_{i}"):
                        st.session_state.pre_filled = item['claim']
                        st.experimental_rerun()
        else:
            st.info("No recent checks yet")
        
        st.divider()
        st.markdown(f"**💡 Fact-Checking Tip:**\n\n{st.session_state.current_tip}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    with st.form("claim_form"):
        claim = st.text_area(
            "Paste headline or claim to verify:",
            value=st.session_state.get("pre_filled", ""),
            placeholder="Enter the claim you want to fact-check...",
            height=100,
            help="Be specific and include important context for accurate analysis"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("🔎 Verify Claim", use_container_width=True)
        with col2:
            clear_clicked = st.form_submit_button("🔄 Clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clear_clicked:
        st.session_state.clear()
        st.experimental_rerun()

    if submitted and claim.strip():
        # Rate limiting
        current_time = time.time()
        if current_time - st.session_state.last_request < 2:
            st.warning("⏳ Please wait a few seconds between requests")
            st.stop()
        st.session_state.last_request = current_time
        
        # Processing stages
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            "🔍 Searching news sources...",
            "📰 Fetching article content...",
            "🧠 Analyzing evidence...",
            "⚖️ Generating verdict..."
        ]
        
        # Stage 1: Search news
        progress_bar.progress(10)
        status_text.text(stages[0])
        items, used_rss = fetch_google_news(claim, region)
        
        # Show what we searched
        st.markdown("**🔎 RSS search URL used:**")
        st.code(used_rss)
        st.markdown("**🔎 Results returned by RSS (titles & published dates):**")
        if items:
            for i, e in enumerate(items[:max_articles], 1):
                st.markdown(f"{i}. [{e.get('title')}]({e.get('link')}) — {e.get('published') or 'no published date'}")
        else:
            st.warning("RSS returned no entries. Try a different query or region.")
        
        # Filter by freshness (allow items without published date)
        cutoff = datetime.utcnow() - timedelta(hours=freshness_hours)
        filtered = []
        for it in items:
            dt = parse_pubdate_safe(it.get("published"))
            # include item if date missing (we'll still try to fetch it) or if newer than cutoff
            if dt is None or dt >= cutoff:
                filtered.append(it)
        filtered = filtered[:max_articles]
        
        if not filtered:
            st.error("❌ No recent articles found after filtering. Try widening the time window or rephrasing the claim.")
            st.stop()
        
        # Stage 2: Extract content
        progress_bar.progress(40)
        status_text.text(stages[1])
        
        docs = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(extract_article_text, item["link"]): item for item in filtered}
            idx = 0
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    text = fut.result(timeout=12)
                except Exception:
                    text = ""
                idx += 1
                credibility = rate_source_credibility(item.get("link", ""), text)
                docs.append({
                    "idx": idx,
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "published": item.get("published"),
                    "source": item.get("source"),
                    "text": text,
                    "credibility": credibility
                })
        
        if not docs:
            st.error("❌ Could not extract content from articles. Please try a different claim.")
            st.stop()
        
        # Stage 3: AI Analysis
        progress_bar.progress(75)
        status_text.text(stages[2])
        result = reason_with_gemini(claim, docs, temperature)
        
        # Stage 4: Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Store in history
        st.session_state.history.insert(0, {
            "claim": claim,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "sources_count": len(docs)
        })
        
        # Display results
        st.markdown("---")
        
        # Verdict display
        verdict = result.get("verdict", "Uncertain").lower()
        confidence = result.get("confidence", 0.5)
        
        if "true" in verdict:
            st.markdown(f'<div class="verdict-card verdict-true">✅ Verdict: {result["verdict"]}</div>', unsafe_allow_html=True)
        elif "false" in verdict:
            st.markdown(f'<div class="verdict-card verdict-false">❌ Verdict: {result["verdict"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-card verdict-uncertain">⚠️ Verdict: {result["verdict"]}</div>', unsafe_allow_html=True)
        
        # Confidence visualization
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Confidence", f"{confidence:.0%}")
        with col2:
            color = '#2e8b57' if confidence > 0.7 else '#ff8c00' if confidence > 0.4 else '#dc143c'
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Rationale (model returns 'rationale' array)
        st.subheader("📋 Analysis / Rationale")
        for point in result.get("rationale", []):
            st.markdown(f"• {point}")
        
        # Show model-cited sources if any
        cited = result.get("cited_sources", [])
        if cited:
            st.subheader("🔎 Model-cited snippets")
            for c in cited:
                idx = c.get("idx")
                quote = c.get("quote_or_summary", "")
                ref = next((d for d in docs if d["idx"] == idx), None)
                if ref:
                    st.markdown(f"> {quote}\n\n— Source {idx}: [{ref['title']}]({ref['url']})")
        
        # Sources section
        st.markdown("---")
        st.subheader("📰 Sources Analyzed")
        for doc in docs:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**[{doc['title']}]({doc['url']})**")
                    meta = []
                    if doc.get("source"): meta.append(doc["source"])
                    if doc.get("published"): meta.append(doc["published"])
                    if meta: st.caption(" • ".join(meta))
                    
                    # Credibility badge
                    if doc["credibility"] > 0.7:
                        st.markdown(f'<span class="source-badge">👍 High Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)
                    elif doc["credibility"] > 0.5:
                        st.markdown(f'<span class="source-badge">⚠️ Medium Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="source-badge">❗ Low Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)
                    
                    # Preview expander (text truncated)
                    with st.expander("📖 Preview / Excerpt"):
                        preview_text = (doc.get("text") or "")[:2000]
                        st.text(preview_text if preview_text else "No extractable text; click source link to open article.")
                
                with col2:
                    st.write("")
        
        # Share and export section
        st.markdown("---")
        st.subheader("📤 Share Results")
        
        report = create_shareable_report(claim, result, docs)
        share_text = f"FactCheckAI analysis: '{claim[:60]}...' - Verdict: {result['verdict']}"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📋 Show Report", use_container_width=True):
                with st.expander("Full Report"):
                    st.code(report)
        with col2:
            st.download_button(
                "📥 Download",
                data=report,
                file_name=f"factcheck-{datetime.now().date()}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col3:
            twitter_url = f"https://twitter.com/intent/tweet?text={quote_plus(share_text)}"
            st.markdown(f"[🐦 Tweet result]({twitter_url})", unsafe_allow_html=True)
        with col4:
            wa_url = f"https://wa.me/?text={quote_plus(share_text)}"
            st.markdown(f"[💬 Share on WhatsApp]({wa_url})", unsafe_allow_html=True)
        
        # Educational footer
        st.markdown("---")
        st.markdown('<div class="educational-tip">💡 Remember: Always verify critical information with multiple reliable sources. This tool is an aid, not a replacement for critical thinking.</div>', unsafe_allow_html=True)

    elif not submitted:
        # Welcome state
        st.info("👆 Enter a claim above to start fact-checking. For best results, use specific claims with clear verification criteria.")
        
        # Example claims
        st.markdown("### 💡 Example Claims to Try:")
        examples = [
            "NASA discovered water on Mars",
            "Eating chocolate improves memory",
            "The Great Wall of China is visible from space",
            "COVID-19 vaccines contain microchips",
            "Shark attacks are more common than lightning strikes"
        ]
        
        for example in examples:
            if st.button(f"🔍 {example}", use_container_width=True):
                st.session_state.pre_filled = example
                st.experimental_rerun()

if __name__ == "__main__":
    main()
