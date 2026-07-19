# app.py
# Path: ./app.py
"""
FactCheckAI - Streamlit app for lightweight claim fact-checking with evidence,
refactored for clarity, typing, and robustness.
"""

from __future__ import annotations
import os
import time
import re
import math
import json
import random
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup

# Optional: better extraction where allowed
try:
    from newspaper import Article  # type: ignore
    NEWSPAPER_OK = True
except Exception:
    NEWSPAPER_OK = False

# -----------------------
# Gemini / genai setup
# -----------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def _safe_get_secret(key: str) -> Optional[str]:
    """Try st.secrets safely, fallback to environment variable."""
    try:
        # st.secrets may be malformed, guard against attribute errors
        val = None
        try:
            val = st.secrets.get(key)  # may raise if secrets file broken
        except Exception:
            val = None
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key)


GEMINI_API_KEY = _safe_get_secret("GEMINI_API_KEY")
gemini_client = None
GEMINI_INIT_ERROR: Optional[str] = None
if GEMINI_API_KEY:
    try:
        from google import genai as genai_module  # type: ignore
        from google.genai import types as genai_types  # type: ignore
        gemini_client = genai_module.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        GEMINI_INIT_ERROR = str(e)

# -----------------------
# Streamlit config & CSS
# -----------------------
st.set_page_config(
    page_title="FactCheckAI - Fake News Detector with Evidence & Sources",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root { --ink: #172033; --muted: #65728a; --line: #e2e8f0; --brand: #2563eb; --brand-dark: #1d4ed8; }
    .stApp { background: #f3f6fb; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(243,246,251,.9); border-bottom: 1px solid rgba(226,232,240,.8); }
    [data-testid="stSidebar"] { background: #0f172a; border-right: 0; }
    [data-testid="stSidebar"] > div:first-child { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #dbeafe; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div { background: #1e293b; border-color: #334155; }
    [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] { background-color: #60a5fa; }
    [data-testid="stSidebar"] hr { border-color: #26354d; }
    .block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin:0 0 .75rem; padding:.85rem 1.15rem; background:#ffffff; border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 14px rgba(15,23,42,.04); }
    .brand { display:flex; align-items:center; gap:.8rem; }
    .brand-mark { display:grid; place-items:center; width:40px; height:40px; border-radius:12px; color:#fff; font-size:1.1rem; background:linear-gradient(135deg, #635bff, #4338ca); box-shadow:0 8px 16px rgba(79,70,229,.22); }
    .brand-name { font-size:1.08rem; line-height:1.2; font-weight:750; letter-spacing:-.02em; color:var(--ink); }
    .brand-caption { font-size:.75rem; color:#7b879d; margin-top:2px; }
    .top-status { display:flex; align-items:center; gap:.5rem; padding:.45rem .7rem; border-radius:999px; background:#f0fdf4; color:#15803d; font-size:.75rem; font-weight:650; border:1px solid #dcfce7; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#10b981; }
    .hero { position:relative; overflow:hidden; padding:2.4rem 2.25rem 2rem; margin:0 0 1.55rem; background:linear-gradient(120deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%); border:0; border-radius:18px; box-shadow:0 16px 32px rgba(37,99,235,.18); }
    .hero:after { content:""; position:absolute; width:290px; height:290px; right:-90px; top:-130px; border:42px solid rgba(255,255,255,.11); border-radius:50%; }
    .hero-eyebrow { color:#bfdbfe; font-size:.72rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }
    .workspace-title { position:relative; z-index:1; max-width:650px; font-size:2.55rem; line-height:1.08; letter-spacing:-.055em; color:#fff; font-weight:750; margin:.55rem 0 0; }
    .workspace-copy { position:relative; z-index:1; max-width:610px; color:#dbeafe; font-size:1rem; line-height:1.55; margin:.75rem 0 0; }
    .hero-chips { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.3rem; }
    .hero-chip { padding:.35rem .65rem; border:1px solid rgba(255,255,255,.28); border-radius:999px; color:#eff6ff; font-size:.75rem; background:rgba(15,23,42,.12); }
    .sidebar-brand { margin:.25rem 0 1.55rem; padding:.25rem .2rem 1.25rem; border-bottom:1px solid #26354d; }
    .sidebar-brand h2 { margin:0; color:#fff; font-size:1.18rem; letter-spacing:-.02em; }
    .sidebar-brand p { margin:.35rem 0 0; color:#94a3b8; font-size:.78rem; }
    .side-label { color:#94a3b8; font-size:.7rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin:.2rem 0 .55rem; }
    .side-nav { display:grid; gap:.35rem; margin:0 0 1.35rem; }
    .side-nav-item { display:flex; align-items:center; gap:.6rem; padding:.62rem .7rem; border-radius:9px; color:#aebed5; font-size:.84rem; }
    .side-nav-item.active { background:#1e3a8a; color:#fff; font-weight:650; }
    .side-nav-icon { width:18px; text-align:center; color:#93c5fd; }
    .sidebar-tip { padding:.85rem .9rem; border-radius:12px; background:#172554; border:1px solid #1e3a8a; color:#bfdbfe; font-size:.85rem; line-height:1.45; }
    .sidebar-tip strong { color:#fff; }
    .main-header, .sub-header { display:none; }
    [data-testid="stSidebar"] h2 { display:none; }
    .sidebar-brand h2 { display:block !important; }
    .verdict-true { color: #2e8b57; font-weight: bold; font-size: 1.8rem; padding: 1.5rem; background: linear-gradient(135deg, #f0fff0 0%, #e0ffe0 100%); border-radius: 15px; border-left: 5px solid #2e8b57; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; }
    .verdict-false { color: #dc143c; font-weight: bold; font-size: 1.8rem; padding: 1.5rem; background: linear-gradient(135deg, #fff0f5 0%, #ffe0e6 100%); border-radius: 15px; border-left: 5px solid #dc143c; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; }
    .verdict-uncertain { color: #ff8c00; font-weight: bold; font-size: 1.8rem; padding: 1.5rem; background: linear-gradient(135deg, #fffaf0 0%, #fff5e0 100%); border-radius: 15px; border-left: 5px solid #ff8c00; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 1rem 0; }
    .evidence-card { border: 1px solid #e0e0e0; padding: 1.2rem; margin: 0.8rem 0; border-radius: 12px; background: #fafafa; transition: all 0.3s ease; }
    .evidence-card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15); background: #ffffff; }
    .source-badge { background: #e3f2fd; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; color: #1976d2; display: inline-block; margin: 0.2rem; }
    .confidence-bar { height: 8px; background: #f0f0f0; border-radius: 4px; margin: 0.5rem 0; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 4px; transition: width 1s ease-in-out; }
    .progress-container { margin: 2rem 0; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; border: 1px solid #e9ecef; }
    .share-buttons { display: flex; gap: 0.5rem; margin: 1rem 0; flex-wrap: wrap; }
    .educational-tip { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.2rem; border-radius: 12px; border-left: 4px solid #2196f3; margin: 1rem 0; }
    div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.4rem; box-shadow:0 8px 22px rgba(15,23,42,.055); }
    div[data-testid="stForm"] textarea { border-radius:10px; border-color:#cbd5e1; background:#fbfdff; }
    .stButton > button, .stFormSubmitButton > button { border-radius:9px; font-weight:650; min-height:2.55rem; }
    .stFormSubmitButton:first-child > button { background:var(--brand); border-color:var(--brand); color:#fff; }
    @media (max-width: 768px) { .workspace-title { font-size:2rem; } .top-status, .top-links { display:none; } .hero { padding:1.7rem 1.25rem 1.5rem; } .block-container { padding-top:1rem; } .verdict-true, .verdict-false, .verdict-uncertain { font-size: 1.5rem; padding: 1rem; } }
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
""",
    unsafe_allow_html=True,
)

EDUCATIONAL_TIPS = [
    "💡 Always verify information with multiple reliable sources before sharing",
    "🔍 Check the publication date - older information may be outdated or inaccurate",
    "🏛️ Government (.gov) and educational (.edu) sources tend to be more reliable",
    "📊 Look for original research and primary sources rather than interpretations",
    "⏰ Be wary of breaking news - initial reports often contain errors",
    "🌐 Consider the source's reputation and potential biases",
    "📝 Check if other reputable news organizations are reporting the same information",
    "🔎 Look for supporting evidence like data, studies, or expert opinions",
]

CREDIBLE_DOMAINS: Dict[str, float] = {
    "reuters.com": 0.95,
    "ap.org": 0.95,
    "bbc.com": 0.9,
    "bbc.co.uk": 0.9,
    "nytimes.com": 0.9,
    "theguardian.com": 0.9,
    "wsj.com": 0.9,
    ".gov": 0.95,
    ".edu": 0.9,
    ".ac.uk": 0.9,
    ".edu.au": 0.9,
    "who.int": 0.95,
    "un.org": 0.95,
    "nasa.gov": 0.95,
    "nih.gov": 0.95,
}

# -----------------------
# Helper functions
# -----------------------


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def fetch_google_news(query: str, region: str) -> Tuple[List[Dict], str]:
    """
    Fetch Google News RSS results; try region-specific and fallback.
    Returns (list_of_items, rss_url_used)
    """
    primary = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-{region}&gl={region}&ceid={region}:en"
    fallback = f"https://news.google.com/rss/search?q={quote_plus(query)}"
    for rss_url in (primary, fallback):
        feed = feedparser.parse(rss_url)
        if getattr(feed, "entries", None):
            results = []
            for entry in feed.entries:
                results.append(
                    {
                        "title": entry.get("title"),
                        "link": entry.get("link"),
                        "published": getattr(entry, "published", None),
                        "source": getattr(entry, "source", {}).get("title")
                        if hasattr(entry, "source")
                        else None,
                    }
                )
            return results, rss_url
    return [], fallback


@st.cache_data(show_spinner=False, ttl=1800, max_entries=50)
def extract_article_text(url: str, timeout: int = 8) -> str:
    """
    Extract article text with newspaper (optional) and BeautifulSoup fallback.
    Return empty string on failure.
    """
    try:
        if NEWSPAPER_OK:
            try:
                art = Article(url)
                art.download()
                art.parse()
                text = (art.text or "").strip()
                if text and len(text.split()) > 50:
                    return text
            except Exception:
                # newspaper occasionally fails on certain hosts; fall through
                pass

        headers = {"User-Agent": "Mozilla/5.0 (compatible; FactCheckAI/1.0; +https://github.com/yourusername/factcheck-ai)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        selectors = ["article", "main", "[itemprop='articleBody']", ".article-content", ".post-content", ".story-content"]
        for sel in selectors:
            elems = soup.select(sel)
            if elems:
                text = " ".join(e.get_text(" ", strip=True) for e in elems)
                if len(text.split()) > 50:
                    return text

        # fallback: whole page text
        page_text = soup.get_text(" ", strip=True)
        return page_text
    except Exception:
        return ""


def parse_pubdate_safe(date_str: Optional[str]) -> Optional[datetime]:
    """Attempt to parse published date from feedparser; fall back to None"""
    if not date_str:
        return None
    try:
        parsed = feedparser._parse_date(date_str)
        if parsed:
            return datetime(*parsed[:6])
    except Exception:
        pass
    # try common ISO fallback
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def rate_source_credibility(url: str, content: str) -> float:
    """
    Rate credibility (0.0 - 1.0) from domain heuristics + content signals.
    Keep conservative defaults to avoid overconfidence.
    """
    credibility = 0.5
    url_l = (url or "").lower()
    for pattern, score in CREDIBLE_DOMAINS.items():
        if pattern in url_l:
            credibility = max(credibility, score)
    if len((content or "").split()) > 200:
        credibility = min(credibility + 0.1, 1.0)
    if any(k in (content or "").lower() for k in ("study", "research", "data", "according to", "experts say")):
        credibility = min(credibility + 0.05, 1.0)
    return credibility


def trim_text(text: str, max_chars: int = 4000) -> str:
    """Trim text to nearest sentence boundary under max_chars."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    # find last sentence terminator
    m = re.findall(r"^(.+?[.!?])\s", cut, flags=re.S)
    return m[-1] if m else cut[:max_chars]


def make_prompt_for_gemini(claim: str, evidence_items: List[str]) -> str:
    """Build prompt that asks for JSON structured response."""
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


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract the first JSON-like object from text. Attempt minor fixes.
    """
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    m = re.search(r"(\{[\s\S]*\})", t)
    if m:
        jtxt = m.group(1)
        try:
            return json.loads(jtxt)
        except Exception:
            # attempt safe single->double quotes replace
            try:
                return json.loads(jtxt.replace("'", '"'))
            except Exception:
                return None
    try:
        return json.loads(t)
    except Exception:
        return None


def fallback_rule_based_analysis(claim: str, docs: List[Dict]) -> Dict:
    """
    Deterministic fallback when there is no LLM. Uses keyword matching and naive contradiction signals.
    """
    keywords = [w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3]
    if not keywords:
        keywords = [w.lower() for w in claim.split() if len(w) > 3]
    total = max(1, len(docs))
    support = 0
    contradict = 0
    top_sources = []
    scores = []
    for d in docs:
        txt = ((d.get("title") or "") + " " + (d.get("text") or "")).lower()
        kcount = sum(1 for k in keywords if k in txt)
        if kcount > 0:
            support += 1
        if any(phrase in txt for phrase in ["no evidence", "not true", "debunk", "false", "denied", "not found", "refute"]):
            contradict += 1
        scores.append(d.get("credibility", 0.5))
        excerpt = (d.get("text") or d.get("title") or "")[:280]
        top_sources.append({"idx": d.get("idx"), "quote_or_summary": excerpt, "relevance": "med"})
    avg_cred = sum(scores) / len(scores)
    if support >= max(1, math.ceil(total * 0.6)) and avg_cred > 0.6:
        verdict = "Likely True"
        confidence = min(0.9, avg_cred)
        rationale = [
            f"{support}/{total} sources mention the claim or related keywords.",
            f"Average source credibility is {avg_cred:.2f}. Top sources support the claim.",
            "No strong contradictory language found in majority of sources.",
        ]
    elif contradict >= max(1, math.ceil(total * 0.5)):
        verdict = "Likely False"
        confidence = min(0.85, 0.5 + (avg_cred / 2))
        rationale = [
            f"{contradict}/{total} sources contain language indicating the claim was refuted or denied.",
            f"Average source credibility is {avg_cred:.2f}.",
            "Contradictory phrasing suggests claim is likely false or misrepresented.",
        ]
    else:
        verdict = "Uncertain"
        confidence = min(0.6, avg_cred)
        rationale = [
            "Evidence is mixed or insufficient to make a confident call.",
            f"{support}/{total} sources mention claim keywords; {contradict} show refutation-like language.",
            "Consider more specific search terms or primary sources.",
        ]
    return {"verdict": verdict, "confidence": confidence, "rationale": rationale, "cited_sources": top_sources[:3]}


@st.cache_data(show_spinner=False, ttl=600, max_entries=20)
def reason_with_gemini(claim: str, docs: List[Dict], temperature: float = 0.3) -> Dict:
    """
    Use Gemini (if configured) to reason over pieces of evidence.
    Falls back to deterministic analysis if Gemini unavailable or parsing fails.
    """
    if not gemini_client:
        return fallback_rule_based_analysis(claim, docs)
    try:
        texts = [d.get("text") or d.get("title", "") for d in docs]
        prompt = make_prompt_for_gemini(claim, texts)
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        output = (resp.text or "").strip()
        parsed = extract_json_from_text(output)
        if parsed:
            parsed.setdefault("verdict", parsed.get("verdict", "Uncertain"))
            parsed.setdefault("confidence", float(parsed.get("confidence", 0.5)))
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
        # try to salvage if JSON not returned
        v_match = re.search(r"(?i)(likely true|likely false|uncertain)", output)
        verdict = v_match.group(0).title() if v_match else "Uncertain"
        c_match = re.search(r"(\d?\.\d+|\d+)%", output)
        if c_match:
            conf = float(c_match.group(1).replace("%", "")) / 100.0
        else:
            num_match = re.search(r"confidence[:\s]*([0-1](?:\.\d+)?)", output, re.I)
            conf = float(num_match.group(1)) if num_match else 0.5
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        rationale = lines[:3] if lines else ["Model returned text but not parseable JSON."]
        cited = []
        for d in docs[:3]:
            cited.append({"idx": d.get("idx"), "quote_or_summary": (d.get("text") or d.get("title"))[:280], "relevance": "med"})
        return {"verdict": verdict, "confidence": conf, "rationale": rationale, "cited_sources": cited}
    except Exception as exc:
        st.warning(f"Gemini could not complete this check ({exc}). Using the evidence-only fallback.")
        return fallback_rule_based_analysis(claim, docs)


def create_shareable_report(claim: str, result: Dict, sources: List[Dict]) -> str:
    """Create a brief plain-text report suitable for download or sharing."""
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


# -----------------------
# Main app
# -----------------------
def main() -> None:
    # session state defaults
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("current_tip", random.choice(EDUCATIONAL_TIPS))
    st.session_state.setdefault("last_request", 0.0)
    st.session_state.setdefault("pre_filled", "")

    st.markdown(
        '''<div class="topbar">
            <div class="brand"><div class="brand-mark">&#10003;</div><div><div class="brand-name">FactCheckAI</div><div class="brand-caption">Evidence-first verification</div></div></div>
            <div class="top-status"><span class="status-dot"></span>Ready to verify</div>
        </div>
        <section class="hero">
            <div class="hero-eyebrow">Evidence, not noise</div>
            <h1 class="workspace-title">Find the facts behind every claim.</h1>
            <p class="workspace-copy">Paste a headline or statement below. We will compare it with recent reporting and credible sources, then show the evidence clearly.</p>
            <div class="hero-chips"><span class="hero-chip">Source analysis</span><span class="hero-chip">Transparent verdicts</span><span class="hero-chip">Current reporting</span></div>
        </section>''',
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="main-header">🔍 FactCheckAI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify claims with evidence-based analysis and transparent sourcing</p>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.markdown('''<div class="sidebar-brand"><h2>FactCheckAI</h2><p>Evidence verification workspace</p></div><div class="side-nav"><div class="side-nav-item active"><span class="side-nav-icon">+</span>New verification</div><div class="side-nav-item"><span class="side-nav-icon">#</span>Recent checks</div></div><div class="side-label">Search settings</div>''', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        region = st.selectbox("News Region", ["US", "GB", "IN", "AU", "CA", "DE", "FR", "SG"], index=0)
        freshness_hours = st.slider("Freshness (hours)", 6, 168, 48)
        max_articles = st.slider("Max Articles", 3, 15, 8)
        temperature = st.slider("Analysis Creativity", 0.0, 1.0, 0.3, 0.1)

        st.divider()
        st.header("📚 Recent Checks")
        if st.session_state["history"]:
            for i, item in enumerate(st.session_state["history"][:5]):
                with st.expander(f"{item['claim'][:40]}...", expanded=(i == 0)):
                    st.caption(f"**Verdict:** {item['result']['verdict']}")
                    st.caption(f"**Confidence:** {item['result']['confidence']:.0%}")
                    st.caption(f"**Sources:** {item['sources_count']}")
                    if st.button("🔍 Review", key=f"review_{i}"):
                        st.session_state["pre_filled"] = item["claim"]
                        st.experimental_rerun()
        else:
            st.info("No recent checks yet")

        st.divider()
        st.markdown(f"**💡 Fact-Checking Tip:**\n\n{st.session_state['current_tip']}")

    # Input form
    with st.form("claim_form"):
        claim = st.text_area(
            "Paste headline or claim to verify:",
            value=st.session_state.get("pre_filled", ""),
            placeholder="Enter the claim you want to fact-check...",
            height=100,
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            submitted = st.form_submit_button("🔎 Verify Claim", use_container_width=True)
        with col2:
            clear_clicked = st.form_submit_button("🔄 Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.clear()
        st.experimental_rerun()

    if submitted and claim.strip():
        now = time.time()
        if now - float(st.session_state.get("last_request", 0)) < 2:
            st.warning("⏳ Please wait a few seconds between requests")
            st.stop()
        st.session_state["last_request"] = now

        progress = st.progress(0)
        status = st.empty()

        stages = [
            "🔍 Searching news sources...",
            "📰 Fetching article content...",
            "🧠 Analyzing evidence...",
            "⚖️ Generating verdict...",
        ]

        # Stage 1: search
        progress.progress(10)
        status.text(stages[0])
        items, used_rss = fetch_google_news(claim, region)
        st.markdown("**🔎 RSS search URL used:**")
        st.code(used_rss)
        st.markdown("**🔎 Results returned by RSS (titles & published dates):**")
        if items:
            for i, e in enumerate(items[: max(50, len(items))], 1):
                st.markdown(f"{i}. [{e.get('title')}]({e.get('link')}) — {e.get('published') or 'no published date'}")
        else:
            st.warning("RSS returned no entries. Try a different query or region.")

        # Filter by freshness (keep items without published date)
        cutoff = datetime.utcnow() - timedelta(hours=freshness_hours)
        filtered = []
        for it in items:
            dt = parse_pubdate_safe(it.get("published"))
            if dt is None or dt >= cutoff:
                filtered.append(it)
        filtered = filtered[:max_articles]

        if not filtered:
            st.error("❌ No recent articles found after filtering. Try widening the time window or rephrasing the claim.")
            st.stop()

        # Stage 2: extract contents
        progress.progress(40)
        status.text(stages[1])

        docs: List[Dict] = []
        max_workers = min(3, max(1, len(filtered)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {executor.submit(extract_article_text, item["link"]): item for item in filtered}
            idx = 0
            for fut in as_completed(future_to_item):
                item = future_to_item[fut]
                try:
                    text = fut.result(timeout=12)
                except Exception:
                    text = ""
                idx += 1
                cred = rate_source_credibility(item.get("link", ""), text)
                docs.append(
                    {
                        "idx": idx,
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "published": item.get("published"),
                        "source": item.get("source"),
                        "text": text,
                        "credibility": cred,
                    }
                )

        if not docs:
            st.error("❌ Could not extract content from articles. Please try a different claim.")
            st.stop()

        # Stage 3: analysis
        progress.progress(75)
        status.text(stages[2])
        result = reason_with_gemini(claim, docs, temperature)

        # Stage 4: done
        progress.progress(100)
        status.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress.empty()
        status.empty()

        # store history
        st.session_state["history"].insert(
            0,
            {
                "claim": claim,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "sources_count": len(docs),
            },
        )

        # Display results
        st.markdown("---")
        verdict_text = result.get("verdict", "Uncertain")
        verdict_lower = verdict_text.lower()
        confidence = float(result.get("confidence", 0.5))

        if "true" in verdict_lower:
            st.markdown(f'<div class="verdict-true">✅ Verdict: {verdict_text}</div>', unsafe_allow_html=True)
        elif "false" in verdict_lower:
            st.markdown(f'<div class="verdict-false">❌ Verdict: {verdict_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-uncertain">⚠️ Verdict: {verdict_text}</div>', unsafe_allow_html=True)

        col1_disp, col2_disp = st.columns([1, 3])
        with col1_disp:
            st.metric("Confidence", f"{confidence:.0%}")
        with col2_disp:
            color = "#2e8b57" if confidence > 0.7 else "#ff8c00" if confidence > 0.4 else "#dc143c"
            st.markdown(
                f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.subheader("📋 Analysis / Rationale")
        for point in result.get("rationale", []):
            st.markdown(f"• {point}")

        cited = result.get("cited_sources", [])
        if cited:
            st.subheader("🔎 Model-cited snippets")
            for c in cited:
                idx = c.get("idx")
                quote = c.get("quote_or_summary", "")
                ref = next((d for d in docs if d["idx"] == idx), None)
                if ref:
                    st.markdown(f"> {quote}\n\n— Source {idx}: [{ref['title']}]({ref['url']})")

        st.markdown("---")
        st.subheader("📰 Sources Analyzed")
        for doc in docs:
            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**[{doc['title']}]({doc['url']})**")
                    meta = []
                    if doc.get("source"):
                        meta.append(doc["source"])
                    if doc.get("published"):
                        meta.append(doc["published"])
                    if meta:
                        st.caption(" • ".join(meta))
                    if doc["credibility"] > 0.7:
                        st.markdown(f'<span class="source-badge">👍 High Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)
                    elif doc["credibility"] > 0.5:
                        st.markdown(f'<span class="source-badge">⚠️ Medium Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="source-badge">❗ Low Credibility ({doc["credibility"]:.0%})</span>', unsafe_allow_html=True)

                    with st.expander("📖 Preview / Excerpt"):
                        preview_text = (doc.get("text") or "")[:2000]
                        st.text(preview_text if preview_text else "No extractable text; click source link to open article.")
                with c2:
                    st.write("")

        st.markdown("---")
        st.subheader("📤 Share Results")
        report = create_shareable_report(claim, result, docs)
        share_text = f"FactCheckAI analysis: '{claim[:60]}...' - Verdict: {result['verdict']}"

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if st.button("📋 Show Report", use_container_width=True):
                with st.expander("Full Report"):
                    st.code(report)
        with col_b:
            st.download_button("📥 Download", data=report, file_name=f"factcheck-{datetime.now().date()}.txt", mime="text/plain", use_container_width=True)
        with col_c:
            twitter_url = f"https://twitter.com/intent/tweet?text={quote_plus(share_text)}"
            st.markdown(f"[🐦 Tweet result]({twitter_url})", unsafe_allow_html=True)
        with col_d:
            wa_url = f"https://wa.me/?text={quote_plus(share_text)}"
            st.markdown(f"[💬 Share on WhatsApp]({wa_url})", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="educational-tip">💡 Remember: Always verify critical information with multiple reliable sources. This tool is an aid, not a replacement for critical thinking.</div>', unsafe_allow_html=True)

    else:
        st.info("👆 Enter a claim above to start fact-checking. For best results, use specific claims with clear verification criteria.")
        st.markdown("### 💡 Example Claims to Try:")
        examples = [
            "NASA discovered water on Mars",
            "Eating chocolate improves memory",
            "The Great Wall of China is visible from space",
            "COVID-19 vaccines contain microchips",
            "Shark attacks are more common than lightning strikes",
        ]
        for ex in examples:
            if st.button(f"🔍 {ex}", use_container_width=True):
                st.session_state["pre_filled"] = ex
                st.experimental_rerun()


if __name__ == "__main__":
    main()
