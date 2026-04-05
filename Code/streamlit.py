import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re

# =========================================================
# SAFE IMPORT: интерфейс всегда открывается
# =========================================================
PIPELINE_ERROR = None
analyze_text = None

try:
    from pipelineFinal import analyze_text
    PIPELINE_READY = True
except Exception as e:
    PIPELINE_READY = False
    PIPELINE_ERROR = str(e)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="KazCausal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB examples ──────────────────────────────────────────────────
DB_EXAMPLES = [
    {
        "group": "SYNTHETIC",
        "text":   "Қысқа да әрі нақты, көпке түсінікті болғандықтан студенттерге бір деммен оқу қиынға соқпауы мүмкін.",
        "cause":  "Қысқа да әрі нақты, көпке түсінікті болғандықтан",
        "effect": "студенттерге бір деммен оқу қиынға соқпауы мүмкін.",
        "marker": "болғандықтан",
        "tv_form": "Tv=ғандықтан, гендіктен, қандықтан, кендіктен",
    },
    {
        "group":  "SYNTHETIC",
        "text":   "Əрбір тайпа бұл киелі міндетке тек өздерін лайықты деп білгендіктен, араларында дау-жанжал шықты.",
        "cause":  "Əрбір тайпа бұл киелі міндетке тек өздерін лайықты деп білгендіктен",
        "effect": "араларында дау-жанжал шықты.",
        "marker": "білгендіктен",
        "tv_form": "Tv=ғандықтан, гендіктен, қандықтан, кендіктен",
    },
    {
        "group":  "SYNTHETIC",
        "text":   "Таңдау мүмкіндігі, әралуандық болғандықтан оларды қолдануда едәуір еркіндік бар.",
        "cause":  "Таңдау мүмкіндігі, әралуандық болғандықтан",
        "effect": "оларды қолдануда едәуір еркіндік бар.",
        "marker": "болғандықтан",
        "tv_form": "Tv=ғандықтан, гендіктен, қандықтан, кендіктен",
    },
    {
        "group":  "SYNTHETIC",
        "text":   "Мен мұғжиза ретінде тамақтың азаймағанын көргендіктен тағы да алпыс адамды шақырып келдім.",
        "cause":  "Мен мұғжиза ретінде тамақтың азаймағанын көргендіктен",
        "effect": "тағы да алпыс адамды шақырып келдім.",
        "marker": "көргендіктен",
        "tv_form": "Tv=ғандықтан, гендіктен, қандықтан, кендіктен",
    },
    {
        "group":  "SYNTHETIC",
        "text":   "Сұрақ беріледі, оның «дұрыс» жауабы алдын-ала белгілі болғандықтан, ақиқат» сол сұрақтың өзінде.",
        "cause":  "Сұрақ беріледі, оның «дұрыс» жауабы алдын-ала белгілі болғандықтан",
        "effect": "ақиқат» сол сұрақтың өзінде.",
        "marker": "болғандықтан",
        "tv_form": "Tv=ғандықтан, гендіктен, қандықтан, кендіктен",
    },
    {
        "group":  "ANALYTIC",
        "text":   "Қоғамдық сенім деңгейі төмендеді, себебі мемлекеттік институттардың ашықтығы жеткіліксіз қамтамасыз етілді.",
        "cause":  "мемлекеттік институттардың ашықтығы жеткіліксіз қамтамасыз етілді.",
        "effect": "Қоғамдық сенім деңгейі төмендеді",
        "marker": "себебі",
        "tv_form": "[(N1) Tv =fin] себебі [(N1) Vfin.]",
    },
    {
        "group":  "ANALYTIC",
        "text":   "Сот шешімдерінің сапасына сын айтылды, себебі дәлелдемелерді бағалау рәсімі бірізді жүргізілмеді.",
        "cause":  "дәлелдемелерді бағалау рәсімі бірізді жүргізілмеді.",
        "effect": "Сот шешімдерінің сапасына сын айтылды",
        "marker": "себебі",
        "tv_form": "[(N1) Tv =fin] себебі [(N1) Vfin.]",
    },
    {
        "group":  "ANALYTIC",
        "text":   "Әлеуметтік теңсіздік күшейді, себебі ресурстарды бөлу тетіктері тиімді жұмыс істемеді.",
        "cause":  "ресурстарды бөлу тетіктері тиімді жұмыс істемеді.",
        "effect": "Әлеуметтік теңсіздік күшейді",
        "marker": "себебі",
        "tv_form": "[(N1) Tv =fin] себебі [(N1) Vfin.]",
    },
    {
        "group":  "ANALYTIC",
        "text":   "Білім беру нәтижелері төмендеді, себебі оқу бағдарламалары заманауи талаптарға толық сәйкес келмеді.",
        "cause":  "оқу бағдарламалары заманауи талаптарға толық сәйкес келмеді.",
        "effect": "Білім беру нәтижелері төмендеді",
        "marker": "себебі",
        "tv_form": "[(N1) Tv =fin] себебі [(N1) Vfin.]",
    },
    {
        "group":  "ANALYTIC",
        "text":   "Ғылыми жарияланым сапасы әркелкі болды, себебі рецензиялау жүйесі қатаң сақталмады.",
        "cause":  "рецензиялау жүйесі қатаң сақталмады.",
        "effect": "Ғылыми жарияланым сапасы әркелкі болды",
        "marker": "себебі",
        "tv_form": "[(N1) Tv =fin] себебі [(N1) Vfin.]",
    },
    {
        "group":  "ANALYTICO_SYNTHETIC",
        "text":   "Сот шешімінің дәлелді жазылғанына көз жеткізгеніне орай, тараптар апелляциялық шағым беруден бас тартты.",
        "cause":  "Сот шешімінің дәлелді жазылғанына көз жеткізгеніне орай",
        "effect": "тараптар апелляциялық шағым беруден бас тартты.",
        "marker": "жазылғанына",
        "tv_form": "Tv=ған=//=на, ген=//=не, қан=//=на, кен=//=не",
    },
    {
        "group":  "ANALYTICO_SYNTHETIC",
        "text":   "Сот шешімі заңды күшіне енген соң, атқарушылық іс жүргізу басталды.",
        "cause":  "Сот шешімі заңды күшіне енген соң",
        "effect": "атқарушылық іс жүргізу басталды.",
        "marker": "енген соң",
        "tv_form": "Tv=ған соң, ген соң, қан соң, кен соң",
    },
    {
        "group":  "ANALYTICO_SYNTHETIC",
        "text":   "Ғылыми зерттеу аяқталған соң, оның нәтижелері халықаралық журналда жарияланды.",
        "cause":  "Ғылыми зерттеу аяқталған соң",
        "effect": "оның нәтижелері халықаралық журналда жарияланды.",
        "marker": "аяқталған соң",
        "tv_form": "Tv=ған соң, ген соң, қан соң, кен соң",
    },
    {
        "group":  "ANALYTICO_SYNTHETIC",
        "text":   "Заң жобасы қабылданған соң, нормативтік-құқықтық актілерге тиісті өзгерістер енгізілді.",
        "cause":  "Заң жобасы қабылданған соң",
        "effect": "нормативтік-құқықтық актілерге тиісті өзгерістер енгізілді.",
        "marker": "қабылданған соң",
        "tv_form": "Tv=ған соң, ген соң, қан соң, кен соң",
    },
    {
        "group":  "ANALYTICO_SYNTHETIC",
        "text":   "Әлеуметтік сауалнама жүргізілген соң, деректер кешенді талдаудан өткізілді.",
        "cause":  "Әлеуметтік сауалнама жүргізілген соң",
        "effect": "деректер кешенді талдаудан өткізілді.",
        "marker": "жүргізілген соң",
        "tv_form": "Tv=ған соң, ген соң, қан соң, кен соң",
    },
]

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:     #0D0F14;
    --surf:   #141720;
    --border: #252A36;
    --accent: #4F8EF7;
    --cause:  #3DDC84;
    --effect: #F7794F;
    --marker: #F7CF4F;
    --text:   #E8ECF4;
    --muted:  #6B7590;
    --rad:    10px;
}
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: var(--bg); color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 4rem; max-width: 1440px; }
.topbar {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0 22px; border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.logo {
    font-size: 20px; font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--cause));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: .08em; margin-left: auto;
}
.pill {
    display: inline-flex; align-items: center;
    padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
    background: rgba(79,142,247,.12); color: var(--accent);
    border: 1px solid rgba(79,142,247,.25);
}
.anno {
    font-family: 'JetBrains Mono', monospace; font-size: 15px;
    line-height: 2.5; background: var(--surf);
    border: 1px solid var(--border); border-radius: var(--rad);
    padding: 20px 24px; word-break: break-word;
}
.sp-cause  { background: rgba(61,220,132,.18); color:#3DDC84; border-bottom:2px solid #3DDC84; border-radius:3px; padding:1px 2px; }
.sp-effect { background: rgba(247,121,79,.18);  color:#F7794F; border-bottom:2px solid #F7794F; border-radius:3px; padding:1px 2px; }
.sp-marker { background: rgba(247,207,79,.28);  color:#F7CF4F; border-bottom:2px solid #F7CF4F; border-radius:3px; padding:1px 2px; font-weight:700; }
.sp-lbl    { font-size:9px; font-weight:800; vertical-align:super; margin-left:1px; letter-spacing:.04em; opacity:.85; }
.leg { display:flex; gap:18px; margin:10px 0 20px; flex-wrap:wrap; }
.leg-item  { display:flex; align-items:center; gap:7px; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; }
.lsq       { width:11px; height:11px; border-radius:3px; flex-shrink:0; }
.sec       { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin:18px 0 8px; }
.result-card-header { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:10px; }
.jblock {
    background:#080A0F; border:1px solid var(--border); border-radius:var(--rad);
    padding:16px 18px; font-family:'JetBrains Mono',monospace;
    font-size:12.5px; line-height:1.9; overflow-x:auto; white-space:pre; color:#8EC07C;
}
.hist-m { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); margin-top:2px; margin-bottom:6px; }
.divl   { height:1px; background:var(--border); margin:18px 0; }
.stButton > button {
    background: linear-gradient(135deg,#2563EB,#1E40AF) !important;
    color:#fff !important; border:none !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    padding:10px 24px !important; letter-spacing:.03em !important;
}
[data-testid="stSidebar"] { background:var(--surf) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color:var(--text) !important; }
textarea {
    background:var(--surf) !important; color:var(--text) !important;
    border:1px solid var(--border) !important; border-radius:var(--rad) !important;
    font-family:'JetBrains Mono',monospace !important; font-size:13.5px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
for k, v in [("history", []), ("results", []), ("input_text", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────
MG_COL = {
    "SYNTHETIC":           "#4F8EF7",
    "ANALYTIC":            "#A78BFA",
    "ANALYTICO_SYNTHETIC": "#F97316",
}
GROUP_LABEL = {
    "SYNTHETIC":           "⚙ SYNTHETIC",
    "ANALYTIC":            "📎 ANALYTIC",
    "ANALYTICO_SYNTHETIC": "🔀 ANALYTICO-SYNTHETIC",
}


def make_result_from_db(ex):
    return {
        "text":                   ex["text"],
        "cause":                  ex["cause"],
        "effect":                 ex["effect"],
        "markers":                [ex["marker"]],
        "tv_form":                ex["tv_form"],
        "tv_confidence":          0.997,
        "semantic_type":          "—",
        "semantic_confidence":    None,
        "semantic_all":           {},
        "model_group":            ex["group"],
        "model_group_confidence": 0.999,
        "timestamp":              datetime.now().strftime("%H:%M:%S"),
        "error":                  None,
        "_from_db":               True,
    }


def annotate_html(text, cause_text, effect_text, markers):
    n = len(text)
    prio = [None] * n

    def mark(substr, css, lbl, p):
        if not substr:
            return
        idx = 0
        tl = text.lower()
        sl = substr.lower().strip()
        while True:
            pos = tl.find(sl, idx)
            if pos == -1:
                break
            for i in range(pos, pos + len(sl)):
                if prio[i] is None or prio[i][0] < p:
                    prio[i] = (p, css, lbl)
            idx = pos + 1

    mark(cause_text,  "sp-cause",  "CAUSE",  2)
    mark(effect_text, "sp-effect", "EFFECT", 1)
    for m in (markers or []):
        mark(m, "sp-marker", "MRK", 3)

    html = ""
    i = 0
    while i < n:
        cell = prio[i]
        if cell is None:
            j = i
            while j < n and prio[j] is None:
                j += 1
            html += text[i:j].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            i = j
        else:
            _, css, lbl = cell
            j = i
            while j < n and prio[j] is not None and prio[j][1] == css:
                j += 1
            chunk = text[i:j].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html += f'<span class="{css}">{chunk}<span class="sp-lbl">{lbl}</span></span>'
            i = j

    return f'<div class="anno">{html}</div>'


def to_table_rows(r):
    markers = r.get("markers") or []
    marker_str = ", ".join(str(m) for m in markers) or "—"

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    rows = [
        {"Field": "cause", "Value": fmt(r.get("cause"))},
        {"Field": "effect", "Value": fmt(r.get("effect"))},
        {"Field": "markers", "Value": marker_str},
        {"Field": "tv_form", "Value": fmt(r.get("tv_form"))},
        {"Field": "tv_confidence", "Value": fmt(r.get("tv_confidence"))},
        {"Field": "semantic_type", "Value": fmt(r.get("semantic_type"))},
        {"Field": "semantic_confidence", "Value": fmt(r.get("semantic_confidence"))},
    ]

    sem_all = r.get("semantic_all") or {}
    for label, score in sem_all.items():
        rows.append({"Field": f"  ↳ {label}", "Value": f"{score:.4f}"})

    rows += [
        {"Field": "model_group", "Value": fmt(r.get("model_group"))},
        {"Field": "mg_confidence", "Value": fmt(r.get("model_group_confidence"))},
    ]
    return rows


def _serialize(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {dk: _serialize(dv) for dk, dv in v.items()}
    if isinstance(v, list):
        return [_serialize(item) for item in v]
    return v


def clean_json(r):
    return {k: _serialize(v) for k, v in r.items() if not k.startswith("_")}


def render_result_card(r, idx=None):
    label = f"Sentence {idx + 1}: " if idx is not None else ""
    mg = r.get("model_group", "—")
    col = MG_COL.get(mg, "#888")

    st.markdown(
        f'<div class="result-card-header">{label}<span style="color:{col}">{mg}</span></div>',
        unsafe_allow_html=True,
    )

    if r.get("_from_db"):
        st.success("✅ Ground-truth annotation from examples")

    st.markdown('<div class="sec">Annotated text</div>', unsafe_allow_html=True)
    cause_t = r.get("cause")
    effect_t = r.get("effect")
    markers_l = r.get("markers") or []

    if cause_t or effect_t or markers_l:
        st.markdown(annotate_html(r["text"], cause_t, effect_t, markers_l), unsafe_allow_html=True)
    else:
        plain = r["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        st.markdown(f'<div class="anno" style="color:var(--muted);">{plain}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">Extracted fields</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(to_table_rows(r)), width="stretch", hide_index=True, height=420)

    st.markdown('<div class="sec">JSON output</div>', unsafe_allow_html=True)
    js = json.dumps(clean_json(r), ensure_ascii=False, indent=2)
    st.markdown(f'<div class="jblock">{js}</div>', unsafe_allow_html=True)


def demo_annotate_text(txt: str):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', txt) if s.strip()]
    results = []

    demo_markers_analytic = ["өйткені", "себебі", "неге десең", "неге десеңіз"]
    demo_markers_analytic_synth = ["соң", "кейін"]
    demo_markers_all = [
        "өйткені", "себебі", "сондықтан", "сол себепті", "сол үшін",
        "неге десең", "неге десеңіз",
        "болғандықтан", "болмағандықтан", "білгендіктен", "көргендіктен",
        "жүргізілгендіктен", "алғандықтан", "берілгендіктен", "туғандықтан",
        "үшін", "соң", "кейін"
    ]

    for s in sents:
        low = s.lower()
        found_marker = None

        for m in sorted(demo_markers_all, key=len, reverse=True):
            if m in low:
                found_marker = m
                break

        if found_marker:
            pos = low.find(found_marker)
            left = s[:pos].strip(" ,")
            right = s[pos + len(found_marker):].strip(" ,")

            if found_marker in demo_markers_analytic:
                cause = right
                effect = left
                model_group = "ANALYTIC"
            elif found_marker in demo_markers_analytic_synth or "соң" in found_marker or "кейін" in found_marker:
                cause = s[:pos + len(found_marker)].strip(" ,")
                effect = right
                model_group = "ANALYTICO_SYNTHETIC"
            else:
                cause = s[:pos + len(found_marker)].strip(" ,")
                effect = right
                model_group = "SYNTHETIC"

            results.append({
                "text": s,
                "cause": cause,
                "effect": effect,
                "markers": [found_marker],
                "tv_form": "— (demo mode)",
                "tv_confidence": None,
                "semantic_type": "—",
                "semantic_confidence": None,
                "semantic_all": {},
                "model_group": model_group,
                "model_group_confidence": None,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "error": PIPELINE_ERROR if not PIPELINE_READY else None,
            })
        else:
            results.append({
                "text": s,
                "cause": None,
                "effect": None,
                "markers": [],
                "tv_form": "— (demo mode)",
                "tv_confidence": None,
                "semantic_type": "—",
                "semantic_confidence": None,
                "semantic_all": {},
                "model_group": "— (demo mode)",
                "model_group_confidence": None,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "error": PIPELINE_ERROR if not PIPELINE_READY else None,
                "_demo_blank": True,
            })

    return results


# ══════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="logo">⚡ KazCausal</div>
  <span class="pill">KazBERT</span>
  <span class="pill">Streamlit</span>
  <div class="sub">Causal Relation Extraction · Kazakh NLP</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Баптаулар")

    if PIPELINE_READY:
        demo_mode = st.toggle(
            "Demo mode", value=True,
            help="Қосулы тұрса — мысалдар және ереже арқылы аннотация. Өшірулі тұрса — real model жұмыс істейді."
        )
        st.success("Pipeline жүктелді.")
    else:
        demo_mode = True
        st.warning("Pipeline жүктелмеді. Қазір demo mode ғана жұмыс істейді.")
        st.markdown("**Қате:**")
        st.code(PIPELINE_ERROR)

    st.markdown("<div class='divl'></div>", unsafe_allow_html=True)
    st.markdown("### 🕓 Тарих")
    if not st.session_state.history:
        st.markdown('<p style="color:#6B7590;font-size:12px">Әлі сұраныс жоқ.</p>', unsafe_allow_html=True)
    else:
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            txt, res_list = h
            short = txt[:50] + ("…" if len(txt) > 50 else "")
            mg = res_list[0].get("model_group", "—") if res_list else "—"
            col = MG_COL.get(mg, "#888")
            if st.button(short, key=f"h{i}", use_container_width=True):
                st.session_state.results = res_list
                st.session_state.input_text = txt
                st.rerun()
            st.markdown(
                f'<div class="hist-m">🕐 {res_list[0].get("timestamp","") if res_list else ""}'
                f' &nbsp;·&nbsp; <span style="color:{col}">{mg}</span>'
                f'{"  +" + str(len(res_list)-1) + " more" if len(res_list) > 1 else ""}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div class='divl'></div>", unsafe_allow_html=True)
        if st.button("🗑 Тарихты тазалау", use_container_width=True):
            st.session_state.history = []
            st.session_state.results = []
            st.rerun()

# ── Two columns ──────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ─────────────────── LEFT ───────────────────────────────────────
with left:
    st.markdown("#### ✏️ Текст енгізіңіз!")

    options = ["— өз мәтініңізді жазыңыз —"]
    opt_map = {}
    for group, glabel in GROUP_LABEL.items():
        for ex in [e for e in DB_EXAMPLES if e["group"] == group]:
            preview = ex["text"][:58] + ("…" if len(ex["text"]) > 58 else "")
            label = f"[{glabel}]  {preview}"
            options.append(label)
            opt_map[label] = ex

    sel = st.selectbox("Мысалдар", options, label_visibility="collapsed")

    if "last_sel" not in st.session_state:
        st.session_state.last_sel = None

    if sel != options[0] and sel in opt_map and sel != st.session_state.last_sel:
        chosen_ex = opt_map[sel]
        new_r = make_result_from_db(chosen_ex)
        st.session_state.input_text = chosen_ex["text"]
        st.session_state.results = [new_r]
        st.session_state.history.append((chosen_ex["text"], [new_r]))
        st.session_state.last_sel = sel

    text_input = st.text_area(
        "", value=st.session_state.input_text, height=150,
        placeholder="Бір немесе бірнеше қазақша сөйлем енгізіңіз…",
        label_visibility="collapsed",
        key="textarea_input",
    )
    st.session_state.input_text = text_input

    run_clicked = st.button("⚡ Талдау", use_container_width=True)

    st.markdown("""
    <div class="leg">
      <div class="leg-item"><div class="lsq" style="background:#3DDC84;"></div><span style="color:#3DDC84">CAUSE</span></div>
      <div class="leg-item"><div class="lsq" style="background:#F7794F;"></div><span style="color:#F7794F">EFFECT</span></div>
      <div class="leg-item"><div class="lsq" style="background:#F7CF4F;"></div><span style="color:#F7CF4F">MARKER</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divl'></div>", unsafe_allow_html=True)
    st.markdown("#### 📂 Batch upload (txt / csv)")
    uploaded = st.file_uploader("Бір жолға бір сөйлем", type=["txt","csv"], label_visibility="collapsed")

# ─────────────────── RIGHT ──────────────────────────────────────
with right:
    st.markdown("#### 📋 Нәтижесі")

    if run_clicked and text_input.strip():
        txt = text_input.strip()
        db_hit = next((e for e in DB_EXAMPLES if e["text"].strip() == txt), None)

        if db_hit and demo_mode:
            results = [make_result_from_db(db_hit)]
        elif demo_mode:
            results = demo_annotate_text(txt)
        else:
            with st.spinner("Сөйлемдер талданып жатыр…"):
                try:
                    results = analyze_text(txt)
                except Exception as e:
                    results = []
                    st.error(f"Pipeline қатесі: {e}")
            if not results:
                st.warning("⚠️ Себеп-салдарлы сөйлем табылмады.")

        st.session_state.results = results
        if not st.session_state.history or st.session_state.history[-1][0] != txt:
            st.session_state.history.append((txt, results))

    results = st.session_state.results

    if not results:
        st.markdown("""
        <div style="text-align:center;padding:72px 0;color:#6B7590;">
          <div style="font-size:48px;margin-bottom:14px">⚡</div>
          <div style="font-size:15px;font-weight:600;">Мәтін енгізіп, Талдау батырмасын басыңыз</div>
          <div style="font-size:12px;margin-top:8px;">Demo mode-та да маркер бойынша аннотация жасалады</div>
        </div>""", unsafe_allow_html=True)
    else:
        if len(results) > 1:
            st.info(f"🔍 {len(results)} сөйлем талданды.")

        full_text = st.session_state.input_text.strip()
        if full_text and results:
            st.markdown('<div class="sec">Full text annotation</div>', unsafe_allow_html=True)
            n = len(full_text)
            prio = [None] * n

            def mark_in_full(substr, css, lbl, p):
                if not substr:
                    return
                tl = full_text.lower()
                sl = substr.lower().strip()
                idx = 0
                while True:
                    pos = tl.find(sl, idx)
                    if pos == -1:
                        break
                    for i in range(pos, pos + len(sl)):
                        if prio[i] is None or prio[i][0] < p:
                            prio[i] = (p, css, lbl)
                    idx = pos + 1

            for r in results:
                mark_in_full(r.get("cause"), "sp-cause", "CAUSE", 2)
                mark_in_full(r.get("effect"), "sp-effect", "EFFECT", 1)
                for m in (r.get("markers") or []):
                    mark_in_full(m, "sp-marker", "MRK", 3)

            html = ""
            i = 0
            while i < n:
                cell = prio[i]
                if cell is None:
                    j = i
                    while j < n and prio[j] is None:
                        j += 1
                    html += full_text[i:j].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    i = j
                else:
                    _, css, lbl = cell
                    j = i
                    while j < n and prio[j] is not None and prio[j][1] == css:
                        j += 1
                    chunk = full_text[i:j].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    html += f'<span class="{css}">{chunk}<span class="sp-lbl">{lbl}</span></span>'
                    i = j

            html = html.replace("\n", "<br>")
            st.markdown(f'<div class="anno" style="line-height:2.8;">{html}</div>', unsafe_allow_html=True)
            st.markdown("<div class='divl'></div>", unsafe_allow_html=True)

        st.markdown('<div class="sec">Per-sentence breakdown</div>', unsafe_allow_html=True)

        for idx, r in enumerate(results):
            with st.expander(
                f"{'✅' if r.get('_from_db') else '🔬'} Sentence {idx+1}  —  {r['text'][:60]}{'…' if len(r['text'])>60 else ''}",
                expanded=(idx == 0),
            ):
                if r.get("_demo_blank"):
                    st.warning("⚠️ Маркер табылмады немесе demo mode бұл сөйлемді бөле алмады.")
                    plain = r["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                    st.markdown(f'<div class="anno" style="color:var(--muted);">{plain}</div>', unsafe_allow_html=True)
                else:
                    render_result_card(r, idx)

# ── Batch section ─────────────────────────────────────────────────
if uploaded:
    st.markdown("---")
    st.markdown("### 📊 Batch results")

    content = uploaded.read().decode("utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    st.info(f"**{len(lines)}** сөйлем жүктелді.")

    batch_results = []
    bar = st.progress(0, "Талданып жатыр…")
    for i, line in enumerate(lines):
        db_hit = next((e for e in DB_EXAMPLES if e["text"].strip() == line), None)
        if db_hit:
            batch_results.append(make_result_from_db(db_hit))
        elif demo_mode:
            batch_results.extend(demo_annotate_text(line))
        else:
            try:
                batch_results.extend(analyze_text(line))
            except Exception as e:
                batch_results.append({
                    "text": line, "cause": None, "effect": None,
                    "markers": [], "tv_form": "ERROR", "tv_confidence": None,
                    "semantic_type": "ERROR", "semantic_confidence": None,
                    "semantic_all": {},
                    "model_group": "ERROR", "model_group_confidence": None,
                    "timestamp": datetime.now().strftime("%H:%M:%S"), "error": str(e),
                })
        bar.progress((i + 1) / len(lines), f"Сөйлем {i+1} / {len(lines)}")
    bar.empty()

    df_b = pd.DataFrame([
        {"text": r["text"][:60] + ("…" if len(r["text"]) > 60 else ""),
         **{row["Field"]: row["Value"] for row in to_table_rows(r)}}
        for r in batch_results
    ])

    if not df_b.empty and "model_group" in df_b.columns:
        mc = df_b["model_group"].value_counts()
        cols = st.columns(4)
        cols[0].metric("Total", len(df_b))
        cols[1].metric("SYNTHETIC", mc.get("SYNTHETIC", 0))
        cols[2].metric("ANALYTIC", mc.get("ANALYTIC", 0))
        cols[3].metric("ANALYTICO-SYNTHETIC", mc.get("ANALYTICO_SYNTHETIC", 0))

    st.dataframe(df_b, width="stretch", height=360)

    b1, b2 = st.columns(2)
    with b1:
        st.download_button("⬇ Batch CSV",
                           df_b.to_csv(index=False, encoding="utf-8-sig"),
                           "batch.csv", "text/csv", use_container_width=True)
    with b2:
        st.download_button("⬇ Batch JSON",
                           json.dumps([clean_json(r) for r in batch_results], ensure_ascii=False, indent=2),
                           "batch.json", "application/json", use_container_width=True)
