import os
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLAUSE_MODEL_DIR = os.path.join(BASE_DIR, "kazbert_clause_model_final")


# =========================================================
# CHECK MODEL FILES
# =========================================================
def ensure_model_dir(model_dir: str):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Модель папкасы табылмады:\n{model_dir}\n\n"
            f"streamlit.py және pipelineFinal.py тұрған папканың ішінде "
            f"'kazbert_clause_model_final' деген папка болуы керек."
        )

    existing = os.listdir(model_dir)

    required_text_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    for f in required_text_files:
        if f not in existing:
            raise FileNotFoundError(
                f"{model_dir} ішінде міндетті файл жоқ: {f}"
            )

    has_weights = ("model.safetensors" in existing) or ("pytorch_model.bin" in existing)
    if not has_weights:
        raise FileNotFoundError(
            f"{model_dir} ішінде model.safetensors немесе pytorch_model.bin жоқ."
        )


ensure_model_dir(CLAUSE_MODEL_DIR)


# =========================================================
# LOAD MODEL
# =========================================================
clause_tokenizer = AutoTokenizer.from_pretrained(
    CLAUSE_MODEL_DIR,
    local_files_only=True
)

clause_model = AutoModelForTokenClassification.from_pretrained(
    CLAUSE_MODEL_DIR,
    local_files_only=True
)

clause_pipeline = pipeline(
    "token-classification",
    model=clause_model,
    tokenizer=clause_tokenizer,
    aggregation_strategy="simple"
)


# =========================================================
# TEXT HELPERS
# =========================================================
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        for line in s.splitlines():
            line = line.strip()
            if line:
                result.append(line)
    return result


def merge_entities(entities, original_text):
    merged = {}

    for ent in entities:
        label = ent["entity_group"]

        # Normalize labels like B-CAUSE / I-CAUSE -> CAUSE
        if "CAUSE" in label:
            label = "CAUSE"
        elif "EFFECT" in label:
            label = "EFFECT"

        s, e = ent["start"], ent["end"]
        orig_word = original_text[s:e]

        if label not in merged:
            merged[label] = {
                "text": orig_word,
                "start": s,
                "end": e,
                "score": float(ent["score"]),
            }
        else:
            new_start = min(merged[label]["start"], s)
            new_end = max(merged[label]["end"], e)
            merged[label]["text"] = original_text[new_start:new_end]
            merged[label]["start"] = new_start
            merged[label]["end"] = new_end
            merged[label]["score"] = (merged[label]["score"] + float(ent["score"])) / 2

    return merged


def extract_markers(sentence):
    known_markers = [
        "өйткені",
        "себебі",
        "сондықтан",
        "сол себепті",
        "сол үшін",
        "неге десең",
        "неге десеңіз",
        "болғандықтан",
        "болмағандықтан",
        "білгендіктен",
        "көргендіктен",
        "жүргізілгендіктен",
        "алғандықтан",
        "берілгендіктен",
        "туғандықтан",
        "үшін",
        "соң",
        "кейін",
    ]

    found = []
    low = sentence.lower()
    for marker in known_markers:
        if marker in low:
            found.append(marker)
    return found


def is_causal(sentence):
    markers = extract_markers(sentence)
    return len(markers) > 0


# =========================================================
# MAIN ANALYSIS
# =========================================================
def analyze_causal_sentence(sentence):
    clause_entities = clause_pipeline(sentence)
    clause_data = merge_entities(clause_entities, sentence)
    markers = extract_markers(sentence)

    cause_text = clause_data.get("CAUSE", {}).get("text")
    effect_text = clause_data.get("EFFECT", {}).get("text")

    # fallback heuristic if model returns incomplete output
    if not cause_text or not effect_text:
        for m in sorted(markers, key=len, reverse=True):
            pos = sentence.lower().find(m.lower())
            if pos != -1:
                left = sentence[:pos].strip(" ,")
                right = sentence[pos + len(m):].strip(" ,")

                # ANALYTIC: effect + marker + cause
                if m in ["өйткені", "себебі", "неге десең", "неге десеңіз"]:
                    effect_text = effect_text or left
                    cause_text = cause_text or right
                    model_group = "ANALYTIC"
                # ANALYTICO-SYNTHETIC
                elif m in ["соң", "кейін"] or "соң" in m or "кейін" in m:
                    cause_text = cause_text or sentence[:pos + len(m)].strip(" ,")
                    effect_text = effect_text or right
                    model_group = "ANALYTICO_SYNTHETIC"
                # SYNTHETIC
                else:
                    cause_text = cause_text or sentence[:pos + len(m)].strip(" ,")
                    effect_text = effect_text or right
                    model_group = "SYNTHETIC"
                break
        else:
            model_group = "—"
    else:
        # rough guess of model group by marker
        model_group = "—"
        for m in markers:
            if m in ["өйткені", "себебі", "неге десең", "неге десеңіз"]:
                model_group = "ANALYTIC"
                break
            elif m in ["соң", "кейін"] or "соң" in m or "кейін" in m:
                model_group = "ANALYTICO_SYNTHETIC"
                break
            else:
                model_group = "SYNTHETIC"

    return {
        "text": sentence,
        "cause": cause_text,
        "effect": effect_text,
        "markers": markers,
        "tv_form": "—",
        "tv_confidence": None,
        "semantic_type": "—",
        "semantic_confidence": None,
        "semantic_all": {},
        "model_group": model_group,
        "model_group_confidence": None,
    }


def analyze_text(text):
    sentences = split_sentences(text)
    results = []

    for sent in sentences:
        if is_causal(sent):
            result = analyze_causal_sentence(sent)
            if result["cause"] or result["effect"] or result["markers"]:
                results.append(result)

    return results