"""
app.py — StyloGuard Multi-Author Forensic Stylometry Backend
Implements Poisson log-likelihood + Bayesian softmax for N student profiles.

Setup:
    pip install flask
    python app.py

To load Student A–E automatically on startup, place their .txt files at:
    data/students/A/   (3+ .txt files)
    data/students/B/
    data/students/C/
    data/students/D/
    data/students/E/

You can also upload these through the "Train Model" tab in the UI.
"""

from flask import Flask, request, jsonify, send_from_directory
import re, math, os, json
from collections import Counter
from datetime import datetime

app = Flask(__name__, static_folder=".")

# ──────────────────────────────────────────────────────── CONSTANTS
FILLERS = [
    "a","all","also","an","and","any","are","as","at","be","been","but","by",
    "can","do","down","even","every","for","from","had","has","have","her","his",
    "if","in","into","is","it","its","may","more","must","my","no","not","now",
    "of","on","one","only","or","our","shall","should","so","some","such","than",
    "that","the","their","then","there","thing","this","to","up","upon","was",
    "were","what","when","which","who","will","with","would","your"
]

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data", "students")
ARCHIVE_FILE = os.path.join(BASE_DIR, "data", "archive.json")
DEFAULT_IDS  = ["A", "B", "C", "D", "E"]

profiles: dict = {}
archive:  list = []

# ── OUTSIDER / NULL MODEL ───────────────────────────────────────────────────
#
# Calibrated for MODERN non-literary English (academic, professional,
# personal-statement, essay, report writing) — NOT Victorian formal prose.
#
# This is the critical distinction: StyloGuard's training corpus is 5 Victorian
# authors. Their prose shares many traits (high "the", "was", "had", "his",
# "upon", "shall"). The Outsider model must represent "everything else" —
# specifically, modern writers who do NOT belong to that corpus.
#
# Key calibration decisions vs old Victorian-leaning rates:
#   "the"  : 38  (not 62) — modern formal prose uses far less "the"
#   "my"   : 10  (not 3.5)— modern writing is more first-person
#   "to"   : 42  (not 28) — infinitives dominate modern formal prose
#   "in"   : 26  (not 20) — prepositional phrases frequent in modern writing
#   "was"  : 4   (not 13.5)— much less past-tense narrative
#   "had"  : 3   (not 7.5) — much less pluperfect narrative
#   "shall": 0.1 (not 0.8) — archaic; nearly absent in modern English
#   "upon" : 0.1 (not 0.5) — archaic; nearly absent in modern English
#   "his"  : 3.5 (not 7.0) — modern prose is less gendered
#   "her"  : 2.0 (not 5.0) — same
#   "it"   : 9   (not 11)  — slightly lower without fictional referents
GLOBAL_ENGLISH_RATES = {
    "a":     20.0, "all":   4.0,  "also":  2.5,  "an":    4.0,
    "and":   28.0, "any":   2.0,  "are":   5.5,  "as":    7.0,
    "at":    5.0,  "be":    7.0,  "been":  3.0,  "but":   4.0,
    "by":    4.0,  "can":   4.0,  "do":    3.0,  "down":  1.0,
    "even":  1.5,  "every": 1.2,  "for":   8.0,  "from":  4.5,
    "had":   3.0,  "has":   2.5,  "have":  5.0,  "her":   2.0,
    "his":   3.5,  "if":    3.0,  "in":    26.0, "into":  1.5,
    "is":    10.0, "it":    9.0,  "its":   2.5,  "may":   2.0,
    "more":  3.0,  "must":  2.0,  "my":    10.0, "no":    2.5,
    "not":   5.0,  "now":   2.0,  "of":    32.0, "on":    5.5,
    "one":   2.5,  "only":  2.5,  "or":    3.0,  "our":   2.0,
    "shall": 0.1,  "should":2.5,  "so":    3.0,  "some":  2.5,
    "such":  1.5,  "than":  2.5,  "that":  11.0, "the":   38.0,
    "their": 4.0,  "then":  2.0,  "there": 3.0,  "thing": 1.0,
    "this":  6.0,  "to":    42.0, "up":    2.5,  "upon":  0.1,
    "was":   4.0,  "were":  3.0,  "what":  3.0,  "when":  3.0,
    "which": 3.5,  "who":   2.5,  "will":  5.0,  "with":  7.0,
    "would": 4.0,  "your":  2.5,  "was":   4.0,  "were":  3.0,  
    "what":  3.0,  "when":  3.0, "which": 3.5,  "who":   2.5,  
    "will":  5.0,  "with":  7.0, "would": 4.0,  "your":  2.5
}

# ──────────────────────────────────────────────────────── TEXT UTILS
def process_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[''`'\u2018\u2019]", "", text)
    text = re.sub(r"[^a-z]", " ", text)
    return [w for w in text.split() if w]

def calculate_rates(words: list, alpha: float = 1) -> tuple:
    total  = len(words)
    counts = Counter(words)
    denom  = total + alpha * len(FILLERS)
    return {f: ((counts.get(f, 0) + alpha) / denom) * 1000 for f in FILLERS}, total

def poisson_log_pmf(k: int, lam: float) -> float:
    """Matches R's dpois(k, lam, log=TRUE) exactly."""
    if lam <= 0: lam = 1e-10  # Prevent log(0)
    # Poisson Log PMF: x*log(λ) - λ - log(x!)
    return k * math.log(lam) - lam - math.lgamma(k + 1)


# ──────────────────────────────────────────────────────── STARTUP
def load_from_dir(student_id: str) -> dict | None:
    folder = os.path.join(DATA_DIR, student_id)
    if not os.path.isdir(folder):
        return None
    txts = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    if not txts:
        return None
    words, names = [], []
    for fn in txts:
        try:
            with open(os.path.join(folder, fn), encoding="utf-8", errors="ignore") as fh:
                words.extend(process_text(fh.read()))
            names.append(fn)
        except Exception:
            pass
    if len(words) < 50:
        return None
    rates, total = calculate_rates(words)
    return {"rates": rates, "total_words": total, "files": names,
            "is_default": True, "added_at": datetime.now().isoformat()}

def bootstrap():
    global archive
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE) as fh:
                archive = json.load(fh)
        except Exception:
            archive = []
    for sid in DEFAULT_IDS:
        r = load_from_dir(sid)
        if r:
            profiles[f"Student {sid}"] = r
    loaded = list(profiles.keys())
    print(f"  Profiles loaded: {loaded if loaded else 'none yet — upload via Train Model tab'}")

def save_archive():
    os.makedirs(os.path.dirname(ARCHIVE_FILE), exist_ok=True)
    with open(ARCHIVE_FILE, "w") as fh:
        json.dump(archive, fh, indent=2, default=str)



def analyze_all(text: str) -> tuple:
    """
    Two-stage Bayesian Authorship Attribution
    ==========================================

    STAGE 1 — Absolute fit test (LLR vs Outsider)
    ------------------------------------------------
    For each known student k, compute the log-likelihood ratio:
        LLR_k = log P(text | student_k) − log P(text | Outsider)

    • LLR_k > 0  → Student k explains the text BETTER than the Outsider model
    • LLR_k < 0  → Outsider explains it better → student k is a poor fit

    If ALL students have LLR_k < 0, the text does not resemble any known
    student more than it resembles generic modern English → VERDICT: Outsider.

    STAGE 2 — Relative ranking (softmax for display)
    -------------------------------------------------
    We still run softmax over ALL hypotheses (students + Outsider) so the UI
    can show a pie chart with meaningful percentages. The Stage-1 verdict
    overrides the softmax winner if Stage-1 signals Outsider.

    WHY THIS FIXES THE OLD BUG
    --------------------------
    The old code used a "dynamic baseline" that self-fit the test text — by
    construction it always scored near-perfectly, making the deviance threshold
    completely arbitrary.  Here, GLOBAL_ENGLISH_RATES is a FIXED model trained
    on modern non-literary English.  It genuinely competes as a hypothesis and
    wins when the text looks more like modern prose than like any specific
    Victorian student.
    """
    if not profiles:
        return None, "No trained profiles. Upload training files via the Train Model tab."

    words  = process_text(text)
    n      = len(words)
    counts = Counter(words)
    lm     = n / 1000

    # ── 1. Compute log-likelihoods for all hypotheses ────────────────────────
    log_likes: dict[str, float] = {}

    for name, prof in profiles.items():
        log_likes[name] = sum(
            poisson_log_pmf(counts.get(f, 0), prof["rates"].get(f, 0.1) * lm)
            for f in FILLERS
        )

    outsider_ll = sum(
        poisson_log_pmf(counts.get(f, 0), GLOBAL_ENGLISH_RATES.get(f, 0.1) * lm)
        for f in FILLERS
    )
    log_likes["Outsider"] = outsider_ll

    # ── 2. Stage-1: LLR test — does ANY student beat the Outsider? ───────────
    student_lls = {k: v for k, v in log_likes.items() if k != "Outsider"}
    llrs        = {k: v - outsider_ll for k, v in student_lls.items()}  # LLR_k

    best_student     = max(student_lls, key=student_lls.get)
    best_student_llr = llrs[best_student]

    # If the best student still scores below the Outsider, it's an Outsider text
    force_outsider = (best_student_llr < 0)

    # ── 3. Numerically stable tempered softmax ────────────────────────────
    # We use a scaling factor to prevent the "100% certainty" bug.
    # Dividing by a factor (e.g., 5.0 to 10.0) "softens" the distribution.
    # Alternatively, use math.sqrt(len(words)) for a dynamic approach.
    temperature = 8.5 

    mx       = max(log_likes.values())
    exp_vals = {k: math.exp((v - mx) / temperature) for k, v in log_likes.items()}
    total_e  = sum(exp_vals.values())
    probs    = {k: round(v / total_e * 100, 2) for k, v in exp_vals.items()}

    # ── 4. Build ranked list and derive verdict ───────────────────────────────
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    if force_outsider:
        # Stage-1 override: Outsider wins regardless of softmax ranking
        top, tp     = "Outsider", probs["Outsider"]
        is_outsider = True
    else:
        top, tp     = ranked[0]
        is_outsider = (top == "Outsider")

    # Confidence from the gap between first and second place in Stage-2 softmax
    gap = (tp - ranked[1][1]) if len(ranked) > 1 else tp

    if n < 80:
        conf = "Low (text too short for reliable attribution)"
    elif is_outsider:
        # Report the margin by which Outsider wins
        conf = "High" if abs(best_student_llr) > 20 else "Medium" if abs(best_student_llr) > 8 else "Low"
    else:
        conf = "High" if gap > 30 else "Medium" if gap > 12 else "Low"

    # ── 5. Per-filler breakdown for the LL table ─────────────────────────────
    breakdown = []
    for f in FILLERS:
        x   = counts.get(f, 0)
        per = {
            "Outsider": {
                "expected": round(GLOBAL_ENGLISH_RATES.get(f, 0.1) * lm, 3),
                "log_p":    round(poisson_log_pmf(x, GLOBAL_ENGLISH_RATES.get(f, 0.1) * lm), 4),
            }
        }
        for name, prof in profiles.items():
            la = prof["rates"].get(f, 0.1) * lm
            per[name] = {"expected": round(la, 3), "log_p": round(poisson_log_pmf(x, la), 4)}
        breakdown.append({"word": f, "count": x, "per_student": per})

    return {
        "probabilities":    probs,
        "log_likelihoods":  {k: round(v, 4) for k, v in log_likes.items()},
        "llrs":             {k: round(v, 4) for k, v in llrs.items()},
        "ranked":           ranked,
        "top_candidate":    top,
        "top_probability":  tp,
        "is_outsider":      is_outsider,
        "force_outsider":   force_outsider,
        "best_student_llr": round(best_student_llr, 3),
        "confidence":       conf,
        "word_count":       n,
        "filler_breakdown": breakdown,
    }, None


# ──────────────────────────────────────────────────────── ROUTES
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/api/profiles")
def api_profiles():
    return jsonify({
        name: {"total_words": p["total_words"], "files": p["files"],
               "is_default": p["is_default"], "added_at": p.get("added_at", "")}
        for name, p in profiles.items()
    })

@app.route("/api/default-slots")
def api_default_slots():
    slots = []
    for sid in DEFAULT_IDS:
        name = f"Student {sid}"
        p    = profiles.get(name)
        slots.append({
            "id": sid, "name": name,
            "trained":     bool(p),
            "files":       p["files"]       if p else [],
            "total_words": p["total_words"] if p else 0,
        })
    return jsonify(slots)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    text, filename = "", "Pasted text"
    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file attached."}), 400
        text, filename = f.read().decode("utf-8", errors="ignore"), f.filename
    else:
        data = request.get_json(force=True) or {}
        text = data.get("text", "")

    text = text.strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    wc = len(process_text(text))
    if wc < 20:
        return jsonify({"error": f"Text too short ({wc} words). Minimum 20 required."}), 400

    result, err = analyze_all(text)
    if err:
        return jsonify({"error": err}), 400

    archive.append({
        "type":          "scan",
        "timestamp":     datetime.now().isoformat(),
        "source":        filename,
        "word_count":    result["word_count"],
        "top_candidate": result["top_candidate"],
        "top_prob":      result["top_probability"],
        "probabilities": result["probabilities"],
        "confidence":    result["confidence"],
        "is_outsider":   result.get("is_outsider", False),
    })
    save_archive()
    result["source"] = filename
    return jsonify(result)

@app.route("/api/upload-default", methods=["POST"])
def api_upload_default():
    sid   = request.form.get("student", "").strip().upper()
    files = request.files.getlist("files")
    if sid not in DEFAULT_IDS:
        return jsonify({"error": f"Invalid student ID '{sid}'."}), 400
    if len(files) < 3:
        return jsonify({"error": "Upload at least 3 .txt files."}), 400

    words, names = [], []
    folder = os.path.join(DATA_DIR, sid)
    os.makedirs(folder, exist_ok=True)
    for f in files:
        raw = f.read().decode("utf-8", errors="ignore")
        words.extend(process_text(raw))
        names.append(f.filename)
        with open(os.path.join(folder, f.filename), "w", encoding="utf-8") as out:
            out.write(raw)

    if len(words) < 100:
        return jsonify({"error": f"Only {len(words)} words found. Need ≥ 100."}), 400

    rates, total = calculate_rates(words)
    name = f"Student {sid}"
    profiles[name] = {"rates": rates, "total_words": total, "files": names,
                      "is_default": True, "added_at": datetime.now().isoformat()}
    return jsonify({"success": True, "name": name, "total_words": total, "files": names})

@app.route("/api/train", methods=["POST"])
def api_train():
    name  = request.form.get("name", "").strip()
    files = request.files.getlist("files")
    if not name:
        return jsonify({"error": "Profile name is required."}), 400
    if name in profiles:
        return jsonify({"error": f"Profile '{name}' already exists."}), 400
    if len(files) < 3:
        return jsonify({"error": "Upload at least 3 .txt files."}), 400

    words, names = [], []
    for f in files:
        if not f.filename.lower().endswith(".txt"):
            return jsonify({"error": f"'{f.filename}' is not a .txt file."}), 400
        words.extend(process_text(f.read().decode("utf-8", errors="ignore")))
        names.append(f.filename)

    if len(words) < 100:
        return jsonify({"error": f"Only {len(words)} words total. Need ≥ 100."}), 400

    rates, total = calculate_rates(words)
    profiles[name] = {"rates": rates, "total_words": total, "files": names,
                      "is_default": False, "added_at": datetime.now().isoformat()}
    archive.append({"type": "student_added", "timestamp": datetime.now().isoformat(),
                    "name": name, "files": names, "total_words": total})
    save_archive()
    return jsonify({"success": True, "name": name, "total_words": total, "files": names})

@app.route("/api/archive")
def api_archive():
    return jsonify(archive)

@app.route("/api/debug", methods=["POST"])
def api_debug():
    """
    Diagnostic endpoint: returns raw log-likelihoods per model for a text
    so you can inspect the numbers without the softmax normalisation.
    """
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text."}), 400
    words  = process_text(text)
    n      = len(words)
    counts = Counter(words)
    lm     = n / 1000
    raw    = {}
    for name, prof in profiles.items():
        raw[name] = sum(
            poisson_log_pmf(counts.get(f, 0), prof["rates"].get(f, 0.1) * lm)
            for f in FILLERS
        )
    raw["Outsider"] = sum(
        poisson_log_pmf(counts.get(f, 0), GLOBAL_ENGLISH_RATES.get(f, 1.0) * lm)
        for f in FILLERS
    )
    return jsonify({"word_count": n, "raw_log_likelihoods": raw,
                    "winner": max(raw, key=raw.get)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global archive
    removed = [n for n, p in profiles.items() if not p["is_default"]]
    for n in removed:
        del profiles[n]
    archive = []
    save_archive()
    return jsonify({"success": True, "removed": removed})

@app.route("/api/rename-profile", methods=["POST"])
def api_rename_profile():
    data = request.get_json(force=True) or {}
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    
    if not old_name or not new_name:
        return jsonify({"error": "Both old_name and new_name are required."}), 400
    if old_name not in profiles:
        return jsonify({"error": f"Profile '{old_name}' does not exist."}), 400
    if new_name in profiles:
        return jsonify({"error": f"Profile '{new_name}' already exists."}), 400
    if profiles[old_name].get("is_default"):
        return jsonify({"error": "Cannot rename default profiles (A-E)."}), 400
    
    # Rename the profile
    profiles[new_name] = profiles[old_name]
    del profiles[old_name]
    
    # Update archive entries
    for entry in archive:
        if entry.get("name") == old_name:
            entry["name"] = new_name
    
    save_archive()
    return jsonify({"success": True, "old_name": old_name, "new_name": new_name})


# ──────────────────────────────────────────────────────── MAIN
if __name__ == "__main__":
    print("=" * 60)
    print("  StyloGuard — Multi-Author Forensic Stylometry Backend")
    bootstrap()
    print("  Running at → http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)