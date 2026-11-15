"""Intrusion Detection System — Streamlit app (NSL-KDD, 3 classification models).

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import ids_core as core

st.set_page_config(
    page_title="Intrusion Detection System", page_icon="🛡️", layout="wide"
)


@st.cache_resource(
    show_spinner="Training the models on NSL-KDD (first run only, about a minute)…"
)
def load_bundle():
    return core.get_bundle()


try:
    B = load_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

FEATS = B["feats"]
ENC = B["encoders"]
MODELS = list(B["models"])
TEST_RAW = B["test_raw"]
KEY = [
    "protocol_type",
    "service",
    "flag",
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "same_srv_rate",
    "logged_in",
]
OTHERS = [f for f in FEATS if f not in KEY]

# ---------- session state defaults ----------
for f in FEATS:
    st.session_state.setdefault(f"in_{f}", str(ENC[f].classes_[0]) if f in ENC else 0.0)
st.session_state.setdefault("actual", None)
st.session_state.setdefault("row_no", 0)
st.session_state.setdefault("loaded_row", None)


def clear_actual():
    st.session_state["actual"] = None
    st.session_state["loaded_row"] = None


def load_row(i):
    row = TEST_RAW.iloc[int(i)]
    for f in FEATS:
        st.session_state[f"in_{f}"] = str(row[f]) if f in ENC else float(row[f])
    st.session_state["actual"] = str(row["class"])
    st.session_state["loaded_row"] = int(i)
    st.session_state["row_no"] = int(i)


def load_random(label):
    idx = np.where(TEST_RAW["class"] == label)[0]
    load_row(int(np.random.choice(idx)))


def feature_input(f, col):
    key = f"in_{f}"
    if f in ENC:
        col.selectbox(
            f, [str(c) for c in ENC[f].classes_], key=key, on_change=clear_actual
        )
    else:
        col.number_input(f, key=key, format="%g", on_change=clear_actual)


# ---------- style + header ----------
CSS = """
<style>
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.8rem;max-width:1180px}
.hero{background:linear-gradient(120deg,#14584a 0%,#1f8a70 100%);border-radius:16px;padding:26px 30px;margin-bottom:18px;color:#fff}
.hero-t{font-size:30px;font-weight:700;line-height:1.2}
.hero-s{opacity:.9;margin-top:6px;font-size:15px}
.stTabs [role="tablist"]{gap:8px}
.stTabs [role="tab"]{background:#fff;border:1px solid #dbe4e0;border-radius:12px;padding:12px 24px !important;height:auto !important;min-height:0 !important;display:flex;align-items:center;justify-content:center}
.stTabs [role="tab"] p,.stTabs [role="tab"] div{font-size:16px !important;font-weight:600 !important;margin:0 !important;line-height:1.2 !important}
.stTabs [role="tab"][aria-selected="true"] p,.stTabs [role="tab"][aria-selected="true"] div{color:#fff !important}
.stTabs [aria-selected="true"]{background:#1f6f5c;color:#fff !important;border-color:#1f6f5c}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none !important}
.stTabs [role="tablist"]{border-bottom:0 !important;box-shadow:none !important}
.stTabs [role="tablist"] > div:not([role="tab"]):not(:has([role="tab"])){display:none !important}
.stTabs [role="tablist"] > div:empty,.stTabs [role="tablist"] ~ div:empty,.stTabs [role="tablist"] + div:empty{display:none !important;height:0 !important}
.stTabs :has(> [role="tablist"]){border-bottom:0 !important;box-shadow:none !important;background-image:none !important}
.stTabs [role="tablist"]::before,.stTabs [role="tablist"]::after,.stTabs :has(> [role="tablist"])::before,.stTabs :has(> [role="tablist"])::after{display:none !important;content:none !important}
.stTabs [role="tab"]{box-shadow:none !important}
.stTabs [role="tab"]::before,.stTabs [role="tab"]::after{display:none !important;content:none !important}
.stButton>button{border-radius:10px;border:1px solid #c9d6d0;font-weight:600;padding:.45rem 1rem}
.stButton>button:hover{border-color:#1f6f5c;color:#1f6f5c}
.stButton>button[kind="primary"]{background:#1f6f5c;color:#fff;border-color:#1f6f5c;padding:.7rem 1rem;font-size:16px}
.stButton>button[kind="primary"]:hover{background:#18594a;color:#fff}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px}
div[data-testid="stMetric"]{background:#fff;border:1px solid #dbe4e0;border-radius:14px;padding:14px 18px}
.sec{font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#5d6b65;margin:6px 0 8px}
.res{border-radius:16px;padding:22px 26px;margin:14px 0;border:2px solid}
.res.ok{background:#e8f6f0;border-color:#1f8a70}
.res.bad{background:#fdecea;border-color:#c0392b}
.res-k{font-size:13px;color:#5d6b65;font-weight:600}
.res-v{font-size:30px;font-weight:800;margin:2px 0}
.res.ok .res-v{color:#14584a}.res.bad .res-v{color:#a5281b}
.res-c{font-size:14px;color:#445}
.bar{height:10px;background:#0000001a;border-radius:6px;margin-top:10px;overflow:hidden}
.fill{height:100%;border-radius:6px}
.res.ok .fill{background:#1f8a70}.res.bad .fill{background:#c0392b}
.act{border-radius:12px;padding:15px 20px;margin:4px 0 8px;font-size:16px;font-weight:700;border-left:8px solid}
.act.ok{background:#cfeee0;border-left-color:#1f8a70;color:#0d4637}
.act.bad{background:#ffe9c2;border-left-color:#e08a00;color:#6b3f00}
.mini{border-radius:14px;padding:14px 16px;background:#fff;border:1px solid #dbe4e0;border-left:6px solid}
.mini.ok{border-left-color:#1f8a70}.mini.bad{border-left-color:#c0392b}
.mini-n{font-weight:700}.mini-p{font-size:20px;font-weight:800;margin:2px 0}
.mini.ok .mini-p{color:#14584a}.mini.bad .mini-p{color:#a5281b}
.mini-d{font-size:12.5px;color:#5d6b65}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def result_card(model, label, p_normal, p_anomaly):
    ok = label == "normal"
    conf = p_normal if ok else p_anomaly
    cls = "ok" if ok else "bad"
    verdict = "✅ NORMAL" if ok else "🚨 ANOMALY: possible attack"
    st.markdown(
        f'<div class="res {cls}"><div class="res-k">{model}</div>'
        f'<div class="res-v">{verdict}</div>'
        f'<div class="res-c">Confidence {conf:.1%}</div>'
        f'<div class="bar"><div class="fill" style="width:{conf * 100:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def actual_card(actual, ok):
    cls = "ok" if ok else "bad"
    msg = "✓ Correct prediction" if ok else "✗ The model got this one wrong"
    st.markdown(
        f'<div class="act {cls}">Actual label in the test set: {actual.upper()} &nbsp;—&nbsp; {msg}</div>',
        unsafe_allow_html=True,
    )


def mini_card(col, model, r):
    ok = r["Prediction"] == "normal"
    cls = "ok" if ok else "bad"
    word = "NORMAL" if ok else "ANOMALY"
    col.markdown(
        f'<div class="mini {cls}"><div class="mini-n">{model}</div>'
        f'<div class="mini-p">{word}</div>'
        f'<div class="mini-d">P(anomaly) {r["P(anomaly)"]:.1%} &nbsp;·&nbsp; '
        f"P(normal) {r['P(normal)']:.1%}</div></div>",
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="hero-t">🛡️ Intrusion Detection System</div>'
    '<div class="hero-s">Classify a network connection as normal or an attack · NSL-KDD · '
    "Decision Tree, Random Forest and Naive Bayes</div></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "Data mining application: **Intrusion Detection**. Technique: **Classification** "
        "(binary: normal / anomaly)."
    )
    st.markdown("### Best model")
    st.metric(B["best"], f"{B['results_df'].loc[B['best'], 'Accuracy']:.2%}")
    st.markdown("### How to use")
    st.write(
        "1. Load a real record, or type values.\n"
        "2. Press **Classify connection**.\n"
        "3. Compare with the actual label."
    )
    st.caption(
        "KDDTest+ contains attack types unseen in training, so accuracy is around 77–78 %."
    )

tab_predict, tab_compare, tab_validate, tab_data = st.tabs(
    ["Predict", "Model comparison", "Interface validation", "Dataset"]
)

# ---------- tab 1: predict ----------
with tab_predict:
    with st.container(border=True):
        st.markdown(
            '<div class="sec">1 · Load a real record (optional)</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1], vertical_alignment="bottom")
        c1.button("🟢 Random normal record", on_click=load_random, args=("normal",))
        c2.button("🔴 Random attack record", on_click=load_random, args=("anomaly",))
        c3.number_input(
            "Test row #", min_value=0, max_value=len(TEST_RAW) - 1, step=1, key="row_no"
        )
        c4.button(
            "Load this row", on_click=lambda: load_row(st.session_state["row_no"])
        )
        if st.session_state["loaded_row"] is not None:
            st.info(
                f"Test-set row {st.session_state['loaded_row']} is loaded "
                "(all 41 features are filled in)."
            )

    with st.container(border=True):
        st.markdown(
            '<div class="sec">2 · Connection features</div>', unsafe_allow_html=True
        )
        cols = st.columns(3)
        for n, f in enumerate(KEY):
            feature_input(f, cols[n % 3])
        with st.expander(f"All other features ({len(OTHERS)})"):
            cols = st.columns(3)
            for n, f in enumerate(OTHERS):
                feature_input(f, cols[n % 3])
        st.caption(
            "Features you do not change stay at 0 (or the first category), so loaded "
            "real records are more reliable than hand-typed ones."
        )

    m1, m2 = st.columns([1, 1], vertical_alignment="bottom")
    model_name = m1.selectbox(
        "Model for the main prediction", MODELS, index=MODELS.index(B["best"])
    )
    go = m2.button("Classify connection ▸", type="primary")

    if go:
        values = {f: st.session_state[f"in_{f}"] for f in FEATS}
        out = core.predict_all(B, core.encode_row(values, B))
        main = out[model_name]
        result_card(
            model_name, main["Prediction"], main["P(normal)"], main["P(anomaly)"]
        )

        actual = st.session_state["actual"]
        if actual:
            ok = actual == main["Prediction"]
            actual_card(actual, ok)
        else:
            st.caption("Manual input: no ground-truth label to compare with.")

        st.markdown(
            '<div class="sec">All three models on this connection</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for col, (name, r) in zip(cols, out.items()):
            mini_card(col, name, r)

# ---------- tab 2: comparison ----------
with tab_compare:
    st.subheader("Performance on the held-out KDDTest+ set")
    res = B["results_df"]
    mcols = st.columns(3)
    for col, name in zip(mcols, MODELS):
        col.metric(
            ("⭐ " if name == B["best"] else "") + name,
            f"{res.loc[name, 'Accuracy']:.2%}",
        )
    st.dataframe((res * 100).round(2).astype(str) + " %")
    st.bar_chart(res)
    st.markdown(
        f"**Best model:** {B['best']} ({res.loc[B['best'], 'Accuracy']:.2%} accuracy)"
    )

    st.subheader("Confusion matrices")
    classes = list(B["label_encoder"].classes_)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, MODELS):
        cm = B["cms"][name]
        ax.imshow(cm, cmap="BuGn")
        ax.set_title(name, fontweight="bold")
        ax.set_xticks(range(len(classes)), classes)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    fig.tight_layout()
    st.pyplot(fig)

    for name in MODELS:
        with st.expander(f"Classification report — {name}"):
            st.code(B["reports"][name])
    st.caption(
        "KDDTest+ deliberately contains attack types that never appear in the training "
        "data, which is why accuracy is around 77–78 % instead of 99 %."
    )

# ---------- tab 3: validation ----------
with tab_validate:
    st.subheader("Actual vs predicted on chosen test rows")
    v1, v2 = st.columns([2, 1])
    rows_text = v1.text_input(
        "Row numbers (comma separated)", "411, 0, 50, 100, 250, 500, 2000"
    )
    v_model = v2.selectbox(
        "Model", MODELS, index=MODELS.index(B["best"]), key="v_model"
    )
    try:
        idx = [
            int(x) for x in rows_text.replace(";", ",").split(",") if x.strip() != ""
        ]
        bad = [i for i in idx if i < 0 or i >= len(TEST_RAW)]
        if bad:
            st.warning(f"Row numbers must be between 0 and {len(TEST_RAW) - 1}: {bad}")
        elif idx:
            vdf = core.validate_rows(B, v_model, idx)
            st.dataframe(vdf, hide_index=True)
            hit = int((vdf["Match"] == "✅").sum())
            st.write(f"**{hit} of {len(vdf)} correct ({hit / len(vdf):.1%})**")
    except ValueError:
        st.warning("Please enter whole numbers separated by commas.")

# ---------- tab 4: dataset ----------
with tab_data:
    st.subheader("NSL-KDD overview")
    s = B["stats"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Train records", f"{s['Train records (raw)']:,}")
    d2.metric("Test records", f"{s['Test records (raw)']:,}")
    d3.metric(
        "Duplicates removed",
        f"{s['Duplicates removed (train)'] + s['Duplicates removed (test)']}",
    )
    d4.metric("Missing values", f"{s['Missing values']}")
    st.markdown("**Class distribution (after removing duplicates)**")
    st.bar_chart(B["class_dist"])
    st.markdown("**Sample of the test set (readable values)**")
    st.dataframe(TEST_RAW.head(50))
