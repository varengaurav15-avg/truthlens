"""
Multimodal Misinformation & Deepfake Detector -- Streamlit frontend v2
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import httpx
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Misinformation Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    API_BASE = st.text_input("API base URL", value="http://127.0.0.1:8000")
    TIMEOUT = st.slider("Request timeout (s)", 30, 180, 90)
    MAX_CLAIMS = st.slider("Max claims to analyse", 3, 25, 10)
    st.markdown("---")

    if st.button("Check API health"):
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=5)
            if r.status_code == 200:
                h = r.json()
                st.success("API is online")
                for mod, status in h.get("models", {}).items():
                    icon = "OK" if status == "loaded" else "WARN"
                    st.write(f"[{icon}] {mod}: {status}")
            else:
                st.error(f"API returned {r.status_code}")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

    st.markdown("---")
    st.markdown(
        "**About**\n\n"
        "Analyzes text claims against a Knowledge Graph "
        "and images with a CNN deepfake detector + Grad-CAM.\n\n"
        "Scores combine into a fused credibility verdict."
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VERDICT_COLORS = {
    "CREDIBLE":    "#27AE60",
    "SUSPICIOUS":  "#F39C12",
    "LIKELY_FAKE": "#E74C3C",
}
VERDICT_ICONS = {
    "CREDIBLE":    "CREDIBLE",
    "SUSPICIOUS":  "SUSPICIOUS",
    "LIKELY_FAKE": "LIKELY FAKE",
}


def verdict_badge(verdict: str, score: float) -> None:
    color = VERDICT_COLORS.get(verdict, "#888")
    label = VERDICT_ICONS.get(verdict, verdict)
    st.markdown(
        f"""
        <div style="background:{color}22;border:2px solid {color};
                    border-radius:12px;padding:16px 24px;text-align:center;
                    margin-bottom:16px;">
            <div style="font-size:1.5em;font-weight:700;color:{color};">{label}</div>
            <div style="font-size:2em;font-weight:800;color:{color};">{score*100:.0f}%</div>
            <div style="color:#888;font-size:0.85em;">credibility score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cred_bar(score: float, label: str = "Credibility") -> None:
    pct = int(score * 100)
    color = "#27AE60" if pct >= 70 else "#F39C12" if pct >= 40 else "#E74C3C"
    st.markdown(
        f"""
        <div style="margin:4px 0 12px 0;">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.85em;color:#555;margin-bottom:2px;">
            <span>{label}</span><span>{pct}%</span>
          </div>
          <div style="background:#eee;border-radius:6px;height:12px;">
            <div style="background:{color};width:{pct}%;height:12px;
                        border-radius:6px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_bars(signals: dict, weights: dict) -> None:
    st.markdown("**Signal breakdown**")
    mapping = [
        ("text_credibility",   "Text"),
        ("visual_credibility", "Image"),
        ("source_reputation",  "Source"),
    ]
    for key, label in mapping:
        val = signals.get(key, 0.0)
        mod = key.split("_")[0]
        w = weights.get(mod, None)
        suffix = f" (weight {w:.0%})" if w is not None else ""
        cred_bar(val, f"{label}{suffix}")


def claims_table(claims: list) -> None:
    if not claims:
        st.info("No claims extracted.")
        return

    label_colors = {
        "SUPPORTED":      "#27AE60",
        "REFUTED":        "#E74C3C",
        "NOT_ENOUGH_INFO": "#7F8C8D",
    }
    label_icons = {
        "SUPPORTED":       "[+]",
        "REFUTED":         "[-]",
        "NOT_ENOUGH_INFO": "[?]",
    }

    for i, c in enumerate(claims, 1):
        v = c.get("verification", {})
        lbl = v.get("label", "NOT_ENOUGH_INFO")
        score = v.get("score", 0.5)
        reason = v.get("reason", "")
        best_fact = v.get("best_fact")
        color = label_colors.get(lbl, "#888")
        icon = label_icons.get(lbl, "[?]")
        entities = c.get("entities", [])
        ent_str = ", ".join(
            f"{e['text']} ({e['label']})" for e in entities
        ) if entities else "none detected"

        preview = c["claim"][:90] + ("..." if len(c["claim"]) > 90 else "")
        with st.expander(f"{icon} Claim {i}: {preview}", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(
                    f"<span style='color:{color};font-weight:700;'>{icon} {lbl}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Confidence: {score:.2f}")
                st.caption(f"Reason: {reason}")
            with col2:
                st.markdown(f"**Entities:** {ent_str}")
                if best_fact:
                    st.markdown(f"**Best KG match:** _{best_fact}_")


def show_gradcam(b64: Optional[str], orig_img: Optional[Image.Image] = None) -> None:
    if not b64:
        st.info("Grad-CAM not available for this image.")
        return
    heatmap_bytes = base64.b64decode(b64)
    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

    if orig_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original image**")
            st.image(orig_img, use_container_width=True)
        with col2:
            st.markdown("**Grad-CAM saliency** (hot = suspicious regions)")
            st.image(heatmap_img, use_container_width=True)
    else:
        st.image(heatmap_img, caption="Grad-CAM saliency", use_container_width=True)


def api_error(resp: httpx.Response) -> None:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    st.error(f"API error {resp.status_code}: {detail}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Multimodal Misinformation & Deepfake Detector")
st.caption(
    "Paste an article or upload an image -- or both -- to get a credibility verdict "
    "backed by Knowledge Graph fact-checking and CNN deepfake analysis."
)
st.divider()

tabs = st.tabs(["Text Analysis", "Image Analysis", "Multimodal Fusion"])

# ===========================================================================
# TAB 1 -- TEXT
# ===========================================================================
with tabs[0]:
    st.subheader("Text Credibility Analysis")
    text = st.text_area(
        "Paste article text here",
        height=220,
        placeholder="Enter a news article, social media post, or any text claim...",
    )

    if st.button("Analyze Text", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running NER, claim extraction, and KG verification..."):
                try:
                    r = httpx.post(
                        f"{API_BASE}/text/analyze",
                        json={"text": text, "max_claims": MAX_CLAIMS},
                        timeout=TIMEOUT,
                    )
                except httpx.ConnectError:
                    st.error("Cannot connect to API. Is the server running?")
                    st.stop()

            if r.status_code != 200:
                api_error(r)
            else:
                d = r.json()
                score = d["credibility_score"]
                verdict = d.get("verdict", "SUSPICIOUS")

                verdict_badge(verdict, score)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Credibility", f"{score*100:.0f}%")
                c2.metric("Supported", d.get("supported_count", 0))
                c3.metric("Refuted", d.get("refuted_count", 0))
                c4.metric("Inconclusive", d.get("nei_count", 0))

                st.caption(d.get("explanation", ""))
                st.divider()

                st.markdown("### Claim-by-claim breakdown")
                claims_table(d.get("claims", []))

                with st.expander("Raw API response"):
                    st.json(d)

# ===========================================================================
# TAB 2 -- IMAGE
# ===========================================================================
with tabs[1]:
    st.subheader("Image Deepfake Analysis")
    img_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_only_uploader",
    )

    orig_img: Optional[Image.Image] = None
    if img_file:
        orig_img = Image.open(img_file)
        st.image(orig_img, caption="Uploaded image", width=400)

    if st.button("Analyze Image", type="primary", use_container_width=True):
        if img_file is None:
            st.warning("Please upload an image.")
        else:
            with st.spinner("Running face detection, CNN inference, Grad-CAM..."):
                try:
                    img_file.seek(0)
                    r = httpx.post(
                        f"{API_BASE}/vision/analyze_image",
                        files={"file": (img_file.name, img_file.read(), img_file.type)},
                        timeout=TIMEOUT,
                    )
                except httpx.ConnectError:
                    st.error("Cannot connect to API.")
                    st.stop()

            if r.status_code != 200:
                api_error(r)
            else:
                d = r.json()
                fake_prob = d["is_fake_prob"]
                cred = d["credibility_score"]
                verdict = (
                    "LIKELY_FAKE" if fake_prob > 0.65
                    else "SUSPICIOUS" if fake_prob > 0.35
                    else "CREDIBLE"
                )

                verdict_badge(verdict, cred)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Credibility", f"{cred*100:.0f}%")
                c2.metric("Fake Prob", f"{fake_prob*100:.0f}%")
                c3.metric("CNN Signal", f"{d.get('cnn_fake_prob', 0)*100:.0f}%")
                c4.metric("Faces detected", d.get("num_faces", 0))

                st.divider()
                cred_bar(cred, "Overall credibility")
                cred_bar(1 - d.get("cnn_fake_prob", 0), "CNN model")
                cred_bar(1 - d.get("frequency_fake_score", 0), "Frequency analysis")

                hfe = d.get("high_frequency_energy", 0)
                st.caption(
                    f"High-frequency energy: {hfe:.3f}  "
                    "(natural images approx 0.70-0.82; elevated values may indicate GAN artifacts)"
                )
                st.caption(d.get("explanation", ""))

                st.divider()
                st.markdown("### Grad-CAM saliency map")
                st.caption(
                    "Highlights regions the model focused on. "
                    "Hot areas drove the manipulation score."
                )
                show_gradcam(d.get("gradcam_heatmap_b64"), orig_img)

                with st.expander("Raw API response (without heatmap)"):
                    d_display = {k: v for k, v in d.items() if k != "gradcam_heatmap_b64"}
                    st.json(d_display)

# ===========================================================================
# TAB 3 -- MULTIMODAL FUSION
# ===========================================================================
with tabs[2]:
    st.subheader("Multimodal Fusion Analysis")
    st.caption("Combine text and image analysis for a stronger credibility signal.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        text2 = st.text_area(
            "Article text (optional)",
            height=180,
            placeholder="Paste article text...",
        )

    with col_right:
        img_file2 = st.file_uploader(
            "Image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="multimodal_uploader",
        )
        orig_img2: Optional[Image.Image] = None
        if img_file2:
            orig_img2 = Image.open(img_file2)
            st.image(orig_img2, width=200)

    rep = st.slider(
        "Source reputation prior",
        0.0, 1.0, 0.5, 0.05,
        help="0 = known bad actor, 1 = highly trusted outlet.",
    )

    if st.button("Run Fusion Analysis", type="primary", use_container_width=True):
        if not text2.strip() and img_file2 is None:
            st.warning("Provide at least text or an image.")
        else:
            with st.spinner("Running full multimodal pipeline..."):
                try:
                    data = {"text": text2, "source_reputation": str(rep)}
                    files = None
                    if img_file2:
                        img_file2.seek(0)
                        files = {"file": (img_file2.name, img_file2.read(), img_file2.type)}

                    r = httpx.post(
                        f"{API_BASE}/fusion/analyze",
                        data=data,
                        files=files,
                        timeout=TIMEOUT,
                    )
                except httpx.ConnectError:
                    st.error("Cannot connect to API.")
                    st.stop()

            if r.status_code != 200:
                api_error(r)
            else:
                d = r.json()
                score = d["credibility_score"]
                verdict = d.get("verdict", "SUSPICIOUS")
                risk = d.get("risk_level", "MEDIUM")
                confidence = d.get("confidence", 0.5)
                weights = d.get("fusion_weights", {})
                signals = d.get("signals", {})

                verdict_badge(verdict, score)

                c1, c2, c3 = st.columns(3)
                c1.metric("Risk level", risk)
                c2.metric("Confidence", f"{confidence*100:.0f}%")
                c3.metric("Credibility", f"{score*100:.0f}%")

                st.divider()

                left, right = st.columns(2)
                with left:
                    signal_bars(signals, weights)
                with right:
                    st.markdown("**Fusion weights applied**")
                    for mod, w in weights.items():
                        label_map = {"text": "Text", "visual": "Image", "source": "Source"}
                        st.progress(w, text=f"{label_map.get(mod, mod)}: {w:.0%}")

                st.divider()

                # Text details
                ta = d.get("text_analysis", {})
                if ta.get("claims"):
                    with st.expander(
                        f"Text analysis -- {ta.get('explanation', '')}",
                        expanded=True,
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Supported", ta.get("supported_count", 0))
                        c2.metric("Refuted", ta.get("refuted_count", 0))
                        c3.metric("NEI", ta.get("nei_count", 0))
                        claims_table(ta.get("claims", []))

                # Vision details
                va = d.get("vision_analysis", {})
                if img_file2 is not None:
                    with st.expander(
                        f"Image analysis -- {va.get('explanation', '')}",
                        expanded=True,
                    ):
                        cv1, cv2, cv3 = st.columns(3)
                        cv1.metric("Fake prob", f"{va.get('is_fake_prob', 0)*100:.0f}%")
                        cv2.metric("Faces", va.get("num_faces", 0))
                        cv3.metric("Freq anomaly", f"{va.get('frequency_fake_score', 0)*100:.0f}%")
                        st.markdown("### Grad-CAM")
                        show_gradcam(va.get("gradcam_heatmap_b64"), orig_img2)

                with st.expander("Raw API response"):
                    d_display = dict(d)
                    if "vision_analysis" in d_display:
                        d_display["vision_analysis"] = {
                            k: v for k, v in d_display["vision_analysis"].items()
                            if k != "gradcam_heatmap_b64"
                        }
                    st.json(d_display)
