import base64
import io
import time

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

MODEL_NAME = "ResNet18-UNet"
IMAGE_SIZE = "256 × 256"
THRESHOLD = 0.60

TEST_DICE = 0.8611
TEST_IOU = 0.8321


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NeuroSeg",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
    ========================= */

    .stApp {
        background-color: #0b0f14;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p {
        color: #94a3b8;
    }


    /* =========================
       SIDEBAR
    ========================= */

    section[data-testid="stSidebar"] {
        background-color: #080c11;
        border-right: 1px solid #1e293b;
    }

    .sidebar-brand {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 1.3rem;
    }

    .sidebar-section {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    .sidebar-value {
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }


    /* =========================
       HEADER
    ========================= */

    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
    }

    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .header-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    .status {
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        border: 1px solid #14532d;
        background: #052e16;
        color: #86efac;
        font-size: 0.78rem;
        font-weight: 600;
    }


    /* =========================
       PANELS
    ========================= */

    .panel {
        background: #111820;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        color: #e2e8f0;
        font-size: 0.92rem;
        font-weight: 650;
        margin-bottom: 0.7rem;
    }

    .panel-caption {
        color: #64748b;
        font-size: 0.78rem;
    }


    /* =========================
       METRICS
    ========================= */

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.6rem;
    }

    .metric {
        background: #0c1219;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.75rem;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }


    /* =========================
       RESULT
    ========================= */

    .result-positive {
        background: #2b1115;
        border: 1px solid #7f1d1d;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #fca5a5;
        font-weight: 650;
    }

    .result-negative {
        background: #052e16;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #86efac;
        font-weight: 650;
    }


    /* =========================
       UPLOADER
    ========================= */

    [data-testid="stFileUploader"] {
        background: #0c1219;
        border: 1px dashed #334155;
        border-radius: 8px;
        padding: 0.5rem;
    }


    /* =========================
       BUTTON
    ========================= */

    .stButton > button {
        height: 42px;
        border-radius: 7px;
        font-weight: 650;
    }


    /* =========================
       IMAGE
    ========================= */

    [data-testid="stImage"] {
        border-radius: 7px;
        overflow: hidden;
    }


    /* =========================
       FOOTER
    ========================= */

    .footer {
        border-top: 1px solid #1e293b;
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        color: #475569;
        font-size: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">🧠 NeuroSeg</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        "Brain MRI segmentation system"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sidebar-value">Architecture&nbsp;&nbsp; '
        f'<b>{MODEL_NAME}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-value">Framework&nbsp;&nbsp; '
        "<b>PyTorch</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sidebar-value">Input&nbsp;&nbsp; '
        f"<b>{IMAGE_SIZE}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-value">Task&nbsp;&nbsp; '
        "<b>Binary Segmentation</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Deployment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-value">API&nbsp;&nbsp; '
        "<b>FastAPI</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-value">Frontend&nbsp;&nbsp; '
        "<b>Streamlit</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-value">Device&nbsp;&nbsp; '
        "<b>CUDA / RTX 4060</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.metric(
        "Test Dice",
        f"{TEST_DICE:.4f}",
    )

    st.metric(
        "Test IoU",
        f"{TEST_IOU:.4f}",
    )

    st.caption(
        "Held-out test set performance"
    )

    st.markdown(
        '<div class="sidebar-section">Inference</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sidebar-value">Threshold&nbsp;&nbsp; '
        f"<b>{THRESHOLD:.2f}</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.caption(
        "⚠ Research / educational system. "
        "Not a medical diagnostic tool."
    )


# ============================================================
# HEADER
# ============================================================

header_col, status_col = st.columns([5, 1])

with header_col:
    st.title("🧠 Brain Tumor Segmentation")
    st.caption("MRI segmentation inference dashboard")

with status_col:
    st.success("● API READY")

st.divider()


# ============================================================
# TOP ROW
# ============================================================

left, right = st.columns(
    [1.6, 1],
    gap="medium",
)


# ============================================================
# UPLOAD
# ============================================================

with left:

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                MRI Input
            </div>
            <div class="panel-caption">
                Upload a TIFF, PNG, or JPEG brain MRI scan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "MRI image",
        type=[
            "tif",
            "tiff",
            "png",
            "jpg",
            "jpeg",
        ],
        label_visibility="collapsed",
    )


# ============================================================
# MODEL PERFORMANCE PANEL
# ============================================================

with right:
    st.subheader("Model Performance")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="Test Dice",
            value="0.8611",
        )

    with metric2:
        st.metric(
            label="Test IoU",
            value="0.8321",
        )

    with metric3:
        st.metric(
            label="Threshold",
            value="0.60",
        )

    st.caption(
        "Evaluation metrics from the held-out test set."
    )

# ============================================================
# PREVIEW
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    original_image = Image.open(
        io.BytesIO(image_bytes)
    )

    st.markdown(
        """
        <div class="panel-title">
            Scan Preview
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_left, preview_right = st.columns(
        [4, 1],
        vertical_alignment="bottom",
    )

    with preview_left:

        st.caption(
            f"{uploaded_file.name} · "
            f"{original_image.width} × "
            f"{original_image.height}"
        )

    with preview_right:

        analyze = st.button(
            "Analyze Scan",
            type="primary",
            width="stretch",
        )

    if analyze:

        # ====================================================
        # API REQUEST
        # ====================================================

        with st.spinner(
            "Running ResNet18-UNet inference..."
        ):

            start = time.perf_counter()

            try:

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type,
                        )
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.ConnectionError:

                st.error(
                    "FastAPI is not reachable. "
                    "Start the API server first."
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "Inference request timed out."
                )

                st.stop()

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"API request failed: {exc}"
                )

                st.stop()

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )

                st.stop()

        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        if result["tumor_detected"]:

            st.markdown(
                """
                <div class="result-positive">
                    🔴 TUMOR DETECTED
                    <span style="float:right;">
                    Segmentation available
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="result-negative">
                    🟢 NO TUMOR DETECTED
                    <span style="float:right;">
                    No positive region predicted
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # ====================================================
        # ALL PREDICTION METRICS
        # ====================================================

        st.markdown(
            """
            <div class="panel-title"
                 style="margin-top:1rem;">
                Prediction Metrics
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:

            st.metric(
                "Tumor Area",
                f'{result["tumor_percentage"]:.2f}%',
            )

        with m2:

            st.metric(
                "Tumor Pixels",
                f'{result["tumor_area_pixels"]:,}',
            )

        with m3:

            st.metric(
                "Inference",
                f'{result["inference_time_ms"]:.1f} ms',
            )

        with m4:

            st.metric(
                "Threshold",
                f'{result["threshold"]:.2f}',
            )


        # ====================================================
        # VISUALIZATION
        # ====================================================

        overlay_bytes = base64.b64decode(
            result["overlay"]
        )

        overlay_image = Image.open(
            io.BytesIO(overlay_bytes)
        )

        st.markdown(
            """
            <div class="panel-title"
                 style="margin-top:1.2rem;">
                Segmentation
            </div>
            """,
            unsafe_allow_html=True,
        )

        image_col1, image_col2 = st.columns(
            2,
            gap="medium",
        )

        with image_col1:

            st.caption("ORIGINAL MRI")

            st.image(
                original_image,
                width="stretch",
            )

        with image_col2:

            st.caption("PREDICTED OVERLAY")

            st.image(
                overlay_image,
                width="stretch",
            )


        # ====================================================
        # DOWNLOAD
        # ====================================================

        download_col1, download_col2 = st.columns(
            [5, 1]
        )

        with download_col1:

            st.caption(
                "Red region indicates pixels classified "
                "as tumor at the 0.60 inference threshold."
            )

        with download_col2:

            st.download_button(
                "Download",
                data=overlay_bytes,
                file_name="tumor_segmentation.png",
                mime="image/png",
                width="stretch",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        NeuroSeg · ResNet18-UNet · PyTorch · FastAPI · Streamlit
        <br>
        Research and educational use only — not for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)