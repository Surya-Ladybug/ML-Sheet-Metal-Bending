"""
STREAMLIT UI: ML ASSISTED SHEET METAL BENDING WITH SPRINGBACK
==============================================================

Production-style predictive digital twin
Text input only - No sliders, no interactive widgets
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image

# ============================================================================
# SETUP PATHS AND IMPORTS
# ============================================================================

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Add to Python path
sys.path.insert(0, os.path.join(project_root, 'models'))
sys.path.insert(0, os.path.join(project_root, 'utils'))
sys.path.insert(0, os.path.join(project_root, 'toolpath'))

# Try importing modules
try:
    from springback_model import SpringbackSurrogateModel
    from inverse_model import InverseRegressionModel
    from plotting import BendingVisualizer
    from toolpath_generator import ToolpathGenerator
    imports_ok = True
    import_error = None
except Exception as e:
    imports_ok = False
    import_error = str(e)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="ML Assisted Sheet Metal Bending",
    layout="wide",
    page_icon="🔧"
)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'computed' not in st.session_state:
    st.session_state.computed = False

# ============================================================================
# CHECK IMPORTS
# ============================================================================
if not imports_ok:
    st.error(f"❌ Failed to import modules: {import_error}")
    st.warning("""
    **Setup Instructions:**
    
    1. Make sure you're in the `project_root/ui/` directory
    2. Check all modules exist
    3. Install dependencies: `pip install -r ../requirements.txt`
    
    **Quick Fix:**
    ```bash
    cd ..
    python3 check_setup.py
    ```
    """)
    st.stop()

# ============================================================================
# LOAD TRAINED MODELS (CACHED)
# ============================================================================
@st.cache_resource
def load_models():
    """Load pre-trained ML models"""
    springback_pkl = os.path.join(project_root, 'models', 'springback_model.pkl')
    inverse_pkl = os.path.join(project_root, 'models', 'inverse_model.pkl')
    
    # Check if models exist
    if not os.path.exists(springback_pkl):
        raise FileNotFoundError(
            f"Springback model not found. Please train models first."
        )
    
    if not os.path.exists(inverse_pkl):
        raise FileNotFoundError(
            f"Inverse model not found. Please train models first."
        )
    
    # Load models
    model1 = SpringbackSurrogateModel()
    model1.load(springback_pkl)
    
    model2 = InverseRegressionModel()
    model2.load(inverse_pkl)
    
    return model1, model2


# Initialize utilities
viz = BendingVisualizer()
toolpath_gen = ToolpathGenerator(angular_step=1.0)

# Try loading models
try:
    model_springback, model_inverse = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")
    models_loaded = False
    st.stop()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #379462;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #7f8c8d;
    margin-bottom: 2rem;
}
.info-box {
    background-color: #1c4a31;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #f7f9fa;
}
</style>

<div class="main-header">🔧 ML Assisted Sheet Metal Bending with Springback</div>
<div class="sub-header">
Predictive Digital Twin — Replaces LS-DYNA Runtime with Tree-Based Regression
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SHOW INFO SECTIONS ONLY BEFORE COMPUTATION
# ============================================================================

if not st.session_state.computed:

    col_m, col_rx = st.columns(2)
    
    # Material Properties
    with col_m:
        st.markdown("### 📋 Material Properties")
        st.markdown("""
        <div class="info-box">
        <b>Material:</b> Stainless Steel<br>
        <b>Status:</b> ✓ Fixed
        </div>
        """, unsafe_allow_html=True)
    
    # RX Constraints
    with col_rx:
        st.markdown("### ℹ️ RX Constraints")
        st.markdown("""
        <div class="info-box">
        <b>RX Range</b><br>
        • Upper limit: <b>-10°</b><br>
        • Lower limit: <b>-140°</b><br>
        Physical tool rotation limits based on FEM simulation range.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # RX Direction Image
    st.markdown("### 🔄 RX Direction Explanation")
    
    try:
        # Load image from assets
        image_path = os.path.join(current_dir, 'assets', 'rx_direction.png')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, caption="RX Direction: Negative values move the tool to the right", use_column_width=True)
        else:
            st.warning("RX direction image not found. Please ensure rx_direction.png is in ui/assets/")
    except Exception as e:
        st.warning(f"Could not load RX direction image: {e}")
    
    st.markdown("---")
    
    # About 
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown("### 🎓 Forward Model")
        st.markdown("""
        - Predicts stable part angle from FEM-validated data  
        - Tree-based regression
        """)

    with col_a2:
        st.markdown("### 🔄 Inverse Model")
        st.markdown("""
        - Predicts required tool parameters  
        - Multi-output regression
        """)

    with col_a3:
        st.markdown("### ⚡ Why ML Surrogate?")
        st.markdown("""
        - Replaces FEM runtime (hours → seconds)  
        - Learns complex material response  
        - No assumed deformation model
        """)

    st.markdown("---")
    

# ============================================================================
# INPUT SECTION (Always visible)
# ============================================================================

st.header("🎯 Input Parameters")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### Target Part Angle")
    target_angle = st.number_input(
        "Desired stable part angle (degrees)",
        min_value=10.0,
        max_value=100.0,
        value=45.0,
        step=1.0,
        format="%.1f",
        help="The final stable part angle you want to achieve after tool release",
        key="target_angle_input"
    )

with col2:
    st.markdown("### Tool Parameter (RX)")
    rx_value = st.number_input(
        "RX value (degrees, negative)",
        min_value=-140.0,
        max_value=-10.0,
        value=-20.0,
        step=1.0,
        format="%.1f",
        help="Tool rotation constraint (must be between -140° and -10°)",
        key="rx_input"
    )

with col3:
    st.markdown("<div style='height: 89px;'></div>", unsafe_allow_html=True)
    compute_btn = st.button(
        "🚀 Compute Bending Solution",
        type="primary",
        use_container_width=True
    )


# ============================================================================
# COMPUTATION
# ============================================================================

if compute_btn:
    st.session_state.computed = True
    
    with st.spinner("Computing ML predictions..."):
        
        try:
            # MODEL 2: INVERSE PREDICTION
            predicted_ryr, predicted_radius = model_inverse.predict(
                target_angle, rx_value
            )
            
            # MODEL 1: FORWARD PREDICTION
            predicted_final, Ks, overbend_angle = model_springback.calculate_springback_factor(
                rx_value, predicted_ryr, target_angle
            )
            
            # Store in session state
            st.session_state.results = {
                'predicted_ryr': predicted_ryr,
                'predicted_radius': predicted_radius,
                'predicted_final': predicted_final,
                'Ks': Ks,
                'overbend_angle': overbend_angle,
                'target_angle': target_angle,
                'rx_value': rx_value
            }
            
        except Exception as e:
            st.error(f"❌ Computation Error: {e}")
            st.session_state.computed = False

# ============================================================================
# SHOW RESULTS (Only if computed)
# ============================================================================

if st.session_state.computed and 'results' in st.session_state:
    
    r = st.session_state.results
    
    st.markdown("---")
    st.header("📊 ML Prediction Results")
    
    # Metrics row
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric(
            "Calculated RYR",
            f"{r['predicted_ryr']:.3f}",
            help="Secondary rotation parameter from inverse model"
        )
        st.caption("**How Calculated:**")
        st.caption("🔹 Inverse Model: `(target, rx) → ryr`")
        st.caption("🔹 GradientBoosting regression")
        st.caption(f"🔹 Input: target={r['target_angle']}°, rx={r['rx_value']}°")
    
    with col_b:
        st.metric(
            "Calculated Radius",
            f"{r['predicted_radius']:.2f} mm",
            help="Inner bend radius from inverse model"
        )
        st.caption("**How Calculated:**")
        st.caption("🔹 Inverse Model: `(target, rx) → radius`")
        st.caption("🔹 Multi-output regression")
        st.caption(f"🔹 Predicted from same model as RYR")
    
    with col_c:
        st.metric(
            "Angle Ratio (Ks)",
            f"{r['Ks']:.4f}",
            help="Ks = FEM_predicted_angle / target_angle"
        )
        st.caption("**Formula:**")
        st.caption(f"🔹 `Ks = {r['predicted_final']:.2f}° / {r['target_angle']:.1f}°`")
        st.caption(f"🔹 `Ks = {r['Ks']:.4f}`")
        st.caption("🔹 Forward Model: `(rx, ryr) → stable_angle`")
    
    with col_d:
        st.metric(
            "Required Tool Angle",
            f"{r['overbend_angle']:.2f}°",
            delta=f"{r['overbend_angle'] - r['target_angle']:.2f}°",
            help="Tool angle computed to achieve target after relaxation"
        )
        st.caption("**Formula:**")
        st.caption(f"🔹 `Tool_angle = target / Ks`")
        st.caption(f"🔹 `= {r['target_angle']:.1f}° / {r['Ks']:.4f}`")
        st.caption(f"🔹 `= {r['overbend_angle']:.2f}°`")
    
    
    st.header("📈 Engineering Visualizations")
    
    angle_diff = r['predicted_final'] - r['overbend_angle']
    
    st.markdown(f"""
    <div style="background-color: #1c4a31; padding: 1rem; border-radius: 5px; border-left: 4px solid #f7f9fa;">
        <h4 style="color: white; margin-top: 0;">📊 What These Plots Show:</h4>
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <strong><br><u>First Plot: Bending Profile</strong></u></br>
                <br>
                • Gray dashed line: Initial flat plate (0° neutral state)<br>
                • Red line: Part shape at maximum tool loading ({r['overbend_angle']:.2f}°)<br>
                • Blue line: Part shape after tool release / stabilization ({r['predicted_final']:.2f}°)<br>
                • Gap (red ↔ blue): Material relaxation magnitude of {abs(angle_diff):.2f}°
            </div>
            <div style="flex: 1;">
                <strong><br><u>Second Plot: Angle Comparison</u></strong></br>
                <br>
                • Green bar: Target stable angle ({r['target_angle']:.2f}°)<br>
                • Red bar: Required tool angle ({r['overbend_angle']:.2f}°)<br>
                • Blue bar: FEM-predicted stable angle ({r['predicted_final']:.2f}°)<br>
                • Annotation: Angle ratio Ks = {r['Ks']:.4f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Bending Profile")
    fig_curves = viz.plot_bending_curves(
        r['target_angle'], r['overbend_angle'], r['predicted_final']
    )
    st.plotly_chart(fig_curves, use_conatiner_width=True)
    
    st.markdown("---")
    
    st.markdown("#### Angle Comparison")
    fig_angles = viz.plot_angle_comparison(
        r['target_angle'], r['overbend_angle'], r['predicted_final'], r['Ks']
    )
    st.plotly_chart(fig_angles, use_container_width=True)
    
    # ========================================================================
    # TOOLPATH GENERATION
    # ========================================================================
    
    st.markdown("---")
    st.header("🤖 Robot Toolpath")
    
    toolpath_df = toolpath_gen.generate_simple_toolpath(
        r['rx_value'], r['predicted_ryr'], r['overbend_angle'], r['target_angle']
    )
    
    st.info(f"""
    **📋 Toolpath Table Columns Explained:**
    
    - **Step:** Sequential command number (1 to {len(toolpath_df)})
    - **RX (°):** Tool rotation parameter (from 0° to {r['rx_value']}°)
    - **RYR:** Secondary constraint parameter (from 0 to {r['predicted_ryr']:.3f})
    - **Tool Angle (°):** Physical tool rotation (absolute value of RX)
    - **Target Angle (°):** Desired stable part angle ({r['target_angle']:.2f}°)
    - **Overbend (°):** Required tool angle at maximum loading ({r['overbend_angle']:.2f}°)
    
    **How Generated:**
    Linear interpolation from neutral position (0°, 0) to final tool position ({r['rx_value']}°, {r['predicted_ryr']:.3f})
    with {len(toolpath_df)} discrete steps.
    """)
    
    # Display table
    st.dataframe(
        toolpath_df.style.format({
            'RX (°)': '{:.2f}',
            'RYR': '{:.3f}',
            'Tool Angle (°)': '{:.2f}',
            'Target Angle (°)': '{:.2f}',
            'Overbend (°)': '{:.2f}'
        }),
        use_container_width=True,
        height=300
    )
    
    # ========================================================================
    # EXPORT OPTIONS
    # ========================================================================
    
    st.markdown("### 📥 Export Options")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        # Toolpath CSV
        csv_toolpath = toolpath_df.to_csv(index=False)
        st.download_button(
            "📄 Download Toolpath (CSV)",
            csv_toolpath,
            file_name=f"toolpath_rx{r['rx_value']}_angle{r['target_angle']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Summary report
        summary = toolpath_gen.generate_summary(
            toolpath_df, r['rx_value'], r['predicted_ryr'],
            r['target_angle'], r['overbend_angle'], r['Ks']
        )
        st.download_button(
            "📋 Download Summary Report",
            summary,
            file_name=f"summary_rx{r['rx_value']}_angle{r['target_angle']}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # ========================================================================
    # ENGINEERING NOTES
    # ========================================================================
    
    deviation_val = abs(r['predicted_final'] - r['target_angle'])
    tool_offset = r['overbend_angle'] - r['target_angle']
    
    # Determine behavior type
    if r['predicted_final'] > r['overbend_angle']:
        behavior_desc = "⚠️ Material shows angle INCREASE after tool release (strain hardening / residual stress effects)"
    elif r['predicted_final'] < r['overbend_angle']:
        behavior_desc = "✓ Material shows classical springback behavior (elastic recovery reduces angle)"
    else:
        behavior_desc = "✓ Material shows minimal relaxation after tool release"
    
    st.markdown("---")
    st.info(f"""
    **✓ Calculation Complete**
    
    - Target stable angle: **{r['target_angle']}°**
    - FEM-predicted stable angle: **{r['predicted_final']:.2f}°**
    - Deviation from target: **{deviation_val:.2f}°**
    - Tool angle offset: **{tool_offset:.2f}°**
    
    **Material Behavior:** {behavior_desc}
    
    **Robot Command:** Set tool to {r['overbend_angle']:.2f}°. After release, part will stabilize at approximately {r['predicted_final']:.2f}°.
    """)
    
    # Reset button
    if st.button("🔄 New Calculation", use_container_width=True):
        st.session_state.computed = False
        st.rerun()

    # ========================================================================
    # CALCULATION PIPELINE EXPLANATION
    # ========================================================================
    
    st.markdown("---")
    st.header("🔬 Calculation Pipeline (Step-by-Step)")
    
    with st.expander("📖 Click to see detailed calculation methodology", expanded=True):
        
        angle_ratio_pct = abs((r['Ks'] - 1) * 100)
        validation_check = r['overbend_angle'] * r['Ks']
        validation_error = abs(validation_check - r['target_angle'])
        
        st.markdown(f"""
        ### 🔬 Detailed Calculation Pipeline
        
        <style>
        .step-block {{
            margin-bottom: 25px;
            font-size: 1.05rem;
            line-height: 1.6;
            color: #ecf0f1;
        }}
        .step-title {{
            font-weight: bold;
            font-size: 1.2rem;
            margin-bottom: 5px;
            color: #2ecc71;
            border-bottom: 1px solid #7f8c8d;
            padding-bottom: 5px;
            display: inline-block;
            width: 100%;
        }}
        .info-row {{
            margin-bottom: 4px;
        }}
        .highlight-val {{
            font-weight: bold;
            color: #f1c40f;
        }}
        .label {{
            font-weight: bold;
            color: #bdc3c7;
        }}
        .separator {{
            color: #7f8c8d;
            margin: 0 10px;
        }}
        </style>

        <div class="step-block">
            <div class="step-title">Step 1: Inverse Model Prediction (GradientBoosting)</div>
            <div class="info-row">
                <span class="label">Input:</span> Target stable angle <span class="highlight-val">{r['target_angle']:.2f}°</span>, RX <span class="highlight-val">{r['rx_value']:.2f}°</span>
                <span class="separator">|</span> 
                <span class="label">Output:</span> RYR <span class="highlight-val">{r['predicted_ryr']:.4f}</span>, Radius <span class="highlight-val">{r['predicted_radius']:.2f} mm</span>
            </div>
            <div class="info-row">
                <span class="label">Mapping:</span> <code>(target_stable_angle, rx) → (ryr, radius)</code>
                <span class="separator">|</span>
                <span class="label">Why ML?:</span> Complex material response; no analytical inverse solution.
            </div>
        </div>

        <div class="step-block">
            <div class="step-title">Step 2: Forward Model Prediction (RandomForest)</div>
            <div class="info-row">
                <span class="label">Input:</span> RX <span class="highlight-val">{r['rx_value']:.2f}°</span>, RYR <span class="highlight-val">{r['predicted_ryr']:.4f}</span>
                <span class="separator">|</span>
                <span class="label">Output:</span> Stable Angle <span class="highlight-val">{r['predicted_final']:.2f}°</span>
            </div>
            <div class="info-row">
                <span class="label">Mapping:</span> <code>(rx, ryr) → stable_angle_post_release</code>
                <span class="separator">|</span>
                <span class="label">Why RF?:</span> Captures FEM-validated material behavior without constitutive assumptions.
            </div>
        </div>

        <div class="step-block">
            <div class="step-title">Step 3: Angle Ratio (Ks) Calculation</div>
            <div class="info-row">
                <span class="label">Formula:</span> <code>Ks = FEM_predicted_stable / target_stable</code>
                <span class="separator">|</span>
                <span class="label">Calculation:</span> {r['predicted_final']:.2f}° / {r['target_angle']:.2f}° = <span class="highlight-val">{r['Ks']:.4f}</span>
            </div>
            <div class="info-row">
                <span class="label">Result:</span> Ks = <span class="highlight-val">{r['Ks']:.4f}</span> ({angle_ratio_pct:.1f}% deviation from unity)
                <span class="separator">|</span>
                <i>Ks quantifies material relaxation behavior after tool release</i>
            </div>
        </div>

        <div class="step-block">
            <div class="step-title">Step 4: Tool Angle Calculation</div>
            <div class="info-row">
                <span class="label">Command Logic:</span> Tool set to <span class="highlight-val">{r['overbend_angle']:.2f}°</span> → Part relaxes to ~<span class="highlight-val">{r['predicted_final']:.2f}°</span>
            </div>
            <div class="info-row">
                <span class="label">Calculation:</span> Tool_angle = {r['target_angle']:.2f}° / {r['Ks']:.4f} = <span class="highlight-val">{r['overbend_angle']:.2f}°</span>
                <span class="separator">|</span>
                <span class="label">Validation:</span> {r['overbend_angle']:.2f}° × {r['Ks']:.4f} ≈ {r['target_angle']:.2f}° (Diff: {validation_error:.2f}°)
            </div>
        </div>

        <div style="font-size: 1.0rem; margin-top: 10px; padding-top: 10px; border-top: 1px dotted #ccc;">
            <span class="label">Summary Chain:</span>
            Target ({r['target_angle']}°) → 
            [Inverse] → RYR ({r['predicted_ryr']:.3f}) → 
            [Forward] → Stable angle ({r['predicted_final']:.2f}°) → 
            [Ks Calc] → {r['Ks']:.4f} → 
            [Tool Angle] → <b>Command tool to {r['overbend_angle']:.2f}°</b>
        </div>
        
        <div style="margin-top: 8px; font-size: 0.85rem; color: #7f8c8d;">
            <b>Model Specs:</b> Forward Model (RandomForest, MAE~9.4°, R²~0.72) • Inverse Model (GradientBoosting, MAE~0.30) • Trained on 16 LS-DYNA FEM simulations (Stainless Steel).
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    "<div style='text-align: center; color: #95a5a6; font-size: 0.9rem;'>"
    "ML Assisted Sheet Metal Bending — FEM-Validated Digital Twin<br>"
    "Powered by RandomForest (Forward Model) + GradientBoosting (Inverse Model)"
    "</div>",
    unsafe_allow_html=True
)
