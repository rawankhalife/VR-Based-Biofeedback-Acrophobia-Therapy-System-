import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="VR Biofeedback Dashboard", layout="wide")

# ── Custom CSS for upload screen ───────────────────────────────────────────────
upload_css = """
<style>
.upload-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 500px;
    padding: 3rem 2rem;
    text-align: center;
}

.upload-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    border: 2px dashed #4a90e2;
    border-radius: 12px;
    padding: 3rem 2rem;
    max-width: 500px;
    width: 100%;
}

.upload-icon {
    font-size: 64px;
    margin-bottom: 1.5rem;
    opacity: 0.8;
}

.upload-title {
    font-size: 28px;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 0.5rem 0;
}

.upload-subtitle {
    font-size: 16px;
    color: #666;
    margin: 0 0 2rem 0;
}

.upload-hint {
    font-size: 14px;
    color: #999;
    margin-top: 2rem;
    line-height: 1.6;
}

.upload-hint strong {
    color: #4a90e2;
}
</style>
"""

# ── Upload screen ──────────────────────────────────────────────────────────────
if "session_data" not in st.session_state:
    st.markdown(upload_css, unsafe_allow_html=True)
    
    with st.container():
        st.markdown(
            """
            <div class="upload-container">
                <div class="upload-card">
                    <div class="upload-icon">📁</div>
                    <div class="upload-title">VR Biofeedback Analysis</div>
                    <div class="upload-subtitle">Upload a patient session to review</div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Choose a JSON session file",
            type="json",
            label_visibility="collapsed"
        )
        
        st.markdown(
            """
                    <div class="upload-hint">
                        <strong>Supported format:</strong> JSON session files exported from VR therapy system<br>
                        Contains baseline metrics, exposure data, and event logs
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.session_state["session_data"] = data
                st.session_state["session_name"] = uploaded_file.name
                st.success("✓ File loaded successfully. Opening dashboard...")
                st.rerun()
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please check the format and try again.")
    st.stop()

# ── Analysis page ──────────────────────────────────────────────────────────────
data = st.session_state["session_data"]

col_title, col_btn = st.columns([6, 1])
with col_title:
    st.title("VR Biofeedback – Patient Session Dashboard")
    st.caption(f"File: {st.session_state['session_name']}")
with col_btn:
    st.write("")
    if st.button("↻ Upload new file"):
        del st.session_state["session_data"]
        del st.session_state["session_name"]
        st.rerun()

# ── Everything else stays the same ─────────────────────────────────────────────
baseline_df = pd.DataFrame(data.get("baselineMetrics", []))
exposure_df = pd.DataFrame(data.get("metrics", []))
events_df   = pd.DataFrame(data.get("events", []))

if baseline_df.empty and exposure_df.empty:
    st.warning("No baseline or exposure data found in this file.")
    st.stop()

def prepare_df(df):
    if df.empty:
        return df
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "bpm" in df.columns:
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    if "gsr" in df.columns:
        df["gsr"] = df["gsr"].fillna("NA").astype(str)
    return df

baseline_df = prepare_df(baseline_df)
exposure_df = prepare_df(exposure_df)

session_id   = data.get("sessionId", "N/A")
raw_date     = data.get("date", "N/A")
duration     = round(float(data.get("durationSeconds", 0)), 2)

try:
    formatted_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %I:%M %p")
except Exception:
    formatted_date = raw_date

baseline_avg_bpm    = round(baseline_df["bpm"].dropna().mean(), 1) if not baseline_df.empty and "bpm" in baseline_df.columns else "N/A"
exposure_avg_bpm    = round(exposure_df["bpm"].dropna().mean(), 1) if not exposure_df.empty and "bpm" in exposure_df.columns else "N/A"
exposure_max_bpm    = int(exposure_df["bpm"].dropna().max()) if not exposure_df.empty and "bpm" in exposure_df.columns and not exposure_df["bpm"].dropna().empty else "N/A"
exposure_min_bpm    = int(exposure_df["bpm"].dropna().min()) if not exposure_df.empty and "bpm" in exposure_df.columns and not exposure_df["bpm"].dropna().empty else "N/A"

baseline_gsr = "N/A"
if not baseline_df.empty and "gsr" in baseline_df.columns:
    valid = baseline_df[baseline_df["gsr"].str.upper() != "NA"]["gsr"]
    if not valid.empty:
        baseline_gsr = valid.mode().iloc[0]

dominant_exposure_gsr = "N/A"
if not exposure_df.empty and "gsr" in exposure_df.columns:
    valid = exposure_df[exposure_df["gsr"].str.upper() != "NA"]["gsr"]
    if not valid.empty:
        dominant_exposure_gsr = valid.mode().iloc[0]

event_count       = len(events_df)
baseline_samples  = len(baseline_df)
exposure_samples  = len(exposure_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Session ID", session_id)
c2.metric("Session Date", formatted_date)
c3.metric("Session Duration (sec)", duration)
c4.metric("Total Events", event_count)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Baseline Summary")
    b1, b2 = st.columns(2)
    b1.metric("Baseline Avg BPM", baseline_avg_bpm)
    b2.metric("Baseline GSR", baseline_gsr)
    st.caption(f"Baseline samples recorded: {baseline_samples}")

with right:
    st.subheader("Exposure Summary")
    e1, e2, e3 = st.columns(3)
    e1.metric("Exposure Avg BPM", exposure_avg_bpm)
    e2.metric("Exposure Max BPM", exposure_max_bpm)
    e3.metric("Exposure Min BPM", exposure_min_bpm)
    st.caption(f"Dominant exposure GSR: {dominant_exposure_gsr} | Exposure samples recorded: {exposure_samples}")

st.divider()

st.subheader("Session Interpretation")

interpretation = "No exposure data available."
if not exposure_df.empty and baseline_avg_bpm != "N/A" and exposure_avg_bpm != "N/A":
    if exposure_max_bpm != "N/A" and exposure_max_bpm >= baseline_avg_bpm * 1.4:
        interpretation = "This session shows a strong stress response during exposure compared to baseline."
    elif exposure_avg_bpm >= baseline_avg_bpm * 1.2:
        interpretation = "This session shows an elevated physiological response during exposure."
    else:
        interpretation = "This session remained relatively close to baseline, indicating a stable response."

st.info(interpretation)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Baseline Signals", "Exposure Signals", "GSR Summary", "Events", "Raw Tables"
])

with tab1:
    st.subheader("Baseline Heart Rate")
    if not baseline_df.empty and "bpm" in baseline_df.columns:
        fig = px.line(baseline_df, x="timestamp", y="bpm", title="Baseline BPM vs Time", markers=False)
        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="BPM")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No baseline BPM data available.")

with tab2:
    st.subheader("Exposure Heart Rate")
    if not exposure_df.empty and "bpm" in exposure_df.columns:
        fig = px.line(
            exposure_df, x="timestamp", y="bpm",
            color="systemState" if "systemState" in exposure_df.columns else None,
            title="Exposure BPM vs Time", markers=False
        )
        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="BPM")
        st.plotly_chart(fig, use_container_width=True)

        if "level" in exposure_df.columns:
            level_fig = px.scatter(
                exposure_df, x="timestamp", y="level",
                color="systemState" if "systemState" in exposure_df.columns else None,
                title="Exposure Level Progression"
            )
            level_fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Level")
            st.plotly_chart(level_fig, use_container_width=True)
    else:
        st.write("No exposure BPM data available.")

with tab3:
    st.subheader("GSR State Distribution")
    if not exposure_df.empty and "gsr" in exposure_df.columns:
        gsr_plot_df = exposure_df[exposure_df["gsr"].str.upper() != "NA"].copy()
        if not gsr_plot_df.empty:
            gsr_counts = gsr_plot_df["gsr"].value_counts().reset_index()
            gsr_counts.columns = ["GSR State", "Count"]
            fig = px.bar(gsr_counts, x="Count", y="GSR State", color="GSR State",
                         text="Count", orientation="h", title="Exposure GSR Distribution")
            fig.update_traces(textposition="outside", cliponaxis=False)
            max_count = gsr_counts["Count"].max()
            fig.update_layout(xaxis_title="Count", yaxis_title="GSR State", showlegend=False,
                               xaxis=dict(range=[0, max_count * 1.2 if max_count > 0 else 1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No valid exposure GSR data available.")
    else:
        st.write("No GSR data available.")

with tab4:
    st.subheader("Session Events")
    if not events_df.empty:
        if "timestamp" in events_df.columns:
            events_df["timestamp"] = pd.to_numeric(events_df["timestamp"], errors="coerce").round(2)
        cols = [c for c in ["timestamp", "eventType", "details", "level", "path"] if c in events_df.columns]
        st.dataframe(events_df[cols], use_container_width=True, height=350)

        if "eventType" in events_df.columns:
            event_counts = events_df["eventType"].value_counts().reset_index()
            event_counts.columns = ["Event Type", "Count"]
            fig = px.bar(event_counts, x="Count", y="Event Type", orientation="h",
                         text="Count", title="Event Frequency")
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No events found in this session file.")

with tab5:
    st.subheader("Raw Baseline Data")
    if not baseline_df.empty:
        baseline_show = baseline_df.copy()
        if "timestamp" in baseline_show.columns:
            baseline_show["timestamp"] = baseline_show["timestamp"].round(4)
        st.dataframe(baseline_show, use_container_width=True, height=250)
    else:
        st.write("No baseline table available.")

    st.subheader("Raw Exposure Data")
    if not exposure_df.empty:
        exposure_show = exposure_df.copy()
        if "timestamp" in exposure_show.columns:
            exposure_show["timestamp"] = exposure_show["timestamp"].round(4)
        st.dataframe(exposure_show, use_container_width=True, height=250)
    else:
        st.write("No exposure table available.")
