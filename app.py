import json
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="VR Biofeedback Dashboard",
    layout="wide",
    page_icon="🧠"
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- Styling ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #eef4ff 0%, #f9fbff 100%);
    }

    h1, h2, h3 {
        color: #1f2a44;
    }

    .hero-card {
        background: linear-gradient(135deg, #4f8cff 0%, #6fc3ff 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 24px;
        box-shadow: 0 12px 30px rgba(79,140,255,0.25);
    }

    .hero-card p, .hero-card li {
        color: rgba(255,255,255,0.95);
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 16px;
        border-left: 5px solid #4f8cff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px 18px;
        border-radius: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #4f8cff !important;
        color: white !important;
    }

    .stButton button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #4f8cff, #6fc3ff);
        color: white;
        font-weight: 600;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #3b73e0, #5ab4ff);
        color: white;
    }

    .status-pill {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }

    .status-green {
        background: #e8f7ee;
        color: #18794e;
    }

    .status-yellow {
        background: #fff4db;
        color: #9a6700;
    }

    .status-red {
        background: #fdecec;
        color: #b42318;
    }

    .analysis-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def prepare_df(df):
    if df.empty:
        return df

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    if "bpm" in df.columns:
        df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")

    if "level" in df.columns:
        df["level"] = pd.to_numeric(df["level"], errors="coerce")

    if "gsr" in df.columns:
        df["gsr"] = df["gsr"].fillna("NA").astype(str)

    if "systemState" in df.columns:
        df["systemState"] = df["systemState"].fillna("Unknown").astype(str)

    if "path" in df.columns:
        df["path"] = df["path"].fillna("None").astype(str)

    return df


def load_json(uploaded_file):
    try:
        uploaded_file.seek(0)
        return json.load(uploaded_file)
    except Exception:
        return None


def reset_uploader():
    st.session_state.uploader_key += 1
    st.rerun()


def compute_behavior_metrics(events_df, exposure_df, baseline_avg_bpm, session_duration):
    metrics = {
        "level_times": [],
        "reaction_times": [],
        "safe_zone_time": 0.0,
        "high_stress_ratio": 0.0,
        "elevated_ratio": 0.0,
        "completed_levels": 0,
        "stabilization_times": [],
        "bpm_trend_delta": None,
        "avg_bpm_per_level": pd.DataFrame(),
        "max_bpm_per_level": pd.DataFrame(),
        "time_above_elevated_sec": 0.0,
        "time_above_high_sec": 0.0,
    }

    if not events_df.empty and "timestamp" in events_df.columns:
        events_df = events_df.copy().sort_values("timestamp")

        # --- Level durations from LevelEntered to next LevelEntered or session end ---
        if "eventType" in events_df.columns and "level" in events_df.columns:
            level_entries = events_df[events_df["eventType"] == "LevelEntered"].copy()
            for i in range(len(level_entries)):
                start_row = level_entries.iloc[i]
                start_time = float(start_row["timestamp"])
                level_value = start_row["level"]

                if i < len(level_entries) - 1:
                    end_time = float(level_entries.iloc[i + 1]["timestamp"])
                else:
                    end_time = float(session_duration)

                duration = max(0, end_time - start_time)
                metrics["level_times"].append((level_value, duration))

        # --- Reaction time: ProceedPromptShown -> next ProceedButtonPressed ---
        if "eventType" in events_df.columns:
            prompts = events_df[events_df["eventType"] == "ProceedPromptShown"]
            clicks = events_df[events_df["eventType"] == "ProceedButtonPressed"]

            used_click_indices = set()

            for _, prompt in prompts.iterrows():
                future_clicks = clicks[clicks["timestamp"] > prompt["timestamp"]]
                chosen_click_idx = None

                for click_idx, click_row in future_clicks.iterrows():
                    if click_idx not in used_click_indices:
                        chosen_click_idx = click_idx
                        rt = float(click_row["timestamp"]) - float(prompt["timestamp"])
                        if rt >= 0:
                            metrics["reaction_times"].append(rt)
                            used_click_indices.add(click_idx)
                        break

        # --- Safe zone time ---
        if "eventType" in events_df.columns:
            safe_in = events_df[events_df["eventType"] == "SafeZoneEntered"].copy()
            safe_out = events_df[events_df["eventType"] == "SafeZoneExited"].copy()

            safe_in = safe_in.sort_values("timestamp").reset_index(drop=True)
            safe_out = safe_out.sort_values("timestamp").reset_index(drop=True)

            out_pointer = 0
            total_safe_time = 0.0

            for _, entry in safe_in.iterrows():
                entry_time = float(entry["timestamp"])

                while out_pointer < len(safe_out) and float(safe_out.iloc[out_pointer]["timestamp"]) <= entry_time:
                    out_pointer += 1

                if out_pointer < len(safe_out):
                    exit_time = float(safe_out.iloc[out_pointer]["timestamp"])
                    total_safe_time += max(0, exit_time - entry_time)
                    out_pointer += 1

            metrics["safe_zone_time"] = total_safe_time

        # --- Stabilization times: LevelEntered -> StableThresholdReached in same level ---
        if "eventType" in events_df.columns and "level" in events_df.columns:
            stable_events = events_df[events_df["eventType"] == "StableThresholdReached"].copy()
            level_entries = events_df[events_df["eventType"] == "LevelEntered"].copy()

            for _, stable_row in stable_events.iterrows():
                level_value = stable_row["level"]
                stable_time = float(stable_row["timestamp"])

                matching_entries = level_entries[
                    (level_entries["level"] == level_value) &
                    (level_entries["timestamp"] <= stable_time)
                ]

                if not matching_entries.empty:
                    latest_entry_time = float(matching_entries.iloc[-1]["timestamp"])
                    metrics["stabilization_times"].append((level_value, stable_time - latest_entry_time))

    # --- Exposure-derived analysis ---
    if not exposure_df.empty and "bpm" in exposure_df.columns:
        valid_exposure = exposure_df.dropna(subset=["bpm", "timestamp"]).copy()
        valid_exposure = valid_exposure.sort_values("timestamp")

        if not valid_exposure.empty and isinstance(baseline_avg_bpm, (int, float)):
            elevated_threshold = baseline_avg_bpm * 1.2
            high_threshold = baseline_avg_bpm * 1.4

            elevated_rows = valid_exposure[valid_exposure["bpm"] >= elevated_threshold]
            high_rows = valid_exposure[valid_exposure["bpm"] >= high_threshold]

            metrics["elevated_ratio"] = len(elevated_rows) / len(valid_exposure)
            metrics["high_stress_ratio"] = len(high_rows) / len(valid_exposure)

            # Approximate time above threshold using timestamp differences
            valid_exposure["dt"] = valid_exposure["timestamp"].diff().fillna(0)
            valid_exposure["dt"] = valid_exposure["dt"].clip(lower=0)

            metrics["time_above_elevated_sec"] = valid_exposure.loc[
                valid_exposure["bpm"] >= elevated_threshold, "dt"
            ].sum()

            metrics["time_above_high_sec"] = valid_exposure.loc[
                valid_exposure["bpm"] >= high_threshold, "dt"
            ].sum()

        if "level" in valid_exposure.columns:
            level_stats = valid_exposure.groupby("level")["bpm"].agg(["mean", "max", "min", "std", "count"]).reset_index()
            metrics["avg_bpm_per_level"] = level_stats[["level", "mean"]].rename(columns={"mean": "avg_bpm"})
            metrics["max_bpm_per_level"] = level_stats[["level", "max"]].rename(columns={"max": "max_bpm"})

        if "level" in valid_exposure.columns and not valid_exposure["level"].dropna().empty:
            metrics["completed_levels"] = int(valid_exposure["level"].max())

        # Simple trend: first 20% avg vs last 20% avg
        n = len(valid_exposure)
        if n >= 10:
            window = max(1, int(n * 0.2))
            first_avg = valid_exposure["bpm"].iloc[:window].mean()
            last_avg = valid_exposure["bpm"].iloc[-window:].mean()
            metrics["bpm_trend_delta"] = last_avg - first_avg

    return metrics


def compute_session_score(behavior):
    score = 100.0

    score += behavior.get("completed_levels", 0) * 8

    score -= behavior.get("high_stress_ratio", 0) * 35
    score -= behavior.get("elevated_ratio", 0) * 20
    score -= behavior.get("safe_zone_time", 0) * 0.12

    reaction_times = behavior.get("reaction_times", [])
    if reaction_times:
        score -= (sum(reaction_times) / len(reaction_times)) * 0.5

    trend_delta = behavior.get("bpm_trend_delta", None)
    if trend_delta is not None and trend_delta < 0:
        score += min(abs(trend_delta) * 2, 10)

    return max(0, min(100, round(score, 1)))


def classify_session_status(baseline_avg_bpm, exposure_avg_bpm, exposure_max_bpm, behavior):
    if baseline_avg_bpm == "N/A" or exposure_avg_bpm == "N/A" or exposure_max_bpm == "N/A":
        return "Unknown", "status-yellow", "🟡"

    if behavior.get("high_stress_ratio", 0) >= 0.15 or exposure_max_bpm >= baseline_avg_bpm * 1.4:
        return "High Stress", "status-red", "🔴"

    if behavior.get("elevated_ratio", 0) >= 0.20 or exposure_avg_bpm >= baseline_avg_bpm * 1.2:
        return "Elevated", "status-yellow", "🟡"

    return "Stable", "status-green", "🟢"


def build_interpretation(status, behavior):
    safe_zone_time = behavior.get("safe_zone_time", 0)
    high_ratio = behavior.get("high_stress_ratio", 0)
    trend_delta = behavior.get("bpm_trend_delta", None)

    if status == "High Stress":
        text = "This session shows a strong physiological stress response during exposure."
        if safe_zone_time > 0:
            text += f" The patient also spent {safe_zone_time:.1f} seconds in a safe zone, suggesting avoidance behavior."
        return text

    if status == "Elevated":
        text = "This session shows an elevated physiological response compared to baseline."
        if trend_delta is not None and trend_delta < 0:
            text += " However, the later part of the session appears calmer than the early part, suggesting partial adaptation."
        return text

    text = "This session remained relatively close to baseline, indicating stable exposure tolerance."
    if high_ratio == 0 and safe_zone_time == 0:
        text += " No clear high-stress period or safe-zone avoidance was detected."
    return text


# ---------- Landing ----------
left, right = st.columns([1.25, 1], vertical_alignment="center")

with left:
    st.markdown("""
    <div class="hero-card">
        <h1 style="margin-bottom:0.5rem; color:white;">VR Biofeedback Dashboard</h1>
        <p style="margin-top:0;">
            Review patient session data from a single uploaded JSON file.
        </p>
        <hr style="margin:1rem 0 1.2rem 0; border:none; border-top:1px solid rgba(255,255,255,0.25);">
        <p style="margin-bottom:0.8rem;">This dashboard helps you inspect:</p>
        <ul style="line-height:1.9;">
            <li>Baseline physiological signals</li>
            <li>Exposure-phase heart rate trends</li>
            <li>GSR state distribution</li>
            <li>Session events and progression</li>
            <li>Behavioral and therapy-response indicators</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.subheader("Upload Session File")
        uploaded_file = st.file_uploader(
            "Choose a JSON session file",
            type=["json"],
            key=f"session_uploader_{st.session_state.uploader_key}",
            help="Upload one session JSON file to open the analysis dashboard."
        )
        st.caption("Accepted format: .json")

if uploaded_file is None:
    st.info("Upload a session JSON file to open the analysis page.")
    st.stop()

data = load_json(uploaded_file)

if data is None:
    st.error("The uploaded file is not a valid JSON file.")
    st.stop()

baseline_df = pd.DataFrame(data.get("baselineMetrics", []))
exposure_df = pd.DataFrame(data.get("metrics", []))
events_df = pd.DataFrame(data.get("events", []))

if baseline_df.empty and exposure_df.empty:
    st.warning("No baseline or exposure data found in this file.")
    st.stop()

baseline_df = prepare_df(baseline_df)
exposure_df = prepare_df(exposure_df)
events_df = prepare_df(events_df)

session_id = data.get("sessionId", "N/A")
raw_date = data.get("date", "N/A")
duration = round(float(data.get("durationSeconds", 0)), 2)

try:
    formatted_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %I:%M %p")
except Exception:
    formatted_date = raw_date

baseline_avg_bpm = round(baseline_df["bpm"].dropna().mean(), 1) if not baseline_df.empty and "bpm" in baseline_df.columns and not baseline_df["bpm"].dropna().empty else "N/A"
exposure_avg_bpm = round(exposure_df["bpm"].dropna().mean(), 1) if not exposure_df.empty and "bpm" in exposure_df.columns and not exposure_df["bpm"].dropna().empty else "N/A"
exposure_max_bpm = int(exposure_df["bpm"].dropna().max()) if not exposure_df.empty and "bpm" in exposure_df.columns and not exposure_df["bpm"].dropna().empty else "N/A"
exposure_min_bpm = int(exposure_df["bpm"].dropna().min()) if not exposure_df.empty and "bpm" in exposure_df.columns and not exposure_df["bpm"].dropna().empty else "N/A"

baseline_gsr = "N/A"
if not baseline_df.empty and "gsr" in baseline_df.columns:
    valid_baseline_gsr = baseline_df[baseline_df["gsr"].str.upper() != "NA"]["gsr"]
    if not valid_baseline_gsr.empty:
        baseline_gsr = valid_baseline_gsr.mode().iloc[0]

dominant_exposure_gsr = "N/A"
if not exposure_df.empty and "gsr" in exposure_df.columns:
    valid_exposure_gsr = exposure_df[exposure_df["gsr"].str.upper() != "NA"]["gsr"]
    if not valid_exposure_gsr.empty:
        dominant_exposure_gsr = valid_exposure_gsr.mode().iloc[0]

event_count = len(events_df)
baseline_samples = len(baseline_df)
exposure_samples = len(exposure_df)

behavior = compute_behavior_metrics(
    events_df=events_df,
    exposure_df=exposure_df,
    baseline_avg_bpm=baseline_avg_bpm if baseline_avg_bpm != "N/A" else 0,
    session_duration=duration
)

session_score = compute_session_score(behavior)
status, status_class, status_icon = classify_session_status(
    baseline_avg_bpm,
    exposure_avg_bpm,
    exposure_max_bpm,
    behavior
)
interpretation = build_interpretation(status, behavior)

st.markdown("---")

header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("Patient Session Analysis")
    st.caption(f"Loaded file: {uploaded_file.name}")
    st.markdown(
        f'<div class="status-pill {status_class}">{status_icon} Session Status: {status}</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.write("")
    st.write("")
    st.button("Upload Another File", on_click=reset_uploader, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Session ID", session_id)
c2.metric("Session Date", formatted_date)
c3.metric("Duration (sec)", duration)
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

if status == "High Stress":
    st.error("🔴 " + interpretation)
elif status == "Elevated":
    st.warning("🟡 " + interpretation)
else:
    st.success("🟢 " + interpretation)

st.divider()

st.subheader("Behavioral Analysis")

m1, m2, m3, m4 = st.columns(4)

avg_rt = round(sum(behavior["reaction_times"]) / len(behavior["reaction_times"]), 2) if behavior["reaction_times"] else "N/A"
m1.metric("Therapy Performance Score", session_score)
m2.metric("Avg Decision Time (sec)", avg_rt)
m3.metric("Time in Safe Zone (sec)", round(behavior["safe_zone_time"], 2))
m4.metric("Completed Level", behavior["completed_levels"])

m5, m6, m7, m8 = st.columns(4)
m5.metric("Elevated Stress %", f"{behavior['elevated_ratio'] * 100:.1f}%")
m6.metric("High Stress %", f"{behavior['high_stress_ratio'] * 100:.1f}%")
m7.metric("Time Above Elevated (sec)", round(behavior["time_above_elevated_sec"], 2))
m8.metric("Time Above High Stress (sec)", round(behavior["time_above_high_sec"], 2))

if behavior["bpm_trend_delta"] is not None:
    trend_text = "decrease" if behavior["bpm_trend_delta"] < 0 else "increase"
    st.caption(f"BPM trend across the session: {abs(behavior['bpm_trend_delta']):.2f} BPM {trend_text} from early to late exposure.")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Baseline Signals",
    "Exposure Signals",
    "Level Analysis",
    "GSR Summary",
    "Events",
    "Raw Tables"
])

with tab1:
    st.subheader("Baseline Heart Rate")
    if not baseline_df.empty and "bpm" in baseline_df.columns:
        fig = px.line(
            baseline_df,
            x="timestamp",
            y="bpm",
            title="Baseline BPM vs Time",
            markers=False
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time (seconds)",
            yaxis_title="BPM"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No baseline BPM data available.")

with tab2:
    st.subheader("Exposure Heart Rate")
    if not exposure_df.empty and "bpm" in exposure_df.columns:
        fig = px.line(
            exposure_df.sort_values("timestamp"),
            x="timestamp",
            y="bpm",
            color="systemState" if "systemState" in exposure_df.columns else None,
            title="Exposure BPM vs Time",
            markers=False
        )

        if baseline_avg_bpm != "N/A":
            fig.add_hline(
                y=baseline_avg_bpm,
                line_dash="dash",
                annotation_text="Baseline",
                annotation_position="top left"
            )
            fig.add_hline(
                y=baseline_avg_bpm * 1.2,
                line_dash="dot",
                annotation_text="Elevated Threshold",
                annotation_position="top left"
            )
            fig.add_hline(
                y=baseline_avg_bpm * 1.4,
                line_dash="dot",
                annotation_text="High Stress Threshold",
                annotation_position="top left"
            )

        if not events_df.empty and "timestamp" in events_df.columns and "eventType" in events_df.columns:
            important_events = events_df[
                events_df["eventType"].isin([
                    "LevelEntered",
                    "ProceedPromptShown",
                    "ProceedButtonPressed",
                    "StableThresholdReached",
                    "SafeZoneEntered",
                    "SafeZoneExited"
                ])
            ].copy()

            if not important_events.empty:
                # place event markers near top of the graph
                y_marker = exposure_df["bpm"].max() if not exposure_df["bpm"].dropna().empty else 0
                important_events["marker_y"] = y_marker

                fig.add_trace(go.Scatter(
                    x=important_events["timestamp"],
                    y=important_events["marker_y"],
                    mode="markers",
                    name="Events",
                    text=important_events["eventType"],
                    hovertemplate="Time: %{x:.2f}s<br>Event: %{text}<extra></extra>"
                ))

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time (seconds)",
            yaxis_title="BPM"
        )
        st.plotly_chart(fig, use_container_width=True)

        if "level" in exposure_df.columns:
            level_fig = px.scatter(
                exposure_df.sort_values("timestamp"),
                x="timestamp",
                y="level",
                color="systemState" if "systemState" in exposure_df.columns else None,
                title="Exposure Level Progression"
            )
            level_fig.update_layout(
                template="plotly_white",
                xaxis_title="Time (seconds)",
                yaxis_title="Level"
            )
            st.plotly_chart(level_fig, use_container_width=True)
    else:
        st.write("No exposure BPM data available.")

with tab3:
    st.subheader("Level-Based Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        if behavior["avg_bpm_per_level"] is not None and not behavior["avg_bpm_per_level"].empty:
            fig = px.bar(
                behavior["avg_bpm_per_level"],
                x="level",
                y="avg_bpm",
                title="Average BPM per Level"
            )
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Level",
                yaxis_title="Average BPM"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No level-based BPM averages available.")

    with col_b:
        if behavior["max_bpm_per_level"] is not None and not behavior["max_bpm_per_level"].empty:
            fig = px.bar(
                behavior["max_bpm_per_level"],
                x="level",
                y="max_bpm",
                title="Maximum BPM per Level"
            )
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Level",
                yaxis_title="Max BPM"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No level-based BPM maxima available.")

    st.subheader("Time Spent per Level")
    if behavior["level_times"]:
        level_time_df = pd.DataFrame(behavior["level_times"], columns=["Level", "Duration (sec)"])
        fig = px.bar(
            level_time_df,
            x="Level",
            y="Duration (sec)",
            title="Duration per Level"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(level_time_df, use_container_width=True)
    else:
        st.write("No level duration data available.")

    st.subheader("Stabilization Time per Level")
    if behavior["stabilization_times"]:
        stability_df = pd.DataFrame(behavior["stabilization_times"], columns=["Level", "Time to Stabilize (sec)"])
        fig = px.bar(
            stability_df,
            x="Level",
            y="Time to Stabilize (sec)",
            title="Time Needed to Reach Stability"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stability_df, use_container_width=True)
    else:
        st.write("No stabilization events found for this session.")

with tab4:
    st.subheader("GSR State Distribution")

    if not exposure_df.empty and "gsr" in exposure_df.columns:
        gsr_plot_df = exposure_df[exposure_df["gsr"].str.upper() != "NA"].copy()

        if not gsr_plot_df.empty:
            gsr_counts = gsr_plot_df["gsr"].value_counts().reset_index()
            gsr_counts.columns = ["GSR State", "Count"]

            fig = px.bar(
                gsr_counts,
                x="Count",
                y="GSR State",
                color="GSR State",
                text="Count",
                orientation="h",
                title="Exposure GSR Distribution"
            )
            max_count = gsr_counts["Count"].max()
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Count",
                yaxis_title="GSR State",
                showlegend=False,
                xaxis=dict(range=[0, max_count * 1.2 if max_count > 0 else 1])
            )
            st.plotly_chart(fig, use_container_width=True)

            if "level" in gsr_plot_df.columns:
                st.subheader("Dominant GSR by Level")
                gsr_level = (
                    gsr_plot_df.groupby(["level", "gsr"])
                    .size()
                    .reset_index(name="count")
                )
                fig2 = px.bar(
                    gsr_level,
                    x="level",
                    y="count",
                    color="gsr",
                    barmode="group",
                    title="GSR Counts per Level"
                )
                fig2.update_layout(
                    template="plotly_white",
                    xaxis_title="Level",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("No valid exposure GSR data available.")
    else:
        st.write("No GSR data available.")

with tab5:
    st.subheader("Session Events")
    if not events_df.empty:
        events_show = events_df.copy()
        if "timestamp" in events_show.columns:
            events_show["timestamp"] = pd.to_numeric(events_show["timestamp"], errors="coerce").round(2)

        cols = [c for c in ["timestamp", "eventType", "details", "level", "path"] if c in events_show.columns]
        st.dataframe(events_show[cols], use_container_width=True, height=350)

        if "eventType" in events_show.columns:
            event_counts = events_show["eventType"].value_counts().reset_index()
            event_counts.columns = ["Event Type", "Count"]

            fig = px.bar(
                event_counts,
                x="Count",
                y="Event Type",
                orientation="h",
                text="Count",
                title="Event Frequency"
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No events found in this session file.")

with tab6:
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

    st.subheader("Raw Events Data")
    if not events_df.empty:
        events_show = events_df.copy()
        if "timestamp" in events_show.columns:
            events_show["timestamp"] = events_show["timestamp"].round(4)
        st.dataframe(events_show, use_container_width=True, height=250)
    else:
        st.write("No events table available.")
