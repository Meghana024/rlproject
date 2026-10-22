import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hospital_mdp import (
    MDPParams,
    HospitalMDP,
    ACTION_NAMES,
    ACTION_ICONS,
    valid_actions
)

# Set page config
st.set_page_config(
    page_title="Hospital Resource MDP Optimizer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    .action-badge-crit {
        background-color: #ffebe9;
        color: #cf222e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #ff8182;
    }
    .action-badge-norm {
        background-color: #ddf4ff;
        color: #0969da;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #54aeff;
    }
    .action-badge-wait {
        background-color: #fff8c5;
        color: #9a6700;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #d4a72c;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_mdp(gamma, theta, crit_arr_p, norm_arr_p, crit_dis_p, norm_dis_p,
                   reward_crit, reward_norm, queue_crit_penalty, queue_norm_penalty,
                   terminal_overload_penalty):
    params = MDPParams(
        gamma=gamma,
        theta=theta,
        crit_arr_p=crit_arr_p,
        norm_arr_p=norm_arr_p,
        crit_dis_p=crit_dis_p,
        norm_dis_p=norm_dis_p,
        reward_crit=reward_crit,
        reward_norm=reward_norm,
        queue_crit_penalty=queue_crit_penalty,
        queue_norm_penalty=queue_norm_penalty,
        terminal_overload_penalty=terminal_overload_penalty
    )
    mdp = HospitalMDP(params)
    V, Q, policy, delta_history = mdp.solve_value_iteration()
    return mdp, V, Q, policy, delta_history


# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("⚙️ MDP Configuration")
st.sidebar.markdown("Customize stochastic transition rates & rewards:")

with st.sidebar.expander("Arrival & Discharge Probabilities", expanded=False):
    crit_arr = st.slider("Critical Arrival Prob P(ca=1)", 0.05, 0.95, 0.25, 0.05)
    norm_arr = st.slider("Non-Critical Arrival Prob P(na=1)", 0.05, 0.95, 0.55, 0.05)
    crit_dis = st.slider("Critical Discharge Prob P(cd)", 0.05, 0.90, 0.20, 0.05)
    norm_dis = st.slider("Non-Critical Discharge Prob P(nd)", 0.05, 0.90, 0.15, 0.05)

with st.sidebar.expander("Rewards & Penalties", expanded=False):
    r_crit = st.number_input("Reward: Critical Admission", value=40.0, step=5.0)
    r_norm = st.number_input("Reward: Non-Critical Admission", value=10.0, step=5.0)
    pen_qc = st.number_input("Penalty: Critical Queue (per patient)", value=12.0, step=1.0)
    pen_qn = st.number_input("Penalty: Non-Critical Queue (per patient)", value=3.0, step=0.5)
    pen_overload = st.number_input("Penalty: Terminal Overload State", value=100.0, step=10.0)

with st.sidebar.expander("Algorithm Hyperparameters", expanded=False):
    gamma = st.slider("Discount Factor (γ)", 0.50, 0.99, 0.90, 0.01)
    theta = 1e-6

# Load / solve MDP
with st.spinner("Computing Optimal Policy via Value Iteration..."):
    mdp, V, Q, policy, delta_history = get_cached_mdp(
        gamma, theta, crit_arr, norm_arr, crit_dis, norm_dis,
        r_crit, r_norm, pen_qc, pen_qn, pen_overload
    )

st.sidebar.success(f"MDP Solved: {len(delta_history)} iterations to reach ε < {theta}")
st.sidebar.markdown("---")
st.sidebar.caption("System Capacity: 2 beds / ventilators / staff per department. Total States: 729.")


# --- MAIN HEADER ---
st.title("🏥 Hospital Resource Allocation & Admission Decision System")
st.markdown("""
This system uses a **Markov Decision Process (MDP)** solved with **Value Iteration** to determine 
optimal patient admissions and resource deployment under stochastic patient arrivals and discharges.
""")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Real-Time Decision Support",
    "📈 Monte Carlo Simulation",
    "🔍 Full Policy & State Explorer",
    "⚡ Value Iteration Convergence"
])

# --- TAB 1: Real-Time Decision Support ---
with tab1:
    st.subheader("Current Hospital State Input")
    st.markdown("Set the current hospital resource and queue levels to view the mathematically optimal action:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 👥 Waiting Queues")
        qc = st.selectbox("Critical Queue (Qc)", options=[0, 1, 2], index=1, help="Patients waiting for ICU")
        qn = st.selectbox("Non-Critical Queue (Qn)", options=[0, 1, 2], index=1, help="Patients waiting for General Ward")

    with col2:
        st.markdown("##### 🛏️ Available Beds")
        bi = st.selectbox("Available ICU Beds (Bi)", options=[0, 1, 2], index=2, help="Empty ICU beds ready for use")
        bg = st.selectbox("Available General Beds (Bg)", options=[0, 1, 2], index=2, help="Empty general beds ready for use")

    with col3:
        st.markdown("##### 🩺 Equipment & Staff")
        v = st.selectbox("Available Ventilators (V)", options=[0, 1, 2], index=2, help="Functioning ventilators available")
        s = st.selectbox("Available Staff Units (S)", options=[0, 1, 2], index=2, help="Doctors/nurses ready for assignment")

    current_state = (qc, qn, bg, bi, v, s)
    state_idx = mdp.state_to_idx[current_state]
    best_action = policy[current_state]
    valid_acts = valid_actions(current_state)
    q_vals = Q[current_state]

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Optimal Admission Recommendation")
        
        if current_state == mdp.terminal_state:
            st.error("⚠️ CRITICAL OVERLOAD STATE: All queues full (2, 2) and all resources depleted (0, 0, 0, 0).")
        else:
            if best_action == 1:
                st.markdown(f"<div class='action-badge-crit'>🚨 ADMIT CRITICAL PATIENT</div>", unsafe_allow_html=True)
                st.info("Resources allocated: 1 ICU Bed, 1 Ventilator, 1 Staff. Critical Queue decremented.")
            elif best_action == 2:
                st.markdown(f"<div class='action-badge-norm'>🩺 ADMIT NON-CRITICAL PATIENT</div>", unsafe_allow_html=True)
                st.info("Resources allocated: 1 General Bed, 1 Staff. Non-Critical Queue decremented.")
            else:
                st.markdown(f"<div class='action-badge-wait'>⏳ HOLD / WAIT</div>", unsafe_allow_html=True)
                st.warning("Action: Wait. Conserving staff/beds for higher priority arrivals or waiting for pending discharges.")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Expected Value V(s)", f"{V[state_idx]:.2f}")
        with m_col2:
            st.metric("Action Confidence Q*(s, a)", f"{q_vals[best_action]:.2f}")

        # Feasibility report
        st.markdown("##### Resource Feasibility Check:")
        crit_feasible = (qc > 0 and bi > 0 and v > 0 and s > 0)
        norm_feasible = (qn > 0 and bg > 0 and s > 0)
        st.write(f"- Critical Admission Feasible: {'✅ Yes' if crit_feasible else '❌ No (Missing Queue, ICU Bed, Ventilator, or Staff)'}")
        st.write(f"- Non-Critical Admission Feasible: {'✅ Yes' if norm_feasible else '❌ No (Missing Queue, General Bed, or Staff)'}")

    with res_col2:
        st.subheader("Action Value Comparison Q(s, a)")
        chart_data = []
        for a_id in [0, 1, 2]:
            a_name = ACTION_NAMES[a_id]
            is_valid = a_id in valid_acts
            q_val = q_vals.get(a_id, None)
            chart_data.append({
                "Action": a_name,
                "Status": "Valid" if is_valid else "Constrained (Invalid)",
                "Q-Value": q_val if is_valid else 0.0
            })
        
        df_q = pd.DataFrame(chart_data)
        st.dataframe(df_q, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        colors = ['#f39c12' if a==0 else ('#e74c3c' if a==1 else '#3498db') for a in df_q["Action"].map({"Wait": 0, "Admit Critical": 1, "Admit Non-Critical": 2})]
        bars = ax.bar(df_q["Action"], df_q["Q-Value"], color=colors, alpha=0.85)
        ax.set_ylabel("Expected Cumulative Return")
        ax.set_title("Q(s, a) per Candidate Action")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        st.pyplot(fig)


# --- TAB 2: Monte Carlo Simulation ---
with tab2:
    st.subheader("Hospital Dynamic Operations Simulation")
    st.markdown("Simulate hospital patient flow over multiple episodes under the MDP optimal policy.")

    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        episodes = st.slider("Number of Episodes", 50, 500, 200, 50)
        max_steps = st.slider("Steps per Episode", 10, 100, 50, 10)
        seed = st.number_input("Random Seed", value=42, step=1)
        run_sim = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

    if run_sim:
        with st.spinner("Simulating hospital episodes..."):
            sim_res = mdp.simulate(policy, num_episodes=episodes, max_steps=max_steps, seed=seed)
        
        with sim_col2:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Mean Discounted Return", f"{sim_res['mean_return']:.2f} ± {sim_res['std_return']:.2f}")
            sc2.metric("Mean Episode Length", f"{sim_res['mean_length']:.1f} steps")
            total_actions = sum(sim_res['action_counts'].values())
            crit_share = (sim_res['action_counts'][1] / total_actions * 100) if total_actions > 0 else 0
            sc3.metric("Critical Admissions Made", f"{sim_res['action_counts'][1]} ({crit_share:.1f}%)")

        st.markdown("---")
        st.subheader("Running Average Cumulative Reward")
        running_avg_df = pd.DataFrame({
            "Episode": range(1, episodes + 1),
            "Running Average Return": sim_res["running_avg"]
        })
        st.line_chart(running_avg_df.set_index("Episode"))

        st.subheader("Sample Episode Operational Trajectory (First 15 Steps)")
        if sim_res["sample_trajectory"]:
            traj_data = []
            for t in sim_res["sample_trajectory"][:15]:
                traj_data.append({
                    "Step": t["step"],
                    "State (Qc, Qn, Bg, Bi, V, S)": str(t["state"]),
                    "Action Chosen": t["action"],
                    "Reward": t["reward"],
                    "Next State": str(t["next_state"])
                })
            st.dataframe(pd.DataFrame(traj_data), use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Simulation' to evaluate policy performance.")


# --- TAB 3: Policy & State Explorer ---
with tab3:
    st.subheader("Global Optimal Policy Table (All 729 States)")
    st.markdown("Inspect or search the optimal action and calculated value function for every system configuration:")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        filter_action = st.multiselect("Filter by Action", ["Wait", "Admit Critical", "Admit Non-Critical"], default=["Wait", "Admit Critical", "Admit Non-Critical"])
    with f_col2:
        filter_qc = st.multiselect("Filter by Critical Queue (Qc)", [0, 1, 2], default=[0, 1, 2])
    with f_col3:
        filter_qn = st.multiselect("Filter by Non-Critical Queue (Qn)", [0, 1, 2], default=[0, 1, 2])

    table_rows = []
    for s in mdp.states:
        act_name = ACTION_NAMES[policy[s]]
        if act_name in filter_action and s[0] in filter_qc and s[1] in filter_qn:
            table_rows.append({
                "State (Qc, Qn, Bg, Bi, V, S)": str(s),
                "Qc": s[0],
                "Qn": s[1],
                "Bg": s[2],
                "Bi": s[3],
                "V": s[4],
                "S": s[5],
                "Value V(s)": round(V[mdp.state_to_idx[s]], 4),
                "Optimal Policy": act_name,
                "Q(Wait)": round(Q[s].get(0, np.nan), 4),
                "Q(Critical)": round(Q[s].get(1, np.nan), 4),
                "Q(Non-Critical)": round(Q[s].get(2, np.nan), 4)
            })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, height=400)

    # Download button for CSV
    csv_bytes = df_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Policy Table as CSV",
        data=csv_bytes,
        file_name="hospital_optimal_policy.csv",
        mime="text/csv"
    )


# --- TAB 4: Convergence ---
with tab4:
    st.subheader("Value Iteration Algorithmic Convergence")
    st.markdown(f"Value Iteration completed in **{len(delta_history)}** iterations with stopping threshold $\\theta = 10^{{-6}}$.")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(delta_history) + 1), delta_history, marker='o', markersize=3, color='#2980b9')
    ax.set_yscale('log')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Bellman Error ||V_{k+1} - V_k||_∞")
    ax.set_title("Convergence Error (Log Scale)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    st.pyplot(fig)
