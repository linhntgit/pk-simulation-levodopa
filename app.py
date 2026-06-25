import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(page_title="Levodopa/Carbidopa PK Simulation", layout="wide")

st.title("Levodopa and Carbidopa Pharmacokinetic Simulation")
st.markdown("""
This application uses a two-compartment pharmacokinetic (PK) model integrated with enzyme metabolism (AADC, COMT) and the Blood-Brain Barrier (BBB) to predict Levodopa, Carbidopa, and Dopamine concentrations in dogs based on in vitro dissolution data.
""")

def mermaid(code: str):
    components.html(
        f"""
        <div class="mermaid">
            {code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true}});</script>
        """,
        height=650
    )

st.header("Dissolution & Pharmacokinetic Model Diagram")
mermaid_code = """
graph TD
    subgraph Formulation
        Tablet[Tablet Formulation]
    end

    subgraph GI_Tract[Gastrointestinal Tract]
        C_gi[Carbidopa GI]
        L_gi[Levodopa GI]
    end

    subgraph Central_Compartment[Central Compartment / Plasma]
        C_cent[Carbidopa Plasma]
        L_cent[Levodopa Plasma]
        DA_peri[Peripheral Dopamine]
        3OMD[3-OMD]
    end

    subgraph Peripheral_Tissue[Peripheral Tissue Compartment]
        C_peri[Carbidopa Tissue]
        L_peri[Levodopa Tissue]
    end

    subgraph CNS[Brain / CNS Compartment]
        L_brain[Levodopa Brain]
        DA_brain[Brain Dopamine]
    end

    Tablet -- Rate_rel_C --> C_gi
    Tablet -- Rate_rel_L --> L_gi

    C_gi -- "ka_C * F_C" --> C_cent
    L_gi -- "ka_L * F_L" --> L_cent

    C_cent -- "k12_C" --> C_peri
    C_peri -- "k21_C" --> C_cent
    C_cent -- "kel_C (Urine)" --> Excreted_C[Excreted]

    L_cent -- "k12_L" --> L_peri
    L_peri -- "k21_L" --> L_cent

    L_cent -- "kAADC (inhibited by C_cent)" --> DA_peri
    DA_peri -- "kel_DA" --> Excreted_DA_peri[Excreted]

    L_cent -- "kCOMT" --> 3OMD
    
    L_cent -- "kin_BBB (LAT transporter)" --> L_brain
    L_brain -- "kout_BBB" --> L_cent

    L_brain -- "kAADC_brain" --> DA_brain
    DA_brain -- "kel_DA_brain" --> Excreted_DA_brain[Excreted]
    
    style C_cent fill:#ffcccc,stroke:#ff0000
    style L_cent fill:#cce5ff,stroke:#0000ff
    style DA_brain fill:#ccffcc,stroke:#008000
"""
mermaid(mermaid_code)


# --- Helper function for Rate ---
def make_rate_func(time_arr, mass_arr):
    # Calculate piece-wise constant rate (mg/h)
    rates = np.diff(mass_arr) / np.diff(time_arr)
    def rate_func(t):
        if t < time_arr[0] or t >= time_arr[-1]:
            return 0.0
        idx = np.searchsorted(time_arr, t, side='right') - 1
        idx = min(max(idx, 0), len(rates) - 1)
        return rates[idx]
    return np.vectorize(rate_func)

# --- ODE Model ---
def pk_model(y, t, params, rate_L, rate_C):
    C_gi, C_cent, C_peri, L_gi, L_cent, L_peri, L_brain, DA_peri, DA_brain = y
    
    # Carbidopa parameters
    ka_C = params['ka_C']
    F_C = params['F_C']
    kel_C = params['kel_C']
    k12_C = params['k12_C']
    k21_C = params['k21_C']
    Vc_C = params['Vc_C']
    IC50 = params['IC50']
    
    # Levodopa parameters
    ka_L = params['ka_L']
    F_L = params['F_L']
    kAADC = params['kAADC']
    kCOMT = params['kCOMT']
    k12_L = params['k12_L']
    k21_L = params['k21_L']
    kin_BBB = params['kin_BBB']
    kout_BBB = params['kout_BBB']
    Vc_L = params['Vc_L']
    kAADC_brain = params['kAADC_brain']
    
    # Dopamine parameters
    kel_DA = params['kel_DA']
    kel_DA_brain = params['kel_DA_brain']
    
    R_L = rate_L(t)
    R_C = rate_C(t)
    
    # Carbidopa Equations
    dC_gi_dt = R_C - ka_C * C_gi
    dC_cent_dt = F_C * ka_C * C_gi - kel_C * C_cent - k12_C * C_cent + k21_C * C_peri
    dC_peri_dt = k12_C * C_cent - k21_C * C_peri
    
    C_plasma = C_cent / Vc_C
    AADC_act = IC50 / (IC50 + C_plasma)
    
    # Levodopa Equations
    dL_gi_dt = R_L - ka_L * L_gi
    dL_cent_dt = F_L * ka_L * L_gi - (kAADC * AADC_act + kCOMT) * L_cent - k12_L * L_cent + k21_L * L_peri - kin_BBB * L_cent + kout_BBB * L_brain
    dL_peri_dt = k12_L * L_cent - k21_L * L_peri
    dL_brain_dt = kin_BBB * L_cent - kout_BBB * L_brain - kAADC_brain * L_brain
    
    # Dopamine Equations
    dDA_peri_dt = (kAADC * AADC_act) * L_cent - kel_DA * DA_peri
    dDA_brain_dt = kAADC_brain * L_brain - kel_DA_brain * DA_brain
    
    return [dC_gi_dt, dC_cent_dt, dC_peri_dt, dL_gi_dt, dL_cent_dt, dL_peri_dt, dL_brain_dt, dDA_peri_dt, dDA_brain_dt]


# --- UI ---
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload Dissolution Data (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using sample data `sample_data.csv`")
    try:
        df = pd.read_csv("sample_data.csv")
    except:
        df = pd.DataFrame({
            'Time_h': [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0],
            'Levodopa_Released_mg': [0.0, 35.0, 60.0, 100.0, 170.0, 200.0, 200.0],
            'Carbidopa_Released_mg': [0.0, 8.75, 15.0, 25.0, 42.5, 50.0, 50.0]
        })

st.subheader("In Vitro Release Data Table")
st.markdown("You can edit the values directly in the table below. Add/remove rows by clicking the left edge. Click **Save Data** to update `sample_data.csv`.")
edited_df = st.data_editor(df, num_rows="dynamic")

if st.button("Save Data"):
    edited_df.to_csv("sample_data.csv", index=False)
    st.success("Data successfully saved to `sample_data.csv`!")

time_arr = edited_df.iloc[:, 0].values
levo_mass = edited_df.iloc[:, 1].values
carbi_mass = edited_df.iloc[:, 2].values

rate_L = make_rate_func(time_arr, levo_mass)
rate_C = make_rate_func(time_arr, carbi_mass)

# --- Parameters Sidebar ---
st.sidebar.header("Pharmacokinetic Parameters (Dog ~12kg)")

with st.sidebar.expander("Carbidopa Parameters", expanded=False):
    ka_C = st.slider("ka_C (1/h) - Absorption Rate", 0.1, 5.0, 1.04, step=0.01)
    F_C = st.slider("F_C - Bioavailability", 0.1, 1.0, 0.88, step=0.01)
    kel_C = st.slider("kel_C (1/h) - Elimination Rate", 0.01, 1.0, 0.138, step=0.001)
    Vc_C = st.slider("Vc_C (L) - Central Vol of Distribution", 1.0, 50.0, 20.0, step=0.5)
    k12_C = st.slider("k12_C (1/h) - Distribution 1->2", 0.0, 5.0, 0.5, step=0.1)
    k21_C = st.slider("k21_C (1/h) - Distribution 2->1", 0.0, 5.0, 0.5, step=0.1)
    IC50 = st.slider("IC50 (mg/L) - AADC 50% Inhibitory Conc", 0.01, 2.0, 0.1, step=0.01)

with st.sidebar.expander("Levodopa Parameters", expanded=False):
    ka_L = st.slider("ka_L (1/h) - Absorption Rate", 0.1, 5.0, 2.0, step=0.1)
    F_L = st.slider("F_L - Bioavailability", 0.1, 1.0, 0.3, step=0.01)
    Vc_L = st.slider("Vc_L (L) - Central Vol of Distribution", 1.0, 50.0, 18.0, step=0.5)
    kAADC = st.slider("kAADC (1/h) - Peripheral AADC Rate", 0.1, 20.0, 5.0, step=0.1)
    kCOMT = st.slider("kCOMT (1/h) - Peripheral COMT Rate", 0.01, 1.0, 0.05, step=0.01)
    k12_L = st.slider("k12_L (1/h) - Distribution 1->2", 0.0, 5.0, 1.0, step=0.1)
    k21_L = st.slider("k21_L (1/h) - Distribution 2->1", 0.0, 5.0, 1.0, step=0.1)

with st.sidebar.expander("Blood-Brain Barrier & CNS Parameters", expanded=False):
    kin_BBB = st.slider("kin_BBB (1/h) - Rate into Brain (LAT)", 0.001, 0.5, 0.01, step=0.001)
    kout_BBB = st.slider("kout_BBB (1/h) - Rate out of Brain", 0.01, 1.0, 0.1, step=0.01)
    kAADC_brain = st.slider("kAADC_brain (1/h) - Brain DA Formation", 0.1, 5.0, 1.0, step=0.1)
    kel_DA = st.slider("kel_DA (1/h) - Peripheral DA Elimination", 1.0, 20.0, 10.0, step=0.5)
    kel_DA_brain = st.slider("kel_DA_brain (1/h) - Brain DA Elimination", 0.1, 10.0, 5.0, step=0.1)

params = {
    'ka_C': ka_C, 'F_C': F_C, 'kel_C': kel_C, 'k12_C': k12_C, 'k21_C': k21_C, 'Vc_C': Vc_C, 'IC50': IC50,
    'ka_L': ka_L, 'F_L': F_L, 'kAADC': kAADC, 'kCOMT': kCOMT, 'k12_L': k12_L, 'k21_L': k21_L,
    'kin_BBB': kin_BBB, 'kout_BBB': kout_BBB, 'Vc_L': Vc_L, 'kAADC_brain': kAADC_brain,
    'kel_DA': kel_DA, 'kel_DA_brain': kel_DA_brain
}

# --- Simulation ---
t_sim = np.linspace(0, 24, 500)
y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0] # Initial states

sol = odeint(pk_model, y0, t_sim, args=(params, rate_L, rate_C))

# Extract results
C_plasma = sol[:, 1] / Vc_C
L_plasma = sol[:, 4] / Vc_L
L_brain_mass = sol[:, 6]
DA_peri_mass = sol[:, 7]
DA_brain_mass = sol[:, 8]

# Assume Brain Volume is 1% of total body mass (~0.12 L)
V_brain = 0.12
L_brain_conc = L_brain_mass / V_brain
DA_brain_conc = DA_brain_mass / V_brain

results_df = pd.DataFrame({
    'Time (h)': t_sim,
    'Levodopa Plasma (mg/L)': L_plasma,
    'Carbidopa Plasma (mg/L)': C_plasma,
    'Levodopa Brain (mg/L)': L_brain_conc,
    'Peripheral Dopamine (mg)': DA_peri_mass,
    'Brain Dopamine (mg/L)': DA_brain_conc
})

# --- Plotting ---
st.header("Simulation Results")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=("Levodopa and Carbidopa Plasma Concentrations",
                                    "Dopamine Concentrations (Peripheral vs Brain)"),
                    vertical_spacing=0.1)

# Plot 1: Plasma Levo & Carbi
fig.add_trace(go.Scatter(x=t_sim, y=L_plasma, name='Plasma Levodopa', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=t_sim, y=C_plasma, name='Plasma Carbidopa', line=dict(color='red')), row=1, col=1)
fig.update_yaxes(title_text="Concentration (mg/L)", row=1, col=1)

# Plot 2: Dopamine
fig.add_trace(go.Scatter(x=t_sim, y=DA_brain_conc, name='Brain Dopamine (mg/L)', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x=t_sim, y=DA_peri_mass, name='Peripheral Dopamine (Total mg)', line=dict(color='orange', dash='dash')), row=2, col=1)
fig.update_yaxes(title_text="Concentration / Mass", row=2, col=1)
fig.update_xaxes(title_text="Time (h)", row=2, col=1)

fig.update_layout(height=700, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detailed Simulation Data Table")
st.dataframe(results_df.iloc[::10, :].style.format("{:.3f}"))
