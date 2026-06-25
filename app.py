import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

st.set_page_config(page_title="Levodopa/Carbidopa PK Simulation", page_icon="💊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #2C3E50; font-family: 'Inter', sans-serif; font-weight: 800; }
    h2, h3 { color: #34495E; font-family: 'Inter', sans-serif; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #E0E0E0; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; font-size: 16px; }
    .stTabs [aria-selected="true"] { background-color: #F0F2F6; color: #1976D2; border-bottom: 3px solid #1976D2; }
</style>
""", unsafe_allow_html=True)

st.title("💊 Levodopa & Carbidopa PK Simulation")
st.markdown("""
Welcome to the interactive Pharmacokinetic (PK) simulation! This application uses a **two-compartment model** integrated with enzyme metabolism (AADC, COMT) and the Blood-Brain Barrier (BBB) to predict drug concentrations in dogs based on *in vitro* dissolution profiles.
""")

def mermaid(code: str):
    components.html(
        f"""
        <div class="mermaid" style="display: flex; justify-content: center; width: 100%;">
            {code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{startOnLoad: false, theme: 'base', themeVariables: {{ primaryColor: '#f4f4f9', primaryBorderColor: '#ccc', lineColor: '#999', fontFamily: 'Inter, sans-serif' }}}});
            
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if(entry.isIntersecting) {{
                        mermaid.init(undefined, ".mermaid");
                        observer.disconnect();
                    }}
                }});
            }});
            observer.observe(document.querySelector(".mermaid"));
        </script>
        """,
        height=700,
        scrolling=True
    )

# --- UI Sidebar ---
st.sidebar.header("📥 Input Data")
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

# --- Parameters Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Pharmacokinetic Parameters (Dog ~12kg)")

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

with st.sidebar.expander("Blood-Brain Barrier & CNS", expanded=False):
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

# Layout Tabs
tab_dashboard, tab_model, tab_data = st.tabs(["📈 Simulation Dashboard", "🧬 PK Model Diagram", "📊 Raw Data Table"])

with tab_data:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("In Vitro Release Data Table")
        st.info("💡 You can edit the values directly in the table below. Add/remove rows by clicking the left edge.")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save Data", type="primary"):
            edited_df.to_csv("sample_data.csv", index=False)
            st.success("Data successfully saved to `sample_data.csv`!")
    
    with col2:
        st.subheader("Release Profile")
        fig_release = go.Figure()
        fig_release.add_trace(go.Scatter(x=edited_df['Time_h'], y=edited_df['Levodopa_Released_mg'], mode='lines+markers', name='Levodopa (mg)', line=dict(color='#0288D1', width=3)))
        fig_release.add_trace(go.Scatter(x=edited_df['Time_h'], y=edited_df['Carbidopa_Released_mg'], mode='lines+markers', name='Carbidopa (mg)', line=dict(color='#D32F2F', width=3)))
        fig_release.update_layout(title="Cumulative Drug Release", xaxis_title="Time (h)", yaxis_title="Amount Released (mg)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter, sans-serif'))
        fig_release.update_xaxes(gridcolor='#eee')
        fig_release.update_yaxes(gridcolor='#eee')
        st.plotly_chart(fig_release, use_container_width=True)

time_arr = edited_df.iloc[:, 0].values
levo_mass = edited_df.iloc[:, 1].values
carbi_mass = edited_df.iloc[:, 2].values

def make_rate_func(time_arr, mass_arr):
    rates = np.diff(mass_arr) / np.diff(time_arr)
    def rate_func(t):
        if t < time_arr[0] or t >= time_arr[-1]: return 0.0
        idx = np.searchsorted(time_arr, t, side='right') - 1
        idx = min(max(idx, 0), len(rates) - 1)
        return rates[idx]
    return np.vectorize(rate_func)

rate_L = make_rate_func(time_arr, levo_mass)
rate_C = make_rate_func(time_arr, carbi_mass)

def pk_model(y, t, params, rate_L, rate_C):
    C_gi, C_cent, C_peri, L_gi, L_cent, L_peri, L_brain, DA_peri, DA_brain = y
    
    # Carbidopa parameters
    ka_C, F_C, kel_C, k12_C, k21_C, Vc_C, IC50 = params['ka_C'], params['F_C'], params['kel_C'], params['k12_C'], params['k21_C'], params['Vc_C'], params['IC50']
    
    # Levodopa parameters
    ka_L, F_L, kAADC, kCOMT, k12_L, k21_L = params['ka_L'], params['F_L'], params['kAADC'], params['kCOMT'], params['k12_L'], params['k21_L']
    kin_BBB, kout_BBB, Vc_L, kAADC_brain = params['kin_BBB'], params['kout_BBB'], params['Vc_L'], params['kAADC_brain']
    
    # Dopamine parameters
    kel_DA, kel_DA_brain = params['kel_DA'], params['kel_DA_brain']
    
    R_L = rate_L(t)
    R_C = rate_C(t)
    
    dC_gi_dt = R_C - ka_C * C_gi
    dC_cent_dt = F_C * ka_C * C_gi - kel_C * C_cent - k12_C * C_cent + k21_C * C_peri
    dC_peri_dt = k12_C * C_cent - k21_C * C_peri
    
    C_plasma = C_cent / Vc_C
    AADC_act = IC50 / (IC50 + C_plasma)
    
    dL_gi_dt = R_L - ka_L * L_gi
    dL_cent_dt = F_L * ka_L * L_gi - (kAADC * AADC_act + kCOMT) * L_cent - k12_L * L_cent + k21_L * L_peri - kin_BBB * L_cent + kout_BBB * L_brain
    dL_peri_dt = k12_L * L_cent - k21_L * L_peri
    dL_brain_dt = kin_BBB * L_cent - kout_BBB * L_brain - kAADC_brain * L_brain
    
    dDA_peri_dt = (kAADC * AADC_act) * L_cent - kel_DA * DA_peri
    dDA_brain_dt = kAADC_brain * L_brain - kel_DA_brain * DA_brain
    
    return [dC_gi_dt, dC_cent_dt, dC_peri_dt, dL_gi_dt, dL_cent_dt, dL_peri_dt, dL_brain_dt, dDA_peri_dt, dDA_brain_dt]

with tab_dashboard:
    # --- Simulation ---
    t_sim = np.linspace(0, 24, 500)
    y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    sol = odeint(pk_model, y0, t_sim, args=(params, rate_L, rate_C))
    
    C_plasma = sol[:, 1] / Vc_C
    L_plasma = sol[:, 4] / Vc_L
    L_brain_mass = sol[:, 6]
    DA_peri_mass = sol[:, 7]
    DA_brain_mass = sol[:, 8]
    
    V_brain = 0.12 # Assume Brain Volume is 1% of total body mass (~0.12 L)
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("<b>Levodopa and Carbidopa Plasma Concentrations</b>",
                                        "<b>Dopamine Concentrations (Peripheral vs Brain)</b>"),
                        vertical_spacing=0.1)
    
    fig.add_trace(go.Scatter(x=t_sim, y=L_plasma, name='Plasma Levodopa', line=dict(color='#0288D1', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_sim, y=C_plasma, name='Plasma Carbidopa', line=dict(color='#D32F2F', width=3)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=t_sim, y=DA_brain_conc, name='Brain Dopamine (mg/L)', line=dict(color='#388E3C', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_sim, y=DA_peri_mass, name='Peripheral DA (mg)', line=dict(color='#F57C00', dash='dash', width=2)), row=2, col=1)
    
    fig.update_yaxes(title_text="Concentration (mg/L)", row=1, col=1, gridcolor='#eee')
    fig.update_yaxes(title_text="Conc. / Mass", row=2, col=1, gridcolor='#eee')
    fig.update_xaxes(title_text="Time (h)", row=2, col=1, gridcolor='#eee')
    
    fig.update_layout(height=700, template="plotly_white", hovermode="x unified",
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter, sans-serif', size=13),
                      margin=dict(l=20, r=20, t=60, b=20))
    
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.markdown("<h3 style='text-align: center; color: #2C3E50;'>Compartmental Pharmacokinetic Model</h3>", unsafe_allow_html=True)
    mermaid_code = """
    %%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 50, 'rankSpacing': 60}}}%%
    graph LR
        subgraph Formulation [<b>💊 Tablet Formulation</b>]
            direction TB
            Tablet((Tablet))
        end

        subgraph GI_Tract [<b>Gut / GI Tract</b>]
            direction TB
            C_gi(Carbidopa GI)
            L_gi(Levodopa GI)
        end

        subgraph Central_Compartment [<b>Central / Plasma Compartment</b>]
            direction TB
            C_cent([Carbidopa Plasma])
            L_cent([Levodopa Plasma])
            DA_peri(Peripheral DA)
            3OMD(3-OMD)
        end

        subgraph Peripheral_Tissue [<b>Peripheral Tissue</b>]
            direction TB
            C_peri(Carbidopa Tissue)
            L_peri(Levodopa Tissue)
        end

        subgraph CNS [<b>Brain / CNS Compartment</b>]
            direction TB
            L_brain([Levodopa Brain])
            DA_brain([Brain DA])
        end

        Tablet -- "Rate_rel_C" --> C_gi
        Tablet -- "Rate_rel_L" --> L_gi

        C_gi -- "ka_C * F_C" --> C_cent
        L_gi -- "ka_L * F_L" --> L_cent

        C_cent -- "k12_C" --> C_peri
        C_peri -- "k21_C" --> C_cent
        C_cent -- "kel_C" --> ExC((Urine))

        L_cent -- "k12_L" --> L_peri
        L_peri -- "k21_L" --> L_cent

        L_cent -- "kAADC<br>(inhibited by C)" --> DA_peri
        DA_peri -- "kel_DA" --> ExD((Excreted))

        L_cent -- "kCOMT" --> 3OMD
        
        L_cent -- "kin_BBB<br>(LAT)" --> L_brain
        L_brain -- "kout_BBB" --> L_cent

        L_brain -- "kAADC_brain" --> DA_brain
        DA_brain -- "kel_DA_brain" --> ExDb((Excreted))
        
        classDef plasma fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#0D47A1;
        classDef brain fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#1B5E20;
        classDef tissue fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1px,color:#4A148C;
        classDef gi fill:#FFF3E0,stroke:#F57C00,stroke-width:1px,color:#E65100;
        classDef form fill:#ECEFF1,stroke:#607D8B,stroke-width:2px,color:#263238;
        classDef default rx:10px,ry:10px,font-family:Inter;
        
        class C_cent,L_cent,DA_peri,3OMD plasma;
        class L_brain,DA_brain brain;
        class C_peri,L_peri tissue;
        class C_gi,L_gi gi;
        class Tablet form;
    """
    mermaid(mermaid_code)
    
    st.markdown("---")
    st.markdown("<h3 style='color: #2C3E50;'>📚 Model Explanation & Mathematical Equations</h3>", unsafe_allow_html=True)
    st.markdown("""
    This mathematical model uses a system of Ordinary Differential Equations (ODEs) to represent drug mass transfer. 
    It combines **Pharmacokinetics** (absorption, two-compartment distribution, elimination) with **Pharmacodynamics/Metabolism** (enzyme inhibition via an $IC_{50}$ model).
    """)
    
    with st.expander("🔬 Carbidopa Equations (AADC Inhibitor)", expanded=False):
        st.markdown("Carbidopa inhibits the AADC enzyme in the peripheral blood but does not cross the Blood-Brain Barrier (BBB).")
        st.latex(r"\frac{dC_{gi}}{dt} = Rate_{rel, C} - k_{a,C} \cdot C_{gi}")
        st.latex(r"\frac{dC_{cent}}{dt} = F_C \cdot k_{a,C} \cdot C_{gi} - k_{el,C} \cdot C_{cent} - k_{12,C} \cdot C_{cent} + k_{21,C} \cdot C_{peri}")
        st.latex(r"\frac{dC_{peri}}{dt} = k_{12,C} \cdot C_{cent} - k_{21,C} \cdot C_{peri}")
        st.markdown("**AADC Enzyme Inhibition (Emax Model):**")
        st.latex(r"Act_{AADC} = \frac{IC_{50}}{IC_{50} + C_{plasma}} \quad \text{where } C_{plasma} = \frac{C_{cent}}{V_{c,C}}")
    
    with st.expander("💊 Levodopa Equations (Prodrug)", expanded=False):
        st.markdown("Levodopa is a prodrug that is transported across the BBB by the LAT transporter. In the periphery, its metabolism to Dopamine is inhibited by Carbidopa.")
        st.latex(r"\frac{dL_{gi}}{dt} = Rate_{rel, L} - k_{a,L} \cdot L_{gi}")
        st.latex(r"\frac{dL_{cent}}{dt} = F_L \cdot k_{a,L} \cdot L_{gi} - \underbrace{(k_{AADC} \cdot Act_{AADC} + k_{COMT}) \cdot L_{cent}}_{\text{Metabolism}} - \underbrace{k_{12,L} \cdot L_{cent} + k_{21,L} \cdot L_{peri}}_{\text{Distribution}} - \underbrace{k_{in,BBB} \cdot L_{cent} + k_{out,BBB} \cdot L_{brain}}_{\text{BBB Transport}}")
        st.latex(r"\frac{dL_{peri}}{dt} = k_{12,L} \cdot L_{cent} - k_{21,L} \cdot L_{peri}")
        st.latex(r"\frac{dL_{brain}}{dt} = k_{in,BBB} \cdot L_{cent} - k_{out,BBB} \cdot L_{brain} - k_{AADC,brain} \cdot L_{brain}")
        
    with st.expander("🧠 Dopamine Equations (Active Metabolite)", expanded=False):
        st.markdown("**Peripheral Dopamine** (causes side effects like nausea and cardiovascular issues):")
        st.latex(r"\frac{dDA_{peri}}{dt} = (k_{AADC} \cdot Act_{AADC}) \cdot L_{cent} - k_{el,DA} \cdot DA_{peri}")
        st.markdown("**Brain Dopamine** (provides therapeutic Parkinson's relief):")
        st.latex(r"\frac{dDA_{brain}}{dt} = k_{AADC,brain} \cdot L_{brain} - k_{el,DA,brain} \cdot DA_{brain}")
