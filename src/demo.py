import streamlit as st
import pandas as pd
from typing import Dict, List
from rdflib import Graph, Namespace

# Importar o módulo de BN
from demo_inference import load_bayesian_network, predict_with_bayesian_network
from decision_common import (
    BN_MODEL_PATH,
    CAUSES,
    ONTOLOGY_PATH,
    load_knowledge_graph,
    make_decision,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Decision Strategies
DECISION_STRATEGIES = {
    "threshold_based": {
        "name": "Threshold-Based ",
        "description": "Decision based on overheat and cause probability thresholds.",
        "requires": ["overheat_threshold", "cause_threshold"]
    },
    "cost_only": {
        "name": "Cost-Based ",
        "description": "Decision based on overheat and cause probability thresholds, selecting the cheapest maintenance procedure.",
        "requires": ["overheat_threshold", "cause_threshold"]
    },
}

# Namespace da ontologia
NS = Namespace("http://www.semanticweb.org/admin/ontologies/2025/11/untitled-ontology-6/")

# ============================================================================
# KNOWLEDGE GRAPH LOADER
# ============================================================================

@st.cache_resource
def load_knowledge_graph_cached():
    return load_knowledge_graph(ONTOLOGY_PATH)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Industrial Equipment Diagnosis", layout="wide")
    
    st.title("🔧 Intelligent System for Industrial Equipment Failure Diagnosis")
    st.markdown("---")
    
    # ========================================================================
    # CARREGAR MODELOS (KG + BN)
    # ========================================================================
    try:
        kg, num_triples = load_knowledge_graph_cached()
    except Exception as e:
        st.error(f"❌ Error loading KG: {e}")
        st.stop()
    bn_model = load_bayesian_network(BN_MODEL_PATH)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # NOVO: Seleção de Estratégia
        st.subheader("🎯 Decision Strategy")
        selected_strategy = st.selectbox(
            "Strategy",
            options=list(DECISION_STRATEGIES.keys()),
            format_func=lambda x: DECISION_STRATEGIES[x]["name"],
            help="Select the decision-making approach"
        )
        
        # Mostra descrição da estratégia
        st.info(DECISION_STRATEGIES[selected_strategy]["description"])
        
        st.markdown("---")
        
        # NOVO: Parâmetros dinâmicos baseados na estratégia
        st.subheader("📊 Strategy Parameters")
        
        strategy_params = {}
        
        if selected_strategy == "threshold_based":
            strategy_params['overheat_threshold'] = st.slider(
                "Overheat Probability Threshold",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                help="Threshold para considerar risco de overheat"
            )
            strategy_params['cause_threshold'] = st.slider(
                "Cause Probability Threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Threshold para considerar uma causa como provável"
            )
        
        elif selected_strategy == "cost_only":
            strategy_params['overheat_threshold'] = st.slider(
                "Overheat Probability Threshold",
                min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                help="Threshold para considerar risco de overheat"
            )
            strategy_params['cause_threshold'] = st.slider(
                "Cause Probability Threshold",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="Threshold para considerar uma causa como provável"
            )
            strategy_params['m0_ref_eur'] = st.number_input(
                "Maintenance Reference Cost m0 (€)",
                min_value=1.0, value=100.0, step=10.0,
                help="Custo de referência para o score: J = m0·(m/m0)^(1−P(C))."
            )
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Status KG
        if kg is not None:
            st.success(f"✅ KG Connected ({num_triples} triples)")
        else:
            st.error("❌ KG Not Loaded")
        
        # Status BN
        if bn_model is not None:
            model_info = bn_model.get_model_info()
            st.success(f"✅ BN Model Loaded")
            st.caption(f"Nodes: {model_info['num_nodes']} | Edges: {model_info['num_edges']}")
        else:
            st.error("❌ BN Not Loaded")
    
    # ========================================================================
    # TABS
    # ========================================================================
    tab1, tab2 = st.tabs(["📄 Upload CSV", "✏️ Manual Input"])
    
    # ========================================================================
    # TAB 1: Upload CSV
    # ========================================================================
    with tab1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} instances")
            
            st.subheader("Select Instance to Diagnose")
            
            # Obter lista de máquinas únicas
            machines = sorted(df['machine_id'].unique())
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_machine = st.selectbox(
                    "Machine",
                    machines,
                    help="Select the machine to diagnose"
                )
            
            # Filtrar instâncias da máquina selecionada
            machine_df = df[df['machine_id'] == selected_machine].reset_index(drop=True)
            
            with col2:
                instance_idx = st.selectbox(
                    "Instance",
                    range(len(machine_df)),
                    format_func=lambda x: f"Instance {x+1} - {machine_df.iloc[x]['timestamp']}",
                    help=f"Select one of {len(machine_df)} instances from {selected_machine}"
                )
            
            st.markdown("---")
            st.subheader("Current Instance Data")
            instance_data = machine_df.iloc[instance_idx].to_dict()
            
            # Atributos usados no modelo BN (destaque)
            bn_features = ['spindle_temp', 'vibration_rms', 'coolant_flow']
            
            # Separar atributos
            bn_data = {k: v for k, v in instance_data.items() if k in bn_features}
            other_data = {k: v for k, v in instance_data.items() if k not in bn_features and k not in ['timestamp', 'machine_id']}
            
            # Mostrar features do BN em destaque
            st.markdown("**🎯 BN Input Features** (used by the model)")
            cols = st.columns(len(bn_data))
            for idx, (key, value) in enumerate(bn_data.items()):
                with cols[idx]:
                    st.metric(
                        label=key.replace('_', ' ').title(),
                        value=f"{value:.3f}" if isinstance(value, (int, float)) else value,
                        delta_color="off"
                    )
            
            # Mostrar outros atributos de forma mais compacta
            st.markdown("**📊 Other Sensor Data**")
            cols = st.columns(4)
            for idx, (key, value) in enumerate(other_data.items()):
                with cols[idx % 4]:
                    formatted_value = f"{value:.3f}" if isinstance(value, (int, float)) else value
                    st.caption(f"**{key.replace('_', ' ').title()}**: {formatted_value}")
            
            st.markdown("---")
            
            # Botão de diagnóstico maior e mais visível
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 **RUN DIAGNOSIS**", key="diagnose_csv", type="primary", use_container_width=True):
                    diagnose_instance(kg, bn_model, instance_data, selected_strategy, strategy_params)
    
    # ========================================================================
    # TAB 2: Manual Input
    # ========================================================================
    with tab2:
        st.subheader("Enter Instance Data Manually")
        
        # Features do BN em destaque
        st.markdown("**🎯 BN Input Features** (required)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spindle_temp = st.number_input(
                "🌡️ Spindle Temperature (°C)", 
                value=73.96,
                help="Critical parameter for BN inference"
            )
        
        with col2:
            vibration_rms = st.number_input(
                "📳 Vibration RMS", 
                value=0.871,
                help="Critical parameter for BN inference"
            )
        
        with col3:
            coolant_flow = st.number_input(
                "💧 Coolant Flow", 
                value=1.05,
                help="Critical parameter for BN inference"
            )
        
        st.markdown("---")
        
        # Outros atributos de forma mais compacta
        st.markdown("**📊 Other Sensor Data**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            timestamp = st.text_input("Timestamp", "2025-01-01T00:00:00")
            machine_id = st.selectbox("Machine ID", ["M-A", "M-B", "M-C", "M-D"])
        
        with col2:
            ambient_temp = st.number_input("Ambient Temp (°C)", value=24.36)
            feed_rate = st.number_input("Feed Rate", value=0.9)
        
        with col3:
            spindle_speed = st.number_input("Spindle Speed (RPM)", value=3600)
            load_pct = st.number_input("Load (%)", value=0.294)
        
        with col4:
            power_kw = st.number_input("Power (kW)", value=3.193)
            tool_wear = st.number_input("Tool Wear", value=0.0)
        
        manual_data = {
            'timestamp': timestamp,
            'machine_id': machine_id,
            'spindle_temp': spindle_temp,
            'ambient_temp': ambient_temp,
            'vibration_rms': vibration_rms,
            'coolant_flow': coolant_flow,
            'feed_rate': feed_rate,
            'spindle_speed': spindle_speed,
            'load_pct': load_pct,
            'power_kw': power_kw,
            'tool_wear': tool_wear
        }
        
        st.markdown("---")
        
        # Botão de diagnóstico maior e mais visível
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 **RUN DIAGNOSIS**", key="diagnose_manual", type="primary", use_container_width=True):
                diagnose_instance(kg, bn_model, manual_data, selected_strategy, strategy_params)

def diagnose_instance(kg: Graph, bn_model, instance_data: Dict,
                     selected_strategy: str, strategy_params: Dict):
    """Executa o diagnóstico para uma instância."""
    
    if kg is None:
        st.error("❌ Knowledge Graph not loaded. Cannot perform diagnosis.")
        return
    
    if bn_model is None:
        st.error("❌ Bayesian Network not loaded. Cannot perform diagnosis.")
        return
    
    with st.spinner("Running diagnosis..."):
        # Chama a Rede Bayesiana
        overheat_prob, causes_prob = predict_with_bayesian_network(bn_model, instance_data)
        
        # Toma decisão usando a estratégia selecionada
        decision = make_decision(
            kg, overheat_prob, causes_prob,
            selected_strategy,
            **strategy_params
        )
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    st.markdown("---")
    st.header("📋 Diagnosis Results")
    
    # Mostra qual estratégia foi usada
    st.info(f"📌 **Decision Strategy**: {DECISION_STRATEGIES[selected_strategy]['name']}")
    
    # Overheat Probability
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Overheat Risk")
        st.progress(overheat_prob)
        st.metric("Overheat Probability", f"{overheat_prob:.2%}")
    
    with col2:
        st.subheader("Status")
        if overheat_prob < 0.5:
            st.success("✅ Low risk")
        elif overheat_prob < 0.7:
            st.warning("⚠️ Medium risk")
        else:
            st.error("🔴 High risk")
    
    # Causes Probabilities
    st.subheader("Identified Causes")
    cause_cols = st.columns(4)
    
    # Se houver threshold, usa-o para destacar
    cause_threshold = strategy_params.get('cause_threshold', 0.5)
    
    for idx, (cause_id, prob) in enumerate(causes_prob.items()):
        with cause_cols[idx]:
            cause_name = CAUSES[cause_id]
            if prob >= cause_threshold:
                st.error(f"**{cause_id}**: {cause_name}")
                st.metric("Probability", f"{prob:.2%}", delta="Critical")
            else:
                st.info(f"**{cause_id}**: {cause_name}")
                st.metric("Probability", f"{prob:.2%}")
    
    # Mostra informações específicas da estratégia
    if decision['strategy_info']:
        with st.expander("📊 Strategy Details", expanded=False):
            for key, value in decision['strategy_info'].items():
                pretty_key = key.replace('_', ' ').title()

                # Show tables for DataFrames / list-of-dicts
                if isinstance(value, pd.DataFrame):
                    st.markdown(f"**{pretty_key}**")
                    st.dataframe(value, use_container_width=True)
                    continue
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    st.markdown(f"**{pretty_key}**")
                    st.dataframe(pd.DataFrame(value), use_container_width=True)
                    continue

                # Scalars
                if isinstance(value, float):
                    st.metric(pretty_key, f"{value:.4f}")
                elif isinstance(value, int):
                    st.metric(pretty_key, str(value))
                else:
                    # Fallback for strings/others
                    st.write(f"**{pretty_key}:** {value}")
    
    # Decision and Action
    st.markdown("---")
    st.header("🎯 Recommended Action")
    
    if decision['action'] == "CONTINUE":
        st.success("### ✅ CONTINUE OPERATION")
        st.write("The machine can continue working normally. All parameters are within acceptable ranges.")
    
    elif decision['action'] == "SLOW_DOWN":
        st.warning("### ⚠️ SLOW DOWN OPERATION")
        st.write("Overheat risk detected, but no specific cause identified above threshold.")
        st.write("**Recommendation**: Reduce machine speed and monitor closely.")
    
    elif decision['action'] == "APPLY_PROCEDURES":
        st.error("### 🔴 APPLY MAINTENANCE PROCEDURES")
        
        # Mostrar quais causas foram identificadas
        if decision['causes_above_threshold']:
            identified_causes = [CAUSES[c] for c in decision['causes_above_threshold']]
            st.write(f"**Critical causes identified:** {', '.join(identified_causes)}")
        
        if decision['procedures'] is not None and len(decision['procedures']) > 0:
            
            # Sumário
            st.subheader("📊 Maintenance Summary")
            
            total_effort = decision['procedures']['effort_h'].sum()
            total_cost = decision['procedures']['spare_parts_cost_eur'].sum()
            max_risk = decision['procedures']['risk_rating'].max()
            avg_risk = decision['procedures']['risk_rating'].mean()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Procedures", len(decision['procedures']))
            col2.metric("Total Effort", f"{total_effort:.1f}h")
            col3.metric("Total Cost", f"€{total_cost}")
            col4.metric("Max Risk", f"{max_risk}/5")
            col5.metric("Avg Risk", f"{avg_risk:.1f}/5")
            
            st.markdown("---")
            
            # Procedimentos individuais
            st.subheader("🔧 Detailed Procedures")

            for idx, (_, proc) in enumerate(decision['procedures'].iterrows(), 1):
                # Usar container com borda em vez de expander
                with st.container():
                    st.markdown(f"#### {idx}. {proc['name']} → {proc['targets_component']}")
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown(f"""
                        - **Mitigates**: {proc['mitigates_cause']}
                        - **Target Component**: {proc['targets_component']}
                        """)
                    
                    with col2:
                        subcol1, subcol2, subcol3 = st.columns(3)
                        subcol1.metric("Time", f"{proc['effort_h']}h")
                        subcol2.metric("Cost", f"€{proc['spare_parts_cost_eur']}")
                        subcol3.metric("Risk", f"{proc['risk_rating']}/5")
                    
                    # Separador entre procedimentos
                    if idx < len(decision['procedures']):
                        st.markdown("---")
            
        else:
            st.warning("⚠️ No procedures found in Knowledge Graph for the identified causes.")
            st.info("This might indicate an issue with the ontology or the cause names don't match.")

if __name__ == "__main__":
    main()
