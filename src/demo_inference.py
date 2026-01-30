"""
Bayesian Network Inference Module
Integrates trained BN model for spindle overheat diagnosis
"""

import numpy as np
from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
from typing import Dict, Tuple
import streamlit as st


# ============================================================================
# DISCRETIZATION THRESHOLDS
# ============================================================================

THRESHOLDS = {
    'spindle_temp': {
        'low': 63.3175,
        'high': 72.5800
    },
    'vibration_rms': {
        'low': 0.8630,
        'high': 1.1110
    },
    'coolant_flow': {
        'low': 0.4280,
        'high': 0.5810
    }
}


# ============================================================================
# BAYESIAN NETWORK CLASS
# ============================================================================

class BayesianNetworkInference:
    """
    Wrapper para o modelo Bayesiano treinado.
    Carrega o modelo e executa inferência.
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o modelo Bayesiano.
        
        Args:
            model_path: Caminho para o ficheiro .bif do modelo
        """
        self.model_path = model_path
        self.model = None
        self.inference_engine = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo BN do ficheiro .bif"""
        try:
            reader = BIFReader(self.model_path)
            self.model = reader.get_model()
            self.inference_engine = VariableElimination(self.model)
            print(f"✅ BN Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading BN model: {e}")
            raise
    
    def discretize_value(self, value: float, variable: str) -> str:
        """
        Discretiza um valor contínuo numa categoria (low/normal/high).
        
        Args:
            value: Valor contínuo da variável
            variable: Nome da variável (spindle_temp, vibration_rms, coolant_flow)
            
        Returns:
            Categoria discretizada: 'low', 'normal', ou 'high'
        """
        if variable not in THRESHOLDS:
            raise ValueError(f"Unknown variable for discretization: {variable}")
        
        thresholds = THRESHOLDS[variable]
        
        if value < thresholds['low']:
            return 'low'
        elif value > thresholds['high']:
            return 'high'
        else:
            return 'normal'
    
    def predict(self, instance: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Executa inferência na rede Bayesiana.
        
        Args:
            instance: Dicionário com os valores das features do sensor
                     Exemplo: {'spindle_temp': 73.96, 'vibration_rms': 0.871, 
                              'coolant_flow': 1.05, ...}
        
        Returns:
            Tuple com:
            - overheat_prob: Probabilidade de overheat (float entre 0 e 1)
            - causes_prob: Dict com probabilidades de cada causa
                          {'K1': 0.45, 'K2': 0.23, 'K3': 0.67, 'K4': 0.12}
        """
        if self.inference_engine is None:
            raise RuntimeError("BN model not loaded!")
        
        # ====================================================================
        # STEP 1: Discretizar as evidências (valores contínuos → categorias)
        # ====================================================================
        evidence = {
            'spindle_temp': self.discretize_value(
                instance['spindle_temp'], 
                'spindle_temp'
            ),
            'vibration_rms': self.discretize_value(
                instance['vibration_rms'], 
                'vibration_rms'
            ),
            'coolant_flow': self.discretize_value(
                instance['coolant_flow'], 
                'coolant_flow'
            ),
            #'LowCoolingEfficiency': '0',
            #'FanFault': '0'
        }
        
        # ====================================================================
        # STEP 2: Inferir probabilidade de Overheat
        # ====================================================================
        overheat_result = self.inference_engine.query(
            variables=['spindle_overheat'],
            evidence=evidence
        )

        #print(overheat_result)

        # Probabilidade de spindle_overheat = 1
        overheat_prob = overheat_result.values[1]
        
        # ====================================================================
        # STEP 3: Inferir probabilidades das causas latentes
        # ====================================================================
        causes_mapping = {
            'K1': 'BearingWearHigh',
            'K2': 'FanFault',
            'K3': 'CloggedFilter',
            'K4': 'LowCoolingEfficiency'
        }
        
        causes_prob = {}
        
        for cause_id, cause_name in causes_mapping.items():
            cause_result = self.inference_engine.query(
                variables=[cause_name],
                evidence=evidence
            )
            # Probabilidade da causa = 1 (presente)
            causes_prob[cause_id] = cause_result.values[1]
        
        return overheat_prob, causes_prob
    
    def get_model_info(self) -> Dict:
        """
        Retorna informação sobre o modelo carregado.
        
        Returns:
            Dict com informações do modelo
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "nodes": list(self.model.nodes()),
            "edges": list(self.model.edges()),
            "num_nodes": len(self.model.nodes()),
            "num_edges": len(self.model.edges())
        }


# ============================================================================
# CACHED LOADER FOR STREAMLIT
# ============================================================================

@st.cache_resource
def load_bayesian_network(model_path: str = "bn_model_simple.bif"):
    """
    Carrega a Rede Bayesiana uma única vez e mantém em cache.
    
    Args:
        model_path: Caminho para o ficheiro .bif
        
    Returns:
        Instância de BayesianNetworkInference
    """
    try:
        bn = BayesianNetworkInference(model_path)
        return bn
    except Exception as e:
        st.error(f"❌ Error loading Bayesian Network: {e}")
        return None


# ============================================================================
# STANDALONE FUNCTION FOR PREDICTION (para usar na demo.py)
# ============================================================================

def predict_with_bayesian_network(
    bn_model: BayesianNetworkInference,
    instance: Dict
) -> Tuple[float, Dict[str, float]]:
    """
    Função wrapper para usar na demo.py
    
    Args:
        bn_model: Instância do modelo BN carregado
        instance: Dicionário com os valores das features
        
    Returns:
        Tuple com (probabilidade_overheat, dict_probabilidades_causas)
    """
    if bn_model is None:
        # Fallback para valores dummy se o modelo não estiver carregado
        import random
        st.warning("⚠️ BN model not loaded. Using random predictions.")
        
        overheat_prob = random.uniform(0.1, 0.9)
        causes_prob = {
            "K1": random.uniform(0.1, 0.8),
            "K2": random.uniform(0.1, 0.8),
            "K3": random.uniform(0.1, 0.8),
            "K4": random.uniform(0.1, 0.8)
        }
        return overheat_prob, causes_prob
    
    return bn_model.predict(instance)


# ============================================================================
# EXAMPLE USAGE (para testar standalone)
# ============================================================================

if __name__ == "__main__":
    # Exemplo de teste
    bn = BayesianNetworkInference("bn_model_simple.bif")
    
    # Mostrar info do modelo
    info = bn.get_model_info()
    print(f"\n📊 Model Info:")
    print(f"   Nodes: {info['num_nodes']}")
    print(f"   Edges: {info['num_edges']}")
    
    # Exemplo de predição
    test_instance = {
        'spindle_temp': 78.5,      # high
        'vibration_rms': 1.35,     # high
        'coolant_flow': 0.75,      # low
        'machine_id': 'M-A',
    }
    
    print(f"\n🔍 Test Instance:")
    print(f"   Spindle Temp: {test_instance['spindle_temp']}°C")
    print(f"   Vibration RMS: {test_instance['vibration_rms']}")
    print(f"   Coolant Flow: {test_instance['coolant_flow']}")
    
    overheat_prob, causes_prob = bn.predict(test_instance)
    
    print(f"\n📈 Predictions:")
    print(f"   Overheat Probability: {overheat_prob:.2%}")
    print(f"   Causes:")
    for cause_id, prob in causes_prob.items():
        print(f"      {cause_id}: {prob:.2%}")

    from pgmpy.factors.discrete import TabularCPD
    def print_cpds(model):
        backup = TabularCPD._truncate_strtable
        TabularCPD._truncate_strtable = lambda self, s: s

        try:
            for cpd in model.get_cpds():
                print("\n===================================")
                print(f"Variable: {cpd.variable}")
                print(cpd)
        finally:
            TabularCPD._truncate_strtable = backup
    
    #print_cpds(bn.model)