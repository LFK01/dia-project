import numpy as np
from src.pricing.environment import *
from src.pricing.ts_learner import *
from src.pricing.reward_function import rewards


class ContextContainer:
    def __init__(self, user_class, context_probabilities, environment):
        self.context = user_class
        self.probabilities = context_probabilities
        self.ts_learner  # TSLearner for the context
        self.environment = environment

    # Metodo per il TS all'interno del context che ritorna il reward per il contesto corrente
    def run_TS(self):
        # trovare la best arm e in base alla best arm calcolare la somma pesata dei reward
        return

    # Metodo per splittare
    def split_context(self):
        return

    # Metodo per calcolare Hoeffding bounds
    def __compute_hoeffding_bounds(self):
        return
