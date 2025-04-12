#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prueba para validar la función evaluate_agent_fitness.
"""

import time
import random
import numpy as np
from snakepy import SnakeGame, DecisionTable
from snake_experiments import DecisionTableAgent, evaluate_agent_fitness

def test_evaluate_agent_fitness():
    # Crear un agente para probar
    agent = DecisionTableAgent()
    
    # Establecer semilla para reproducibilidad
    base_seed = 42
    
    # Evaluar fitness con diferentes índices de agente
    for idx in range(3):
        # Evaluar fitness del agente
        fitness = evaluate_agent_fitness(agent, idx, base_seed)
        
        # Imprimir resultado
        print(f"Agente {idx} con semilla {base_seed}: Fitness = {fitness}")
        
        # Volver a evaluar con la misma semilla para verificar reproducibilidad
        fitness2 = evaluate_agent_fitness(agent, idx, base_seed)
        
        # Verificar que los resultados son iguales (reproducibilidad)
        if fitness == fitness2:
            print(f"✅ Reproducibilidad confirmada para agente {idx}")
        else:
            print(f"❌ Error: La reproducibilidad falló para agente {idx}")
            print(f"   Primera evaluación: {fitness}, Segunda evaluación: {fitness2}")

    # Probar con distintas semillas base
    print("\nProbando con diferentes semillas base:")
    for seed in [100, 200, 300]:
        fitness = evaluate_agent_fitness(agent, 0, seed)
        print(f"Semilla {seed}: Fitness = {fitness}")

if __name__ == "__main__":
    print("Iniciando prueba de evaluate_agent_fitness...")
    test_evaluate_agent_fitness()
    print("Prueba completada.") 