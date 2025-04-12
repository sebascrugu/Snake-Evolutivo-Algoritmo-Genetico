#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script auxiliar para ejecutar los experimentos del juego Snake.
Este script sirve como punto de entrada para ejecutar los experimentos directamente.
"""

import os
import sys

# Asegurar que el directorio actual está en el path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar directamente el módulo de experimentos
try:
    from snake_experiments import run_experiments
    
    print("\n" + "="*50)
    print("EJECUTANDO EXPERIMENTOS DE SNAKE")
    print("="*50)
    
    # Ejecutar los experimentos
    run_experiments()
    
except ImportError as e:
    print(f"Error al importar el módulo de experimentos: {e}")
    print("Verifica que los archivos snake_experiments.py y snakepy.py están en el mismo directorio.") 