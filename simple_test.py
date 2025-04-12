#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prueba simple para validar el uso de semillas aleatorias.
"""

import random
import numpy as np

def test_random_seed():
    """Prueba la reproducibilidad usando semillas aleatorias"""
    # Establecer semilla
    random.seed(42)
    np.random.seed(42)
    
    # Generar números aleatorios
    random_numbers1 = [random.randint(1, 100) for _ in range(5)]
    np_random_numbers1 = np.random.randint(1, 100, 5)
    
    print("Primera ejecución:")
    print(f"Random: {random_numbers1}")
    print(f"NumPy: {np_random_numbers1}")
    
    # Volver a establecer la misma semilla
    random.seed(42)
    np.random.seed(42)
    
    # Generar números aleatorios nuevamente
    random_numbers2 = [random.randint(1, 100) for _ in range(5)]
    np_random_numbers2 = np.random.randint(1, 100, 5)
    
    print("\nSegunda ejecución (con misma semilla):")
    print(f"Random: {random_numbers2}")
    print(f"NumPy: {np_random_numbers2}")
    
    # Comprobar reproducibilidad
    print("\nComprobando reproducibilidad...")
    random_equal = random_numbers1 == random_numbers2
    np_equal = np.array_equal(np_random_numbers1, np_random_numbers2)
    
    print(f"Random reproducible: {random_equal}")
    print(f"NumPy reproducible: {np_equal}")
    
    # Probar con una semilla diferente
    random.seed(100)
    np.random.seed(100)
    
    # Generar números aleatorios con semilla diferente
    random_numbers3 = [random.randint(1, 100) for _ in range(5)]
    np_random_numbers3 = np.random.randint(1, 100, 5)
    
    print("\nTercera ejecución (semilla diferente):")
    print(f"Random: {random_numbers3}")
    print(f"NumPy: {np_random_numbers3}")
    
    # Comprobar que son diferentes
    random_different = random_numbers1 != random_numbers3
    np_different = not np.array_equal(np_random_numbers1, np_random_numbers3)
    
    print(f"Random diferente: {random_different}")
    print(f"NumPy diferente: {np_different}")

if __name__ == "__main__":
    print("Prueba de reproducibilidad con semillas aleatorias")
    print("="*50)
    test_random_seed()
    print("="*50)
    print("Prueba completada.") 