"""
Thin re-export. The canonical implementation lives in quantum_framework.
Kept here so existing imports inside this experiment continue to work.

QuantumLayerAdapter is preserved as an alias for backwards compatibility
with the rest of experiment 01's model code.
"""

from quantum_framework.layers import QuantumLinearWithAdapter as QuantumLayerAdapter  # noqa: F401
