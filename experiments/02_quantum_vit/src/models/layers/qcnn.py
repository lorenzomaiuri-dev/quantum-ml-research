import torch
import torch.nn as nn
import pennylane as qml

from quantum_framework.layers import PinnedTorchLayer, simulator_device


class QuantumPatchEmbedding(nn.Module):
    """Quantum patch embedding: Linear compression → tanh·π → VQC → ⟨Z⟩."""

    def __init__(self, patch_dim, embed_dim, n_qlayers=2, q_device="default.qubit"):
        super().__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.n_qubits = embed_dim

        # Classical adapter: compresses patch_dim → n_qubits before angle embedding.
        self.classical_pre_process = nn.Linear(patch_dim, self.n_qubits)

        dev = qml.device(q_device, wires=self.n_qubits)

        # Lightning's CPU and GPU statevector devices expose exact adjoint
        # differentiation rather than backpropagation through their internal
        # statevector implementations.
        diff_method = (
            "adjoint"
            if q_device in {"lightning.qubit", "lightning.gpu"}
            else "backprop"
        )

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        weight_shapes = {"weights": (n_qlayers, self.n_qubits, 3)}
        # Pin circuit tensors to the simulator-owned device (CPU for the standard
        # exact simulators, CUDA for lightning.gpu).
        self.q_layer = PinnedTorchLayer(
            qnode, weight_shapes, simulator_device(q_device)
        )

    def forward(self, x):
        # x: (B, n_patches, patch_dim)
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        x_scaled = torch.tanh(self.classical_pre_process(x_flat)) * torch.pi
        return self.q_layer(x_scaled).view(B, N, self.embed_dim)
