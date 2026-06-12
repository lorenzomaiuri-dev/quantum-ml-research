# Quantum ML Research — Summary

## Research Question
Do Variational Quantum Circuits (VQC) offer measurable advantages over 
classical counterparts when integrated into Vision Transformers for image 
classification tasks?

## Experiments and Results

| # | Experiment | Hypothesis | Result |
|---|-----------|------------|--------|
| 1 | Quantum GPT | VQC layers improve language model perplexity | Falsified: comparable loss, 9× more parameters, 7× slower |
| 2 | Quantum ViT | VQC attention/embedding improves classification | Falsified: equivalent accuracy, significantly slower |
| 3 | Quantum Regularization | VQC impose implicit geometric regularization | Falsified: no statistically significant generalization gap reduction |
| 4 | Quantum Kernel | Quantum kernel improves class separability | Falsified: classical RBF kernel equivalent or better |

## Structural Limitation
All experiments run on classical quantum simulators. Demonstrating genuine 
quantum advantage on a classical simulator is theoretically impossible — 
the simulator computes quantum states in polynomial time, eliminating any 
potential exponential speedup. Results reflect expressivity and 
regularization properties only, not computational advantage.

## Methodology
- Rigorous baselines: all quantum models compared against iso-parametric 
  classical counterparts
- Multi-seed evaluation (5 seeds) for variance estimation
- Multi-dataset evaluation (3 MedMNIST datasets) for robustness
- Statistical significance testing (paired t-test)
- Full diagnostic logging (gradient norms, activation saturation)
