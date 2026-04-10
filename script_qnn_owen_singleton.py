"""
Reproduce the QNN experiment from Section 4.2 / Figure 13 of:
  Heese et al., "Explaining Quantum Circuits with Shapley Values"

But run OWEN values instead of Shapley values, using a SINGLETON partition:
  each active gate is its own coalition.

For a singleton partition, Owen values should reduce to Shapley values.
This is therefore a practical validation script for the QuantumOwenValues code.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
QSHAPTOOLS_SRC = ROOT / "qshaptools" / "src"
sys.path.insert(0, str(QSHAPTOOLS_SRC))

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.utils import QuantumInstance
    try:
        from qiskit import Aer
    except Exception:
        from qiskit_aer import Aer
except Exception as e:
    raise ImportError(
        "Could not import Qiskit / Aer. Make sure qiskit and qiskit-aer are installed."
    ) from e

from qshaptools.qowen import QuantumOwenValues
from qshaptools.tools import build_circuit


# ---------------------------------------------------------------------------
# Local replacement for qshaptools.qvalues.value_callable
# ---------------------------------------------------------------------------

def value_callable(qc_data, num_qubits, S, quantum_instance, eval_fun, **kwargs):
    """
    Build the coalition circuit corresponding to S, then evaluate it with eval_fun.

    Parameters
    ----------
    qc_data : extracted circuit data
    num_qubits : int
    S : list[int]
        Active gate indices for this coalition. Locked gates are already included
        by QuantumOwenValues before this function is called.
    quantum_instance : QuantumInstance
    eval_fun : callable
        Signature: eval_fun(quantum_instance, qc, param_def_dict, **kwargs) -> float

    Returns
    -------
    float
    """
    qc, param_def_dict = build_circuit(
        qc_data=qc_data,
        num_qubits=num_qubits,
        S=S,
    )
    return eval_fun(
        quantum_instance=quantum_instance,
        qc=qc,
        param_def_dict=param_def_dict,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = ROOT / "data" / "qnn-data.csv"

THETA_TRAINED = [3.860, -1.070, -1.583, 0.860]

FEAT_PARAM_NAMES = ["feat_p0", "feat_p1", "feat_p2", "feat_p3", "feat_p4", "feat_p5"]
THETA_PARAM_NAMES = ["theta_0", "theta_1", "theta_2", "theta_3"]

# Paper remaining/locked gates R = {1,3} in 1-based indexing = {0,2} in Python indexing
LOCKED_INSTRUCTIONS = [0, 2]

OWEN_SAMPLE_FRAC = 0.01
OWEN_SAMPLE_REPS = 32
N_RUNS = 5


# ---------------------------------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------------------------------

def load_dataset():
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df[["x0", "x1"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# 2) Build the QNN circuit from Figure 12
# ---------------------------------------------------------------------------

def build_qnn_circuit():
    qc = QuantumCircuit(2, 2)

    feat_params = [Parameter(name) for name in FEAT_PARAM_NAMES]
    theta_params = [Parameter(name) for name in THETA_PARAM_NAMES]

    fp = feat_params
    tp = theta_params

    # First feature-map block
    qc.h(0)          # 0 (paper 1)
    qc.p(fp[0], 0)   # 1 (paper 2)
    qc.h(1)          # 2 (paper 3)
    qc.p(fp[1], 1)   # 3 (paper 4)
    qc.cx(0, 1)      # 4 (paper 5)
    qc.p(fp[2], 1)   # 5 (paper 6)
    qc.cx(0, 1)      # 6 (paper 7)

    # Second feature-map block
    qc.h(0)          # 7 (paper 8)
    qc.p(fp[3], 0)   # 8 (paper 9)
    qc.h(1)          # 9 (paper 10)
    qc.p(fp[4], 1)   # 10 (paper 11)
    qc.cx(0, 1)      # 11 (paper 12)
    qc.p(fp[5], 1)   # 12 (paper 13)
    qc.cx(0, 1)      # 13 (paper 14)

    # Trainable layer
    qc.ry(tp[0], 0)  # 14 (paper 15)
    qc.ry(tp[1], 1)  # 15 (paper 16)
    qc.cx(0, 1)      # 16 (paper 17)
    qc.ry(tp[2], 0)  # 17 (paper 18)
    qc.ry(tp[3], 1)  # 18 (paper 19)

    return qc


# ---------------------------------------------------------------------------
# 3) Value function: one-shot accuracy without retraining
# ---------------------------------------------------------------------------

def qnn_accuracy_eval_fun(quantum_instance, qc, param_def_dict, X, y, theta_trained):
    circuits = []

    for row_idx in range(len(y)):
        x0, x1 = X[row_idx]

        phi1 = 2.0 * x0
        phi2 = 2.0 * x1
        phi_cross = 2.0 * (np.pi - x0) * (np.pi - x1)

        name_to_value = {
            "feat_p0": phi1,
            "feat_p1": phi2,
            "feat_p2": phi_cross,
            "feat_p3": phi1,
            "feat_p4": phi2,
            "feat_p5": phi_cross,
            "theta_0": theta_trained[0],
            "theta_1": theta_trained[1],
            "theta_2": theta_trained[2],
            "theta_3": theta_trained[3],
        }

        bindings = {}
        for param in param_def_dict.keys():
            if param.name in name_to_value:
                bindings[param] = name_to_value[param.name]

        bound_qc = qc.copy().assign_parameters(bindings)
        bound_qc.measure(range(bound_qc.num_qubits), range(bound_qc.num_qubits))
        circuits.append(bound_qc)

    result = quantum_instance.execute(circuits)

    correct = 0
    for row_idx, circ in enumerate(circuits):
        counts = result.get_counts(circ)
        bitstring = next(iter(counts.keys()))
        q0_bit = int(bitstring[-1])   # qubit 0 is rightmost bit
        y_pred = q0_bit
        if y_pred == int(y[row_idx]):
            correct += 1

    return correct / len(y)


# ---------------------------------------------------------------------------
# 4) Sanity check: full-circuit one-shot accuracy
# ---------------------------------------------------------------------------

def sanity_check_full_circuit(qc, X, y, n_trials=50):
    backend = Aer.get_backend("qasm_simulator")
    qi = QuantumInstance(backend=backend, shots=1)

    param_def_dict = {p: None for p in qc.parameters}
    accs = []
    for _ in range(n_trials):
        acc = qnn_accuracy_eval_fun(qi, qc, param_def_dict, X, y, THETA_TRAINED)
        accs.append(acc)

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    print(f"Sanity check (full circuit): accuracy = {mean_acc:.3f} ± {std_acc:.3f} over {n_trials} trials")
    print("Paper target is about 0.80 for the trained QNN.")
    return mean_acc, std_acc


# ---------------------------------------------------------------------------
# 5) Build singleton partition over active gates
# ---------------------------------------------------------------------------

def build_singleton_partition(qc, locked_instructions):
    unlocked = [idx for idx in range(len(qc.data)) if idx not in locked_instructions]
    partition = [[idx] for idx in unlocked]
    return unlocked, partition


# ---------------------------------------------------------------------------
# 6) Run one Owen experiment
# ---------------------------------------------------------------------------

def run_owen_once(qc, X, y, seed, silent=False):
    backend = Aer.get_backend("qasm_simulator")
    qi = QuantumInstance(backend=backend, shots=1)

    unlocked, singleton_partition = build_singleton_partition(qc, LOCKED_INSTRUCTIONS)

    qov = QuantumOwenValues(
        qc=qc,
        partition=singleton_partition,
        value_fun=value_callable,
        value_kwargs_dict=dict(
            eval_fun=qnn_accuracy_eval_fun,
            X=X,
            y=y,
            theta_trained=THETA_TRAINED,
        ),
        quantum_instance=qi,
        locked_instructions=LOCKED_INSTRUCTIONS,
        owen_sample_frac=OWEN_SAMPLE_FRAC,
        owen_sample_reps=OWEN_SAMPLE_REPS,
        evaluate_value_only_once=False,
        owen_sample_seed=seed,
        name="qnn_singleton_owen",
        silent=silent,
    )

    print(qov)
    phi_dict = qov.run()
    return phi_dict, singleton_partition


# ---------------------------------------------------------------------------
# 7) Aggregate results and plot
# ---------------------------------------------------------------------------

def aggregate_phi_dicts(all_phi_dicts):
    gate_indices = sorted(all_phi_dicts[0].keys())
    means = {}
    stds = {}
    for g in gate_indices:
        vals = [d[g] for d in all_phi_dicts]
        means[g] = float(np.mean(vals))
        stds[g] = float(np.std(vals))
    return means, stds


def plot_results(means, stds):
    gate_indices = sorted(means.keys())
    paper_indices = [g + 1 for g in gate_indices]

    gate_name_map = {
        1: "P", 3: "P", 4: "CX", 5: "P", 6: "CX",
        7: "H", 8: "P", 9: "H", 10: "P", 11: "CX",
        12: "P", 13: "CX", 14: "RY", 15: "RY", 16: "CX",
        17: "RY", 18: "RY",
    }
    labels = [gate_name_map[g] for g in gate_indices]

    y_mean = [means[g] for g in gate_indices]
    y_std = [stds[g] for g in gate_indices]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(paper_indices, y_mean, yerr=y_std, fmt="o", capsize=3, markersize=5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("gate index g (paper numbering)")
    ax.set_ylabel(r"$\mathrm{Ow}_{(g)}$")
    ax.set_title("QNN Owen values with singleton partition (should mirror Shapley)")
    ax.set_xticks(paper_indices)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(paper_indices)
    ax_top.set_xticklabels(labels, fontsize=8)

    fig.tight_layout()
    out_path = ROOT / "qnn_owen_singleton_reproduction.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to: {out_path}")
    plt.show()


def print_results_table(means, stds):
    gate_indices = sorted(means.keys())
    gate_name_map = {
        1: "P", 3: "P", 4: "CX", 5: "P", 6: "CX",
        7: "H", 8: "P", 9: "H", 10: "P", 11: "CX",
        12: "P", 13: "CX", 14: "RY", 15: "RY", 16: "CX",
        17: "RY", 18: "RY",
    }

    print("\nResults (paper gate index, gate name, mean ± std):")
    for g in gate_indices:
        print(f"g={g+1:2d}  {gate_name_map[g]:3s}   Ow={means[g]:+.6f} ± {stds[g]:.6f}")


# ---------------------------------------------------------------------------
# 8) Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Loaded {len(y)} samples.")

    print("\nBuilding Figure 12 QNN circuit...")
    qc = build_qnn_circuit()
    print(f"Circuit has {len(qc.data)} instructions and {qc.num_qubits} qubits.\n")

    print("Instruction order check (must match paper Figure 12):")
    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        try:
            qubits = [qc.find_bit(q).index for q in qargs]
        except Exception:
            qubits = [getattr(q, "index", "?") for q in qargs]
        print(f"  python idx={idx:2d} | paper gate={idx+1:2d} | {instr.name:4s} | qubits={qubits} | params={instr.params}")

    unlocked, singleton_partition = build_singleton_partition(qc, LOCKED_INSTRUCTIONS)
    print("\nSingleton partition over active gates:")
    print(singleton_partition)

    print("\nRunning sanity check on full circuit...")
    sanity_check_full_circuit(qc, X, y, n_trials=50)

    print("\nRunning 5 independent Owen runs (singleton partition)...")
    all_phi_dicts = []
    for run_idx in range(N_RUNS):
        print(f"\n--- Run {run_idx+1}/{N_RUNS} | owen_sample_seed={run_idx} ---")
        phi_dict, _ = run_owen_once(qc, X, y, seed=run_idx, silent=False)
        all_phi_dicts.append(phi_dict)
        print(f"phi_dict = {phi_dict}")

    means, stds = aggregate_phi_dicts(all_phi_dicts)
    print_results_table(means, stds)
    plot_results(means, stds)

    print("\nExpectation:")
    print("  Because the partition is singleton, these Owen values should be very close")
    print("  to the Shapley values from the original QNN reproduction script.")


if __name__ == "__main__":
    main()