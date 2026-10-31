"""Render the committed exact-channel record; requires optional Matplotlib."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.experiment_artifacts import read_json


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = read_json(ROOT/"docs/restart/observable_noise_results.json.gz")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    def select(case, channel):
        return [r for r in data["rows"] if r["case"] == case and r["channel"] == channel]
    rows = select("tfim6", "dephasing")
    p = [r["strength"] for r in rows]
    ax = axes[0, 0]
    ax.plot(p, [r["fidelity_squared_pure_target"] for r in rows], "o-", label="Full-state fidelity")
    baseline = rows[0]["observables"]["Z0Z1"]["value"]
    ax.plot(p, [r["observables"]["Z0Z1"]["value"]/baseline for r in rows], "s-", label="ZZ correlation / ideal")
    ax.set(title="TFIM6: a fixed observable outlasts fidelity", ylabel="Fraction of ideal value", ylim=(-.02, 1.08))
    ax.legend()
    ax = axes[0, 1]
    ax.plot(p, [r["projections"]["diagonal"]["absolute_energy_drift"] for r in rows], "o-", label="Projected energy drift")
    ax.plot(p, [r["projections"]["diagonal"]["overlap_frobenius_drift"] for r in rows], "s-", label="Overlap drift (Frobenius)")
    ax.set(title="TFIM6: fixed overlap does not fix the energy", ylabel="Energy drift (J) / overlap drift")
    ax.legend()
    rows = select("h4", "dephasing")
    ax = axes[1, 0]
    ax.plot(p, [1000*r["observables"]["energy"]["absolute_error"] for r in rows], "o-", label="Direct energy error")
    ax.plot(p, [1000*r["projections"]["diagonal"]["absolute_energy_drift"] for r in rows], "s-", label="Projected energy drift")
    ax.axhline(1.6, color="black", ls=":", label="1.6 mHa reference scale")
    ax.set(title="H4: number is protected; chemical accuracy is not", ylabel="Energy error (mHa)")
    ax.legend(fontsize=9)
    rows = select("ghz8", "z_rotation")
    ax = axes[1, 1]
    ax.plot(p, [r["fidelity_squared_pure_target"] for r in rows], "o-", label="Full-state fidelity")
    ax.plot(p, [r["observables"]["Z0Z1"]["value"] for r in rows], "s-", label="ZZ correlation")
    ax.set(title="GHZ8 control: coherent phase sensitivity", ylabel="Fidelity / correlation", ylim=(-.02, 1.08))
    ax.legend()
    for j, ax in enumerate(axes.flat):
        ax.set_xlabel("Local Z rotation (radians)" if j == 3 else "Local phase-flip probability p")
        ax.grid(alpha=.2)
    fig.suptitle("Observable stability is task- and channel-dependent\n224 exact simulated outcomes; at most eight qubits", fontsize=14)
    path = ROOT/"docs/restart/observable_noise.png"
    fig.savefig(path, dpi=160, metadata={"Software": "clifford_qc"})
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
