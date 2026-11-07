"""
Train a small deep neural network (DNN) to separate a new-physics SIGNAL from BACKGROUND events.

Signal:      NMSSM  X(600 GeV) -> Y(150 GeV) H(125 GeV),  with  Y -> b b  and  H -> photon photon
Backgrounds: GGJets      (two real photons + jets)
             DDQCDGJets  (photon + jet / multijet events with fake photons, estimated from data)

Every row of the input files is one collision event that already passed a basic selection
(2 photons + 2 jets). The DNN looks at a few numbers per event (the "features") and returns a
score between 0 (looks like background) and 1 (looks like signal).

Run it with:
    python dnn_signal_background.py /Users/yush/ML/data
(the argument is the folder with the .parquet files; default: the current folder)

All plots go to the folder "plots/". See README.md for an explanation of every step.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")                      # make plot files without opening windows
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# =====================================================================================
# 1. SETTINGS  (try changing these!)
# =====================================================================================
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

SIGNAL_FILE = "NMSSM_X600_Y150.parquet"
BACKGROUND_FILES = {"GGJets": "GGJets_MGG-80_Rescaled.parquet",
                    "DDQCDGJets": "DDQCDGJets_Rescaled.parquet"}

# Numbers the network is allowed to look at (column name: short explanation).
# The photon pair mass ("mass") is NOT used: we keep it to count signal events under the 125 GeV peak.
FEATURES = {
    "Res_pholead_PtOverM":        "leading photon pT / m(photon photon)",
    "Res_phosublead_PtOverM":     "second photon pT / m(photon photon)",
    "lead_eta":                   "leading photon direction (eta)",
    "sublead_eta":                "second photon direction (eta)",
    "lead_mvaID":                 "leading photon quality (ID score)",
    "sublead_mvaID":              "second photon quality (ID score)",
    "Res_dijet_mass":             "mass of the two b jets [GeV]",
    "Res_dijet_pt":               "pT of the two b jets [GeV]",
    "Res_lead_bjet_btagPNetB":    "how b-like the leading jet is (b-tag score)",
    "Res_sublead_bjet_btagPNetB": "how b-like the second jet is (b-tag score)",
    "Res_M_X":                    "mass of photons + b jets, m(X) [GeV]",
    "Res_DeltaR_jg_min":          "smallest angle between a jet and a photon",
    "Res_CosThetaStar_gg":        "photon decay angle |cos theta*|",
    "n_jets":                     "number of jets",
    "puppiMET_pt":                "missing transverse momentum [GeV]",
}

# Scale factors for the event weights, to turn them into expected numbers of events.
# !! Ask your supervisor for the right values: the weights in the three files are not on the same scale. !!
WEIGHT_SCALE = {"signal": 1.0, "GGJets": 1.0, "DDQCDGJets": 1.0}

# How the background used for TRAINING is made of the two samples (fractions add up to 1)
TRAIN_BKG_FRACTION = {"GGJets": 0.5, "DDQCDGJets": 0.5}

MAX_EVENTS_PER_SAMPLE = 300_000     # read at most this many events per file (keeps the memory small)
MGG_RANGE = (100.0, 180.0)          # keep events with 100 < m(photon photon) < 180 GeV
SIGNAL_WINDOW = (120.0, 130.0)      # "signal region" in m(photon photon) used to count events
HIDDEN_LAYERS = (64, 64, 32)        # size of the hidden layers of the network
MAX_EPOCHS = 60                     # maximum number of passes over the training data
SEED = 42                           # random seed: same seed -> same result

PLOT_DIR = "plots"
rng = np.random.default_rng(SEED)


# =====================================================================================
# 2. READ THE DATA
# =====================================================================================
def read_sample(filename, name, label):
    """Read one .parquet file: only the columns we need, at most MAX_EVENTS_PER_SAMPLE events."""
    path = os.path.join(DATA_DIR, filename)
    f = pq.ParquetFile(path)
    n_total = f.metadata.num_rows
    keep = min(1.0, MAX_EVENTS_PER_SAMPLE / n_total)          # fraction of events we keep
    columns = list(FEATURES) + ["mass", "weight"]
    pieces = []
    for batch in f.iter_batches(batch_size=100_000, columns=columns):
        df = batch.to_pandas()
        if keep < 1.0:
            df = df[rng.random(len(df)) < keep]                # keep a random subset
        pieces.append(df)
    df = pd.concat(pieces, ignore_index=True)
    df["weight"] = df["weight"] / keep * WEIGHT_SCALE["signal" if label == 1 else name]
    df["sample"] = name
    df["label"] = label
    print(f"  {name:12s}: {n_total:>9,d} events in the file, {len(df):>9,d} read")
    return df


print("Reading the data ...")
t0 = time.time()
frames = [read_sample(SIGNAL_FILE, "signal", 1)]
for name, fn in BACKGROUND_FILES.items():
    frames.append(read_sample(fn, name, 0))
data = pd.concat(frames, ignore_index=True)

# Preselection: the event must have a b-jet pair (missing values are stored as -999)
# and a photon-pair mass inside MGG_RANGE.
good = (data["Res_dijet_mass"] > 0) & (data["mass"] > MGG_RANGE[0]) & (data["mass"] < MGG_RANGE[1])
for c in FEATURES:
    good &= data[c] > -900
data = data[good].reset_index(drop=True)
print(f"after the preselection: {len(data):,d} events  ({time.time() - t0:.0f} s)")
for name in ["signal"] + list(BACKGROUND_FILES):
    d = data[data["sample"] == name]
    print(f"  {name:12s}: {len(d):>9,d} events, sum of weights = {d['weight'].sum():.4g}")

os.makedirs(PLOT_DIR, exist_ok=True)


# =====================================================================================
# 3. LOOK AT THE INPUT VARIABLES
# =====================================================================================
def plot_inputs():
    n = len(FEATURES)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    for ax, (col, text) in zip(axes.flat, FEATURES.items()):
        lo, hi = np.percentile(data[col], [0.5, 99.5])
        bins = np.linspace(lo, hi + 1e-9, 40)
        for name, color in [("signal", "red"), ("GGJets", "tab:blue"), ("DDQCDGJets", "tab:green")]:
            d = data[data["sample"] == name]
            ax.hist(d[col], bins=bins, weights=d["weight"], density=True, histtype="step", lw=1.5,
                    color=color, label=name)
        ax.set_title(text, fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Input variables (each sample normalized to area 1)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "1_inputs.png"), dpi=120)
    plt.close(fig)


plot_inputs()
print("wrote plots/1_inputs.png")


# =====================================================================================
# 4. SPLIT INTO TRAINING AND TEST EVENTS
# =====================================================================================
# Half of the events teach the network ("training"), the other half check it ("test").
# The test events are never shown to the network during training.
is_train = rng.random(len(data)) < 0.5
train, test = data[is_train], data[~is_train]
test = test.assign(weight=test["weight"] * 2)       # the test half stands for all events

# Make a BALANCED training set: as many background as signal events, picked at random with a
# probability proportional to their weight (so important events are picked more often).
def pick(df, n):
    p = df["weight"].clip(lower=0).to_numpy()
    idx = rng.choice(len(df), size=n, replace=True, p=p / p.sum())
    return df.iloc[idx]

n_sig = int((train["label"] == 1).sum())
parts = [pick(train[train["label"] == 1], n_sig)]
for name, frac in TRAIN_BKG_FRACTION.items():
    parts.append(pick(train[train["sample"] == name], int(frac * n_sig)))
train_balanced = pd.concat(parts).sample(frac=1.0, random_state=SEED)   # shuffle
print(f"training events: {len(train_balanced):,d} (half signal, half background); test events: {len(test):,d}")

# Neural networks learn best when every input has a similar size (mean 0, spread 1)
scaler = StandardScaler().fit(train_balanced[list(FEATURES)])
X_train = scaler.transform(train_balanced[list(FEATURES)])
y_train = train_balanced["label"].to_numpy()


# =====================================================================================
# 5. TRAIN THE NEURAL NETWORK
# =====================================================================================
dnn = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
                    learning_rate_init=1e-3, batch_size=512, max_iter=MAX_EPOCHS,
                    early_stopping=True, validation_fraction=0.2, n_iter_no_change=6,
                    random_state=SEED, verbose=True)
print("\nTraining the DNN ... (each 'Iteration' line is one epoch)")
t0 = time.time()
dnn.fit(X_train, y_train)
print(f"done in {time.time() - t0:.0f} s, {dnn.n_iter_} epochs")

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(dnn.loss_curve_, "b-", label="training loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss (lower = better)", color="b")
ax2 = ax1.twinx()
ax2.plot(dnn.validation_scores_, "r-", label="validation accuracy")
ax2.set_ylabel("accuracy on validation events", color="r")
fig.suptitle("Learning curve"); fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "2_learning_curve.png"), dpi=120); plt.close(fig)
print("wrote plots/2_learning_curve.png")


# =====================================================================================
# 6. HOW GOOD IS IT?  (score distributions, ROC curve)
# =====================================================================================
def score(df):
    return dnn.predict_proba(scaler.transform(df[list(FEATURES)]))[:, 1]

test = test.assign(score=score(test))
train_balanced = train_balanced.assign(score=score(train_balanced))

# (a) score distributions: training events (dots) vs test events (lines).
#     If they look very different, the network memorized the training events ("overtraining").
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(0, 1, 41)
for lab, color, text in [(1, "red", "signal"), (0, "tab:blue", "background")]:
    t = test[test["label"] == lab]
    ax.hist(t["score"], bins=bins, weights=t["weight"], density=True, histtype="step", lw=2, color=color,
            label=text + " (test)")
    tr = train_balanced[train_balanced["label"] == lab]
    h, _ = np.histogram(tr["score"], bins=bins, density=True)
    ax.plot(0.5 * (bins[1:] + bins[:-1]), h, "o", ms=3, color=color, label=text + " (training)")
ax.set_yscale("log"); ax.set_xlabel("DNN score"); ax.set_ylabel("normalized")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "3_dnn_score.png"), dpi=120); plt.close(fig)
print("wrote plots/3_dnn_score.png")

# (b) ROC curve: for every score cut, which fraction of signal and of background passes?
cuts = np.linspace(0, 1, 201)
sig, bkg = test[test["label"] == 1], test[test["label"] == 0]
eff_s = np.array([sig["weight"][sig["score"] > c].sum() for c in cuts]) / sig["weight"].sum()
eff_b = np.array([bkg["weight"][bkg["score"] > c].sum() for c in cuts]) / bkg["weight"].sum()
auc = np.trapezoid(eff_s[::-1], eff_b[::-1]) if hasattr(np, "trapezoid") else np.trapz(eff_s[::-1], eff_b[::-1])
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(eff_b, eff_s, "b-", label=f"DNN (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="random guess (AUC = 0.5)")
ax.set_xlabel("background efficiency (fraction of background kept)")
ax.set_ylabel("signal efficiency (fraction of signal kept)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "4_roc.png"), dpi=120); plt.close(fig)
print(f"wrote plots/4_roc.png   AUC = {auc:.3f}  (1.0 = perfect, 0.5 = no separation)")


# =====================================================================================
# 7. CHOOSE THE BEST CUT ON THE DNN SCORE
# =====================================================================================
# Count expected events in the signal window of m(photon photon) for every cut on the score,
# and compute the significance Z (how many "sigmas" the signal would stand out above the background).
win = (test["mass"] > SIGNAL_WINDOW[0]) & (test["mass"] < SIGNAL_WINDOW[1])
s_all = np.array([test["weight"][win & (test["label"] == 1) & (test["score"] > c)].sum() for c in cuts])
b_all = np.array([test["weight"][win & (test["label"] == 0) & (test["score"] > c)].sum() for c in cuts])
nb_raw = np.array([(win & (test["label"] == 0) & (test["score"] > c)).sum() for c in cuts])
with np.errstate(divide="ignore", invalid="ignore"):
    Z = np.sqrt(2 * ((s_all + b_all) * np.log(1 + s_all / b_all) - s_all))
Z[(nb_raw < 10) | ~np.isfinite(Z)] = 0       # ignore cuts that leave fewer than 10 simulated bkg events
best = int(np.argmax(Z))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(cuts, Z, "g-")
ax.axvline(cuts[best], color="k", ls="--", label=f"best cut: score > {cuts[best]:.3f}")
ax.set_xlabel("cut on DNN score"); ax.set_ylabel("significance Z")
ax.set_title(f"{SIGNAL_WINDOW[0]:g} < m(photon photon) < {SIGNAL_WINDOW[1]:g} GeV")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "5_significance.png"), dpi=120); plt.close(fig)
print("wrote plots/5_significance.png")

print("\n================ RESULT ================")
print(f"signal window: {SIGNAL_WINDOW[0]:g} < m(photon photon) < {SIGNAL_WINDOW[1]:g} GeV")
print(f"without the DNN : signal = {s_all[0]:.4g}, background = {b_all[0]:.4g}")
print(f"best DNN cut    : score > {cuts[best]:.3f}")
print(f"                  signal = {s_all[best]:.4g} ({100 * s_all[best] / s_all[0]:.1f}% kept), "
      f"background = {b_all[best]:.4g} ({100 * b_all[best] / b_all[0]:.2f}% kept)")
print(f"                  significance Z = {Z[best]:.3f}   (Z without the DNN = {Z[0]:.3f})")
print("(numbers of events are only meaningful once WEIGHT_SCALE is set correctly, see README)")
