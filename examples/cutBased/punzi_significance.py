"""
Punzi significance for one fixed set of cuts
=============================================

Signal:      NMSSM X(600 GeV) -> Y(150 GeV) H,  with  H -> two photons  and  Y -> two b quarks
Background:  QCD photon + jets (data-driven estimate)

We keep only the events that pass all five cuts below, count how much signal
and background survive, and compute the Punzi figure of merit:

                         signal efficiency
    Punzi  =  -----------------------------------
               a/2  +  sqrt(number of background)

    signal efficiency = (signal left after cuts) / (signal before cuts)
    a = 5  (we want to be able to see a 5-sigma discovery)

A bigger Punzi value means a better set of cuts.
Every event has a "weight": the number of real events it stands for.
So we count events by adding up their weights.

Run with:   python3 punzi_significance.py
Needs:      pip install pandas pyarrow
"""
import math
import os
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Where the files are
# ---------------------------------------------------------------------------
folder = os.path.dirname(os.path.abspath(__file__))     # the folder this script is in
signal_file = os.path.join(folder, "NMSSM_X600_Y150.parquet")
background_file = os.path.join(folder, "DDQCDGJets_Rescaled.parquet")

# ---------------------------------------------------------------------------
# 2. The five input variables and the fixed cuts
#    (try changing the numbers and see how the Punzi value changes!)
# ---------------------------------------------------------------------------
cuts = {
    # variable name                 (lowest allowed, highest allowed)
    "mass":                       (120.0, 130.0),   # mass of the two photons [GeV]; the Higgs is at 125
    "Res_dijet_mass":             (110.0, 190.0),   # mass of the two b jets [GeV]; the Y particle is at 150
    "Res_M_X":                    (500.0, 650.0),   # mass of photons + jets [GeV]; the X particle is at 600
    "Res_lead_bjet_btagPNetB":    (0.50, 1.00),     # how much the 1st jet looks like a b quark (0 to 1)
    "Res_sublead_bjet_btagPNetB": (0.50, 1.00),     # how much the 2nd jet looks like a b quark (0 to 1)
}

a = 5.0   # number of sigmas in the Punzi formula

# ---------------------------------------------------------------------------
# 3. Read only the columns we need (the files have more than 1500 columns!)
# ---------------------------------------------------------------------------
columns = list(cuts.keys()) + ["weight"]
signal = pd.read_parquet(signal_file, columns=columns)
background = pd.read_parquet(background_file, columns=columns)
print("Signal events in the file:     ", len(signal))
print("Background events in the file: ", len(background))
print()


# ---------------------------------------------------------------------------
# 4. Apply the cuts: keep an event only if every variable is inside its range
# ---------------------------------------------------------------------------
def apply_cuts(events):
    keep = pd.Series(True, index=events.index)          # start by keeping every event
    for variable, (low, high) in cuts.items():
        keep = keep & (events[variable] >= low) & (events[variable] <= high)
    return events[keep]
# (Events where the jets were not found have the value -999, so they fail the cuts automatically.)


signal_after = apply_cuts(signal)
background_after = apply_cuts(background)

# ---------------------------------------------------------------------------
# 5. Count events (sum of weights) before and after the cuts
# ---------------------------------------------------------------------------
S_before = signal["weight"].sum()
S_after = signal_after["weight"].sum()
B_before = background["weight"].sum()
B_after = background_after["weight"].sum()

efficiency = S_after / S_before
punzi = efficiency / (a / 2 + math.sqrt(B_after))

# ---------------------------------------------------------------------------
# 6. Print the results
# ---------------------------------------------------------------------------
print("Cuts used:")
for variable, (low, high) in cuts.items():
    print("   %8.2f <= %-27s <= %8.2f" % (low, variable, high))
print()
print("                       before cuts     after cuts")
print("Signal (weighted)      %12.4g   %12.4g" % (S_before, S_after))
print("Background (weighted)  %12.4g   %12.4g" % (B_before, B_after))
print("Signal events left (unweighted):     ", len(signal_after))
print("Background events left (unweighted): ", len(background_after))
print()
print("Signal efficiency          = %.3f  (%.1f %% of the signal is kept)" % (efficiency, 100 * efficiency))
print("Background kept            = %.2e  (fraction of the background)" % (B_after / B_before))
print("Punzi significance (a = %g) = %.4f" % (a, punzi))
