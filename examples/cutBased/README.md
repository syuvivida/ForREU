# Finding a New Particle with Cuts: the Punzi Significance

This is a short Python exercise for high school students. You will use simulated data from the
CMS experiment at the Large Hadron Collider (LHC) to search for a new heavy particle, choose
**cuts** that keep the interesting events and throw away the boring ones, and measure how good
your cuts are with a single number: the **Punzi significance**.

## The physics

Some theories beyond the Standard Model (for example the *NMSSM*) predict new particles.
In this exercise we look for a heavy particle **X** with a mass of **600 GeV** that decays into
the Higgs boson **H** (125 GeV) and another new particle **Y** (150 GeV):

```
X (600 GeV)  →  H (125 GeV)  +  Y (150 GeV)
                │                 │
                └→ γ γ            └→ b b̄
               (two photons)     (two b-quark jets)
```

So every **signal** event has two photons and two b-quark jets. Unfortunately, ordinary
collisions that produce photons together with ordinary quark or gluon jets
(**background**, "QCD γ + jets") look similar and happen about a million times more often.
Our job is to select the events that look like the signal.

## The data

| File | What it is |
|---|---|
| `NMSSM_X600_Y150.parquet` | simulated signal events (X = 600 GeV, Y = 150 GeV) |
| `DDQCDGJets_Rescaled.parquet` | background events (QCD γ + jets, estimated from data) |

Each row is one collision event. Each event also has a **weight**: the number of real
events that this simulated event stands for. To count events we add up the weights.

> The data files are large (hundreds of MB) and are **not stored in this repository**.
> Ask your teacher where to download them, and put them in the same folder as the script.

## The five variables we cut on

| Variable | Meaning | Signal is near... |
|---|---|---|
| `mass` | mass of the two photons | 125 GeV (the Higgs) |
| `Res_dijet_mass` | mass of the two b-quark jets | 150 GeV (the Y) |
| `Res_M_X` | mass of the two photons and the two jets together | 600 GeV (the X) |
| `Res_lead_bjet_btagPNetB` | how much the first jet looks like a b quark (0 = no, 1 = yes) | close to 1 |
| `Res_sublead_bjet_btagPNetB` | how much the second jet looks like a b quark | close to 1 |

If the jets could not be found in an event, the jet variables are set to `-999`, so such events
never pass the cuts.

## The Punzi significance

After the cuts we count:

- **ε** (signal efficiency) = signal left after cuts ÷ signal before cuts
- **B** = background left after cuts

The Punzi significance is

$$
\text{Punzi} = \frac{\varepsilon}{\dfrac{a}{2} + \sqrt{B}}
$$

with **a = 5**, because physicists need a "5-sigma" signal to claim a discovery.
The formula rewards keeping a lot of signal (large ε) and punishes leftover background
(large B). **A bigger Punzi value means better cuts.** It was proposed by Giovanni Punzi in
2003 ([arXiv:physics/0308063](https://arxiv.org/abs/physics/0308063)) and is useful because it
does not depend on how big the signal really is, which we don't know in a search.

## How to run

You need Python 3 with `pandas` and `pyarrow`:

```bash
pip install pandas pyarrow
python3 punzi_significance.py
```

Example output with the cuts in the script:

```
                       before cuts     after cuts
Signal (weighted)            0.4553         0.1492
Background (weighted)     5.923e+05          3.064
Signal events left (unweighted):      15101
Background events left (unweighted):  2

Signal efficiency          = 0.328  (32.8 % of the signal is kept)
Background kept            = 5.17e-06  (fraction of the background)
Punzi significance (a = 5) = 0.0771
```

The cuts kept a third of the signal but only 5 out of every million background events!

## Try it yourself

All cuts are in one place near the top of `punzi_significance.py`:

```python
cuts = {
    "mass":                       (120.0, 130.0),
    "Res_dijet_mass":             (110.0, 190.0),
    "Res_M_X":                    (500.0, 650.0),
    "Res_lead_bjet_btagPNetB":    (0.50, 1.00),
    "Res_sublead_bjet_btagPNetB": (0.50, 1.00),
}
```

1. Make the photon mass window narrower (for example 123–127 GeV). Does the Punzi value go up or down? Why?
2. Remove one cut at a time (set its range very wide, e.g. `(-1000, 10000)`). Which variable is the most important?
3. Raise the b-tag cuts to 0.9. What happens to the signal efficiency and to the background?
4. Look at the **unweighted** number of background events left. With only a few events, how much can you trust the result?
5. Try `a = 3` instead of `a = 5`. Does the best choice of cuts change?

**Next step:** instead of guessing, write a loop that tries many cut values and keeps the one
with the largest Punzi significance (a "cut scan").

## Files

| File | Description |
|---|---|
| `punzi_significance.py` | the script: reads the data, applies fixed cuts, prints the Punzi significance |
| `README.md` | this file |
