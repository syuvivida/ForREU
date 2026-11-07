# Finding new particles with a neural network

In this exercise you teach a computer to tell **signal** collisions (a hypothetical new particle) from
**background** collisions (ordinary physics that looks similar). Physicists at the Large Hadron Collider
(LHC) use the same method.

## The physics in one paragraph

At the LHC, protons collide 40 million times per second. We look for a heavy new particle **X** (600 GeV)
that decays into two lighter particles: **Y** (150 GeV) and the **Higgs boson H** (125 GeV).
- **Y** decays into two **b quarks**, which show up in the detector as two "b jets".
- **H** decays into two **photons** (γ).

So a signal event has **2 photons + 2 b jets**. Unfortunately, ordinary processes can also produce
photons and jets:
- **GGJets:** two real photons plus jets.
- **DDQCDGJets:** jets that are wrongly identified as photons.

These background events are far more common than the signal. The task: find the few signal events in a
huge pile of background.

## The data

| File | What it contains |
|---|---|
| `NMSSM_X600_Y150.parquet` | simulated **signal** events |
| `GGJets_MGG-80_Rescaled.parquet` | background: two photons + jets (simulation) |
| `DDQCDGJets_Rescaled.parquet` | background: fake photons (estimated from real data) |

Each row is one collision event, and each column is a measured quantity. There are more than 1,800
columns, but we use only 15 of them (listed in `FEATURES` at the top of the script).

Each event also has a **weight**. A weight tells you how many real events this one simulated event stands
for. For example, a weight of 2.5 means "this event counts as 2.5 events".

## What is a neural network (DNN)?

A neural network is a function with thousands of adjustable numbers (the "weights" of the network, not to
be confused with event weights). You show it many examples:
- "these 15 numbers came from a signal event → answer 1";
- "these came from a background event → answer 0".

After each example, it nudges its adjustable numbers to make its answer a bit better. This is
**training**. Afterwards, for a new event it gives a **score** between 0 and 1: close to 1 means "looks
like signal".

"Deep" means the network has several **hidden layers** between the input and the output. Ours has three,
with 64, 64 and 32 "neurons".

## Installation (once)

You need Python 3 and four packages:
```bash
python3 -m pip install numpy pandas pyarrow scikit-learn matplotlib
```

## Running

```bash
python3 dnn_signal_background.py /Users/yush/ML/data
```
The argument is the folder that contains the three `.parquet` files. It takes 1–3 minutes on a laptop.
The script prints what it is doing and writes five pictures into the folder `plots/`.

## What the script does, step by step

1. **Settings.** At the top of the script you can change everything: which variables are used, the
   network size, the number of epochs, and so on.
2. **Reading the data.** It reads the 15 variables, plus the photon-pair mass and the weight. To keep it
   fast, it reads at most 300,000 events per file. Then it keeps events that have two b jets and a
   photon-pair mass between 100 and 180 GeV.
3. **Plots of the inputs** (`plots/1_inputs.png`). Each variable is shown for signal (red) and for the two
   backgrounds. **Look for the variables where red differs most from the others:** those help the network
   the most. (Hint: the mass of the two b jets is close to 150 GeV for the signal, because Y has a mass of
   150 GeV.)
4. **Training and test events.** Half of the events are used to train. The other half (**test events**)
   are kept hidden and used afterwards to check the network fairly, like an exam with new questions.
5. **Training** (`plots/2_learning_curve.png`).
   - Each pass over the training events is one **epoch**.
   - The **loss** measures how wrong the network still is, so it should go down.
   - The red curve shows the **accuracy** on a separate set of events the network does not learn from.
   - Training stops by itself when the accuracy stops improving ("early stopping").
6. **How well does it work?**
   - **`plots/3_dnn_score.png`:** the score for signal (red) and background (blue). A good network pushes
     signal towards 1 and background towards 0. The dots (training events) and lines (test events) should
     agree. If they don't, the network has **memorized** its training events instead of learning general
     patterns: this is called **overtraining**.
   - **`plots/4_roc.png`:** the **ROC curve**. For every possible cut on the score, it shows which fraction
     of signal survives (up) against which fraction of background survives (right).
     - A perfect network goes straight up to the top-left corner.
     - The **AUC** (area under the curve) summarizes it in one number: 1.0 is perfect, 0.5 is guessing.
7. **The best cut** (`plots/5_significance.png`). We keep only events with a score above some value (a
   "cut"). Then we count signal (*s*) and background (*b*) events with a photon-pair mass between 120 and
   130 GeV, around the Higgs mass, and compute the **significance**

   $$Z = \sqrt{2\left[(s+b)\ln\left(1+\frac{s}{b}\right) - s\right]} \;\approx\; \frac{s}{\sqrt{b}} \quad (\text{when } s \ll b)$$

   *Z* says how many "standard deviations" the signal sticks out above the background:
   - *Z* = 3 is called "evidence";
   - *Z* = 5 is a "discovery".

   The script finds the cut with the largest *Z*. It ignores cuts that leave fewer than 10 simulated
   background events, because those numbers would be unreliable.

## Important: event weights

The weights in the three files are **not on the same scale**: the sums of the weights differ by large
factors that are not physical. Before you trust the event counts and *Z*, put the correct factors into
`WEIGHT_SCALE` at the top of the script; ask your supervisor for them. Until then:
- **Plots 1–4 are fine:** they compare *shapes* and *fractions*, so the overall scale does not matter.
- **The event numbers and *Z* in step 7 are not physical yet.**

Training uses a **balanced** set: equal numbers of signal and background events, with the two backgrounds
mixed 50/50 (`TRAIN_BKG_FRACTION`). So the training itself does not depend on the weight scale either.

## Things to try

1. **Remove the best variable.** Delete `"Res_dijet_mass"` from `FEATURES` and run again. How much does
   the AUC drop?
2. **Use fewer variables.** Can you find 3–4 variables that give almost the same AUC as all 15?
3. **Change the network.** Try `HIDDEN_LAYERS = (8,)` (tiny) and `(256, 256, 128, 64)` (big). Which is
   better? Does the big one show overtraining in plot 3?
4. **Fewer training events.** Set `MAX_EVENTS_PER_SAMPLE = 20_000`. What happens to the training and test
   curves?
5. **Why is the photon-pair mass not an input?** We use it to count events in the 120–130 GeV window.
   Think about what would happen to the background in that window if the network had already been "told"
   about the mass. (Physicists keep this variable separate so they can measure the background from the
   regions next to the peak.)
6. **Change the signal window** (`SIGNAL_WINDOW`) to (115, 135) or (123, 127). Does *Z* go up or down?
   Why?

## Words to know

| Word | Meaning |
|---|---|
| event | one proton–proton collision |
| feature | one number describing an event (input to the network) |
| label | the right answer: 1 = signal, 0 = background |
| epoch | one pass of the network over all training events |
| loss | how wrong the network is (training tries to make it small) |
| overtraining | the network memorizes the training events and does worse on new ones |
| ROC / AUC | how well signal and background are separated, for all possible cuts |
| significance *Z* | how strongly the signal would stand out above the background |
