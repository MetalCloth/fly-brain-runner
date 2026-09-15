# Experiment log

This log records the project as it is built. Every training result should
include its configuration, seed, and evaluation method.

## 2026-09-14 — Milestone 0 setup

**Question:** Can we create a small, deterministic runner that exposes the
right control problem without needing a game engine or external dependencies?

**Decision:** Use a pure-Python symbolic environment first. The environment
has three lanes, five actions, three obstacle types, deterministic seeds, a
short action duration for jump/roll, rewards, collision termination, and a
maximum episode length.

**Why:** This keeps game logic separate from RL and vision. If a future agent
fails, we can inspect the exact observation and action instead of guessing from
pixels.

**Current status:** Verified.

**Verification:**

- `python3 tests.py` — 6 tests passed.
- `python3 demo.py` — seeded run completed 60 steps without collision and
  passed 6 obstacles.

**Finding:** The environment can produce a clean, deterministic learning
problem before we introduce neural networks. The hand-written policy can
choose lanes and time jump/roll actions, so the reward and collision rules
are behaving as intended for this first case.

**Next experiment:** Add Gymnasium spaces around the numeric observation and
run the official environment checks before PPO.

## 2026-09-14 — Numeric observations and baseline policies

**Question:** Can the readable game state be converted into a fixed-size input
without losing the information needed for control?

**Change:** Added a 17-value numeric encoder. It contains player-lane and
motion one-hot values, motion progress, normalized speed, and per-lane urgency
for each obstacle type.

**Verification:** Seven assert-based tests pass. The benchmark now runs 20
fixed-seed episodes for each simple policy:

```text
random     mean_steps=45.2   median_steps=22.0   mean_score=3.90   collision_rate=1.00
heuristic  mean_steps=300.0  median_steps=300.0  mean_score=36.00  collision_rate=0.00
```

**Finding:** The task has a clear gap between uninformed behavior and a policy
that understands obstacle timing. That gives PPO a useful target to beat.

**Debugging note:** The first benchmark run caught a reporting bug in the
summary calculation. It was fixed before recording the numbers above.

**Next experiment:** Add Gymnasium spaces around the numeric observation and
run the official environment checks before PPO.

## 2026-09-14 — Gymnasium boundary

**Question:** Can the runner expose the fixed numeric observation through the
standard interface that PPO libraries expect?

**Change:** Added `GymRunnerEnv`, a thin adapter around the tested core. It
declares five discrete actions and a 17-value bounded observation space. The
core game remains separate, so Gymnasium is only the interface layer.

**Dependency:** Created a local `.venv` and installed the tested
`gymnasium==1.3.0` dependency. The project’s `requirements.txt` records it.

**Verification:** `gym_check.py` passes Gymnasium’s official environment checks
and a 50-step random smoke test.

**Finding:** We now have a clean socket for PPO. The next meaningful change is
the learning algorithm, not more environment plumbing.

**Next experiment:** Train the MLP PPO baseline for a short, reproducible run
and compare it against the random and heuristic policies.

## 2026-09-14 — MLP PPO baseline

**Question:** Can a standard feed-forward policy learn the symbolic runner
without a hand-written rule?

**Change:** Added an MLP policy with PPO through Stable-Baselines3. The policy
uses two 64-unit hidden layers and the same 17-value numeric observation that
the heuristic receives. Training runs on CPU.

**Configuration:** Seed `7`, 20,000 requested timesteps (20,480 collected by
PPO’s 1,024-step rollout boundary), 300-step maximum episode, spawn interval
of 8 steps, learning rate `3e-4`, discount `0.99`, GAE lambda `0.95`, and
deterministic evaluation over 20 seeds.

**Result:**

```text
MLP PPO    mean_steps=300.0  median_steps=300.0  mean_score=36.00  collision_rate=0.00
```

The trained model is saved at `results/mlp_ppo/mlp_ppo.zip` and the training
monitor is saved at `results/mlp_ppo/monitor.csv`.

**Finding:** The environment and reward are learnable by ordinary PPO. We now
have a real baseline for judging whether the fly-inspired controller adds
anything beyond a generic MLP.

**Next experiment:** Replace only the MLP core with a small sparse recurrent
controller while keeping the environment, PPO loop, evaluation seeds, and
metrics unchanged.

## 2026-09-14 — Sparse recurrent fly-inspired policy

**Question:** Does a sparse, structured sensory pathway plus temporal memory
learn the runner under the same conditions as the ordinary MLP?

**Change:** Added a fixed sparse sensory projection with separate body/motion,
obstacle/looming, and integration pathways. RecurrentPPO supplies a 32-unit
LSTM memory. The action space, rewards, environment, training seed, and
evaluation protocol stay unchanged.

**Interpretation boundary:** This is fly-inspired architecture, not a real
connectome. The biological graph experiment comes after this controlled
comparison.

**Next experiment:** Run the sparse recurrent training and compare its learning
curve and fixed-seed result with `results/mlp_ppo/mlp_ppo.zip`.

## 2026-09-14 — Recurrent comparison and simplification

**Result:** A 20,480-step dense LSTM control and a 20,480-step sparse
recurrent run both stayed near random performance. The MLP baseline solved the
same task in the same budget.

**Interpretation:** This symbolic environment exposes exact obstacle distance,
so memory is not required yet. Recurrent PPO adds a harder optimization problem
before it adds useful information.

**Decision:** Keep the recurrent implementation for the later pixel/partial
observation stage. Test the sparse fly-inspired structure with ordinary PPO
first, isolating topology from recurrence.

**Next experiment:** Train `train_sparse_ppo.py` and compare it with the MLP
baseline under the same seed and metrics.

## 2026-09-14 — Sparse fly-inspired PPO result

**Question:** Can the sparse structured sensory pathway learn the symbolic
runner without changing the PPO or environment setup?

**First result:** The initial 20,480-step sparse run reached only 104.2 mean
steps over 20 evaluation seeds. After preserving each sensory channel through
the sparse pathway, a longer 100,352-step run reached:

```text
episodes=100  mean_steps=300.0  median_steps=300.0  mean_score=36.00  collision_rate=0.00
```

**Cost:** The saved sparse policy has 13,278 total parameters and 108 active
masked edges. The ordinary MLP has 11,014 parameters and solved the task in
20,480 steps. The sparse structure is therefore viable, but not yet more
sample-efficient.

**Finding:** The structured controller can learn the task, but the exact
symbolic observation makes the MLP a stronger and faster baseline. This is a
useful control result, not a failure: it tells us the next biological claim
must be tested on a harder, partially observed input.

**Decision:** Move to rendered pixel observations and use recurrent memory
there. Keep the symbolic results frozen as the first ablation table.

## 2026-09-14 — Pixel observation boundary

**Question:** Can the same runner expose a visual input without changing the
physics or reward function?

**Change:** Added a dependency-free 84x84 RGB renderer and a
`PixelGymRunnerEnv` wrapper. The frame has a simple road, three lane dividers,
a green player, and color/shape-coded obstacles: red blocks, orange jump bars,
and blue roll bars.

**Verification:** `pixel_check.py` passes Gymnasium's environment check, a
50-step random smoke test, shape/type validation, and a non-uniform-frame
assertion.

**Finding:** The visual task now has the correct boundary: the policy can only
use pixels, while the world dynamics remain identical to the symbolic and
sparse-policy experiments. It is still our local runner, not the real Subway
Surfers app.

**Next experiment:** Train a CNN PPO baseline from these frames, then compare
it with a recurrent CNN policy.

## 2026-09-14 — CNN PPO pixel baseline

**Question:** Can ordinary PPO learn the runner when it receives only the
rendered frame instead of the 17-value symbolic vector?

**Configuration:** `CnnPolicy`, seed `7`, 12,000 requested timesteps (12,288
collected by the 1,024-step rollout boundary), CPU training, the same reward,
spawn interval, 300-step limit, and fixed-seed evaluation used by the MLP.

**Result:**

```text
CNN PPO    mean_steps=300.0  median_steps=300.0  mean_score=36.00  collision_rate=0.00
```

The saved model is `results/cnn_ppo/cnn_ppo.zip`.

**Finding:** The tiny visual scene is learnable without symbolic state. The
CNN reached about 100-step training episodes during the run and solved all 20
held-out evaluation seeds. This is a vision control baseline, not yet a
fly-inspired or connectome-constrained controller.

**Next experiment:** Replace the CNN policy with a recurrent CNN policy and
test whether memory helps once the observation is visual.

## 2026-09-14 — Partial-view recurrent CNN comparison

**Question:** Does temporal memory help when the visual input is incomplete?

**Change:** Added `PartialPixelGymRunnerEnv`, which hides the obstacle layer on
every other frame while leaving the player, lanes, physics, and rewards alone.
Added a feed-forward CNN control and a `CnnLstmPolicy` recurrent CNN using the
same 12,000-step budget and seed.

**Results:**

```text
full CNN              mean_steps=300.0  collision_rate=0.00
partial feed-forward  mean_steps=105.1  collision_rate=0.95
partial recurrent CNN mean_steps=183.7  collision_rate=0.65
```

Both partial-view models were evaluated on the same 20 fixed seeds. The
recurrent policy is materially better than the feed-forward control, but it
does not solve the task in this short run.

**Finding:** The experiment demonstrates a real memory effect instead of
assuming recurrence is useful. The result is still an engineering baseline;
neither CNN is a biological fly brain.

**Next experiment:** Build a small reduced connectome graph and compare it
against these frozen numeric and visual baselines.

## 2026-09-14 — Reduced fly-inspired graph baseline

**Question:** Can a sparse, fly-shaped visual pathway control the partial-view
runner with the same recurrent PPO budget?

**Change:** Added `FlyGraphExtractor`. It pools lane-local red, orange, blue,
and player-green signals into 21 receptor values, passes them through two
fixed masked graph layers, and gives the resulting 24 features to a 32-unit
LSTM. The graph has 173 active edges. The receptor check confirms that a red
block activates the lane-0 red pathway.

**Result:**

```text
recurrent CNN       mean_steps=183.7  collision_rate=0.65
fly graph + LSTM    mean_steps=35.6   collision_rate=1.00
```

Both policies used 12,288 collected training steps and the same 20 evaluation
seeds. The graph-shaped controller stayed near the random baseline in this
first run.

**Finding:** The graph is wired and executable, but its hand-designed sparse
visual bottleneck is currently too restrictive or too difficult to optimize.
This is not evidence that a real fly connectome cannot work. It is evidence
that we must inspect the reduced graph and compare it with parameter-matched
controls before claiming a biological advantage.

**Next experiment:** Compare this graph with a dense feature readout and then
map a small, documented subset of an actual published fly connectome onto the
same interface.

## 2026-09-14 — Dense retina readout control

**Question:** Is the graph's poor result caused by its sparse edges, or by the
21-receptor visual bottleneck?

**Change:** Kept the same partial pixels, fixed retina pooling, 24 features,
32-unit LSTM, PPO settings, seed, and 12,288-step budget. Replaced only the
173-edge masked layers with dense 21-to-36-to-24 layers, totaling 1,620 active
edges.

**Result:**

```text
sparse fly graph + LSTM  mean_steps=35.6  collision_rate=1.00
dense retina + LSTM      mean_steps=57.6  collision_rate=1.00
```

The dense readout improves over the sparse graph but still trails the
recurrent CNN at 183.7 steps and 0.65 collision rate. This points to both a
restrictive graph and a lossy hand-designed retina; it does not justify
claiming that a real connectome would fail.

**Decision:** Freeze these controls and begin the actual connectome-data step:
select a small published fly subgraph, document its neuron/edge reduction, and
map it to the same sensory and action interfaces.

## 2026-09-14 — First real MaleCNS data slice

**Question:** Can we bring real published fly wiring into the project without
pretending that a downloaded connectome is already a working brain?

**Change:** Added `fetch_male_cns_subgraph.py`, a standard-library fetcher for
the official `male-cns:v1.0` neuPrint dataset. It records a deliberately small
type-level slice containing visual `R1-R6`, optic-lobe `L1/L2/L3`, and
descending `DNc01/DNc02`. The generated manifest preserves the source,
queries, metadata, edge weights, and reduction limitations.

**Result:** The manifest contains six published neuron types representing
8,708 annotated neurons and 30 nonzero aggregated directed type-level edges.
The read-only API query succeeded without downloading the 1.1 GB full edge
table. The downloaded body-annotation snapshot is also kept under `data/` for
inspection; it is not required by the fetcher.

**Finding:** Good news: the next model can now be constrained by actual
MaleCNS wiring. Honest boundary: this is a six-node type-level wiring summary,
not a neuron-by-neuron simulation and not yet a controller that plays the
runner.

**Next experiment:** Load this manifest into a fixed recurrent feature path,
train only the action readout, and compare it with the frozen visual controls.

## 2026-09-14 — Fixed MaleCNS graph controller

**Question:** Can the real type-level wiring drive a recurrent policy while the
connectome-derived part remains frozen?

**Change:** Added `MaleCNSGraphExtractor`. It loads the six-type manifest,
normalizes `log1p` edge weights by source, applies three fixed message-passing
hops, and exposes the two descending types over three lanes and seven visual
channels. The extractor has zero trainable parameters; PPO trains the
32-unit recurrent policy/value readout after it.

**Configuration:** Partial pixels, seed `7`, 12,288 collected steps, 300-step
episodes, spawn interval `8`, CPU, and the same PPO settings used by the
hand-designed graph and dense-retina controls.

**Result:**

```text
MaleCNS graph  mean_steps=57.6  median_steps=50.0  mean_score=5.45  collision_rate=1.00
dense retina   mean_steps=57.6  median_steps=50.0  mean_score=5.45  collision_rate=1.00
```

The model is saved at `results/male_cns_recurrent_ppo/male_cns_recurrent_ppo.zip`.
An earlier 24-feature attempt stayed at 35.6 steps; preserving the separate
near/far bands produced the 42-feature result above. The data path is now
working, but this short run does not solve the partial-view runner.

**Finding:** Good news: real MaleCNS wiring is executable and reaches parity
with the dense retina control under the same short budget. The remaining
uncertainty is the biological reduction and sensor/action mapping, not whether
the manifest can be loaded.

**Next experiment:** Move from six collapsed type nodes toward a slightly
richer documented neuron-level subset, while keeping the current controls and
evaluation protocol frozen.

## 2026-09-15 — Route-preserving MaleCNS neuron slice

**Question:** Does retaining individual neurons and their real body-level edges
give the controller a more useful path than the six-node type collapse?

**Change:** Added `fetch_male_cns_neuron_subgraph.py`. It queries the internal
`ConnectsTo` edges for the same six types, retains the two-hop
`R1-R6 → L1/L2/L3 → DNc01/DNc02` route first, and then fills the slice by
internal edge weight. The manifest keeps 48 neurons per sensory/intermediate
type and all four descending neurons.

**Result:** The query contained 7,339 participating neurons and 17,640
body-level edges. The selected artifact contains 196 individual neurons and
247 directed edges. `male_cns_neuron_check.py` verifies the graph, the 84-value
DNc output, the randomized topology control, and the trainable-edge variant.

**Finding:** The graph can now operate at individual-neuron granularity while
remaining small enough for a laptop. The selection is intentionally biased
toward a sensory-to-descending route, so it is a documented experiment rather
than an unbiased sample of the connectome.

**Next experiment:** Train the fixed graph, a randomized graph with the same
edge count, and a plastic-edge graph under the same recurrent PPO setup.

## 2026-09-15 — Neuron-level frozen/randomized/plastic comparison

**Question:** Can the selected neuron graph learn the partial-view runner, and
does allowing a small amount of edge plasticity help?

**Change:** Added `MaleCNSNeuronGraphExtractor`. It pools the same 21 fixed
visual receptors, injects them into the selected `R1-R6` neurons, applies three
message-passing hops, and exposes four DNc neurons × 21 channels as 84 policy
features. The fixed graph has no trainable graph parameters. The plastic mode
keeps the 247-edge topology but learns its 247 edge gains with PPO.

**Short-run results:**

```text
fixed graph       12,288 steps, seed 7, 20 eval seeds   mean_steps=57.6  collision_rate=1.00
randomized graph  12,288 steps, seed 7, 100 eval seeds  mean_steps=42.5  collision_rate=1.00
plastic edges     12,288 steps, seed 7, 100 eval seeds  mean_steps=76.7  collision_rate=1.00
```

The randomized control shuffles the edge endpoints and weights while keeping
the node and edge counts. The short comparison is not enough to claim a
benefit from the published topology; it does show that the graph path is
causal and measurable.

## 2026-09-15 — Plastic MaleCNS graph reaches the toy-runner target

**Question:** Does a longer run let the fixed-topology, trainable-edge model
learn reliable obstacle timing?

**Configuration:** Plastic neuron graph, seed `7`, 50,000 requested steps
(50,176 collected at the 1,024-step rollout boundary), partial pixels,
300-step episodes, spawn interval 8, CPU, and the existing RecurrentPPO
settings.

**Result:**

```text
100 fixed evaluation seeds       mean_steps=300.0  median_steps=300.0  mean_score=36.00  collision_rate=0.00
20 seeds, speed=1.25              mean_steps=300.0  collision_rate=0.00
20 seeds, spawn_interval=6        mean_steps=300.0  collision_rate=0.00
20 seeds, speed=1.5, interval=6   mean_steps=300.0  collision_rate=0.00
```

The checkpoint is saved at
`results/male_cns_neuron_plastic_50k/male_cns_neuron_plastic_recurrent_ppo.zip`.
This is the first connectome-backed controller in the project to meet the
local visual runner target. The conclusion is deliberately narrow: learned
edge gains make this selected topology trainable on the toy task. It is not a
biologically faithful fly brain and it has not yet been tested on Subway
Surfers.

**Stability note:** Separate 50,176-step plastic runs with training seeds `1`
and `2` reached, respectively, `mean_steps=107.8`, `median_steps=78.0`,
`collision_rate=0.92` and `mean_steps=177.2`, `median_steps=182.0`,
`collision_rate=0.76` on the same 100 evaluation seeds. Extending seed `2` to
100,352 collected steps improved it to `mean_steps=208.8`, `median_steps=238.0`,
and `collision_rate=0.54`, but did not solve the task. The successful seed-7
checkpoint is robust once trained, but convergence from initialization is still
variable and should not be hidden by reporting only one training seed.

**Next experiment:** prepare the explicitly gated Android capture and gesture
adapter, then ask for a connected phone only when the visual interface needs
calibration. Do not send gestures automatically during setup.

## 2026-09-15 — Live local viewer

**Question:** Can we inspect the trained controller acting instead of relying
only on aggregate evaluation numbers?

**Change:** Added `play_model.py`, a dependency-light local browser viewer. It
serves the same 84x84 frame seen by the policy, advances one recurrent PPO
step per poll, and displays the current action, reward, score, pause/resume,
step-once, and reset controls. The viewer uses the solved plastic-edge
MaleCNS checkpoint by default.

**Verification:** The PNG encoder, FFmpeg round trip, model load, and one live
controller step pass `viewer_check.py`. The viewer was launched at
`http://127.0.0.1:8765/` and its HTML/state endpoints returned successfully.

**Boundary:** This is a visual inspection tool for the local toy runner. It is
not the Android bridge and does not send phone input.

## 2026-09-15 — Rich environment Layer 2: trains and route changes

**Question:** Is the rich simulator still only showing isolated colored
blocks, while the real game asks the player to read longer routes and train
geometry?

**Change:** Added long trains as connected segments, a ramp-before-train roof
route, faster oncoming trains, track gaps, two-lane lane-change patterns, and
consecutive same-lane hazards. The six-action contract and 84x84 frame shape
did not change, so existing model and viewer plumbing remain usable.

**Verification:** `python3 tests.py` passes 14 tests, including deterministic
long-train roof traversal, jump-over-gap, and failure on a normal jump against
an oncoming train. Rich numeric, pixel, and partial-pixel Gymnasium checks
also pass.

**Interpretation:** This is a more useful training target, but it is still a
decision-equivalent simulator. It does not recreate the commercial game's
exact meshes, camera, acceleration, lane interpolation, menus, ads, or
seasonal content. Those are separate layers and should not be mixed into the
physics test until this layer is measurable.

**Next experiment:** Render and inspect the new train patterns, then train a
fresh rich baseline and compare held-out seeds against the earlier 30,720-step
run (`mean_steps=193.5`, `collision_rate=0.70`).

**Baseline result:** A fresh seed-7 RecurrentPPO run for 30,720 collected
steps produced `mean_steps=86.0`, `median_steps=74.0`,
`mean_score=14.84`, and `collision_rate=1.00` over 50 held-out seeds. This is
not a solved controller; the new layer is materially harder than the earlier
rich distribution and needs either more training or curriculum scheduling.
The checkpoint is saved at
`results/rich_layer2_recurrent_cnn/rich_recurrent_cnn_ppo.zip`.

**Next experiment:** Keep the environment layer fixed while checking whether
curriculum training (Layer 1 first, then Layer 2) improves generalization;
only after that should Layer 3 be added.

## 2026-09-15 — Viewer control-room redesign

**Question:** Can a human quickly tell what the fly sees, what action it chose,
and why the run ended?

**Change:** Replaced the single centered image and overloaded status sentence
with a responsive control-room layout: large pixel feed, explicit policy
decision card, run telemetry, event label, keyboard-friendly controls, model
identity, and a compact color legend. The viewer now also exposes lane,
motion, speed, and event fields in its state response.

**Verification:** `viewer_check.py` passes. A headless browser check passes at
1440px and 390px widths with no horizontal overflow and no console errors. The
viewer is running at `http://127.0.0.1:8766/` with the Layer 2 checkpoint.

## 2026-09-15 — Desktop resource incident and guardrails

**Incident:** During the local curriculum experiment, the desktop session
became unusable. The 30,720-step run itself completed and wrote its checkpoint;
there was no remaining training process afterward.

**Evidence:** Current-boot logs show an AMDGPU page fault naming Brave, a GPU
ring timeout/reset, and Hyprland/Brave coredumps. They do not show an
out-of-memory kill. The training was CPU-only, so CPU pressure may have made
the session less responsive, but the logs do not prove that PPO directly
caused the graphics-driver failure.

**Change:** The trainer now defaults to two PyTorch CPU threads and lower
process priority, checkpoints every 2,048 steps, and has a callback that stops
cleanly above 3 GiB process RSS or below 2 GiB system-available RAM.
`resource_guard_check.py` covers the stop path. No OS, driver, browser, or
Android settings were changed.

**Boundary:** No further training should run until the desktop is confirmed
stable. The safe next step is a short, manually supervised smoke run—not an
overnight job.

## 2026-09-15 — Guarded curriculum training result

**Run:** Started a fresh 30,720-step rich curriculum run with two PyTorch CPU
threads, low process priority, checkpointing, and the resource guard enabled.
The run completed normally at 15,360 Layer-1 steps followed by 15,360 Layer-2
steps; no guard fired and the desktop remained responsive.

**Evaluation:** The final checkpoint reached `mean_steps=81.2/300`,
`median_steps=62.0`, and `collision_rate=0.98` over 50 held-out Layer-2
seeds. A 20-seed Layer-1 check reached `mean_steps=88.0` with collision rate
`1.00`; the phase-boundary checkpoint was also weak (`78.8` mean steps).

**Conclusion:** This particular run is not a usable controller. The failure
was already present in the basic phase, so the next change should diagnose
visual learning stability or use a measured teacher/baseline—not simply run
more unrestricted PPO.

**Checkpoint:**
`results/rich_curriculum_safe_30720/rich_recurrent_cnn_ppo.zip`.

## 2026-09-15 — Canonical obstacle schema

**Question:** Are trains, buses, tunnels, and barriers being treated as too
many different RL concepts just because they look different?

**Change:** Rich observations now use six behavior classes: `solid`, `jump`,
`roll`, `either`, `gap`, and `roof_route`. A separate `appearance` field can
describe a block, train, bus, tunnel, barrier, or ramp. `speed_scale` describes
timing differences such as an oncoming train, and `roofable` describes whether
a solid train can be crossed from an elevated route. The old named obstacle
inputs remain accepted as debug aliases, but new observations expose the
canonical schema.

**Bug fixed:** Rich collision processing now handles the nearest object first.
That makes a ramp activate elevation before the train it crosses reaches the
player on the same decision step; insertion order no longer changes the route
result.

**Verification:** Added a behavior/appearance contract test and updated the
deterministic train, ramp, and oncoming-train tests. The next check is the full
local test and Gymnasium validation run before any new training.

## 2026-09-15 — Solvability check and numeric RL probe

**Question:** Is the poor pixel result caused by an impossible environment, or
by the learning/representation budget?

**Baseline:** `rich_benchmark.py` evaluated 50 fixed seeds. A random policy
survived `62.8` steps on average and collided in every episode; a symbolic
teacher survived the full `300`-step cap in all 50 episodes with zero
collisions. The task is therefore solvable under the current rules.

**Probe:** A guarded `RecurrentPPO("MlpLstmPolicy")` run used the readable
numeric observation for `65,536` steps, two PyTorch threads, low process
priority, checkpointing, and the resource guard. It completed without a guard
stop: `mean_steps=108.3/300`, `median_steps=82.0`, `mean_score=21.46`, and
`collision_rate=0.94` over 50 held-out seeds.

**Conclusion:** Numeric learning improves over the earlier 16,384-step probe
(`46.4` mean steps) and the current pixel curriculum result (`81.2` mean
steps), but it is still not a usable controller. The environment is not the
main blocker; the next experiment should target RL stability or representation
learning with a measured comparison, not blindly increase the run length.

**Checkpoint:**
`results/rich_numeric_probe_65536/rich_numeric_probe.zip`.

**Isolation run:** A separate 32,768-step Layer-1 numeric run (only the core
solid/jump/roll/either hazards) reached `mean_steps=107.8/300`,
`median_steps=70.0`, and `collision_rate=0.90`. Because this is close to the
Layer-2 result, adding more obstacle types is not the immediate fix; the next
experiment should change the learning signal or representation and compare it
against these held-out baselines.

## 2026-09-15 — MLP PPO and held-out checkpoint selection

**Change:** The readable state is already Markov, so a standard `MlpPolicy` was
tested alongside the recurrent policy. The probe now supports multiple
independent environment streams and reports disjoint seed bands. A separate
`evaluate_numeric_probe.py` audits checkpoints without changing their weights.

**Result:** The single-stream 65,536-step MLP reached `228.6` and `226.4`
mean steps on two 250-seed bands, with collision rates `0.42` and `0.49`.
Four streams reduced the gap. After 131,072 steps, the 98,304-step checkpoint
was strongest in the audit: `256.2` steps / `0.32` collisions on seeds 0–249,
`251.2` / `0.36` on seeds 1000–1249, and `251.3` / `0.30` on a third untouched
band, seeds 2000–2249. The final checkpoint was weaker on all three bands.

**Conclusion:** The selected MLP is a useful numeric simulator controller and
shows no material seed-band overfit. It is not yet a visual policy and makes no
claim about the real Subway Surfers app. The next visual step must transfer or
relearn this behavior from pixels, with the same disjoint-seed audit.

**Selected checkpoint:**
`results/rich_mlp_probe_4env_131072/checkpoints/rich_numeric_probe_98304_steps.zip`.

## 2026-09-15 — Held-out level variants

**Question:** Could the strong numeric result be memorizing one level layout?

**Change:** Added reproducible `standard`, `dense`, and `fast` rich levels plus
per-episode `mixed` training. The level changes affect spawn timing, pattern
mixture, and world speed while keeping the Gymnasium action/observation
contract unchanged. Added tests for seeded level selection and a level-aware
checkpoint evaluator.

**Run:** A four-stream MLP PPO model trained for 131,072 steps on `mixed`
levels. On 250-seed bands, the final checkpoint reached:

- `standard`: `262.3` and `260.2` mean steps, collision rate `0.26`/`0.26`;
- `dense`: `230.4` and `224.4`, collision rate `0.45`/`0.46`;
- `fast`: `253.9` and `256.2`, collision rate `0.32`/`0.31`.

**Conclusion:** Performance drops on the denser held-out level, as expected,
but remains consistent across seed bands and does not collapse outside the
training layout. The mixed-level model is the current robust simulator model;
it is not a pixel policy or real-game competence.

**Checkpoint:** `results/rich_mlp_mixed_131072/rich_numeric_probe.zip`.

## 2026-09-15 — Dense-level curriculum improvement

**Question:** Can the hard level improve without making the policy memorize
that level?

**Change:** Added a two-phase numeric curriculum: Layer 1 core timing first,
then Layer 2 mechanics. The run used four independent environments, trained on
the dense level, and kept the resource guard/checkpointing enabled. The dense
level teacher still clears `300/300` steps with zero collisions, confirming it
is solvable.

**Result:** The 131,072-step curriculum model reached `276.2` and `279.8`
mean steps on standard seeds 0–249 and 1000–1249, `273.4` and `271.3` on fast
seeds, and `252.4` and `234.4` on dense seeds. Collision rates were
`0.16`/`0.14`, `0.18`/`0.21`, and `0.33`/`0.41` respectively. An untouched
dense band, seeds 2000–2099, reached `241.5` steps with `0.41` collisions.

**Conclusion:** This is the current strongest and most transferable numeric
simulator controller. The dense level remains the hardest case, but the model
does not collapse on other levels or show a large seed-band gap. It is still
not a visual policy and has not been tested on the real phone game.

**Checkpoint:** `results/rich_mlp_dense_curriculum_131072/rich_numeric_probe.zip`.

## 2026-09-15 — Visual teacher bootstrap and DAgger recovery

**Question:** Can the strong numeric policy be transferred to pixels without
falling back to the weak from-scratch CNN PPO result?

**Change:** Added a small 84×84 RGB CNN, balanced teacher/maneuver examples,
brightness augmentation, fixed validation data, an RSS/RAM guard, and a
standalone closed-loop evaluator. Two DAgger rounds collect teacher labels from
states visited by the current visual student, including its mistakes.

**Result:** The mixed-level bootstrap reached 211.5–224.4 mean steps on
standard levels, 123.4–166.9 on dense levels, and 208.4–237.1 on fast levels.
After dense-targeted DAgger, the larger three-band audit reached:

- standard: `300.0`, `297.9`, `296.9` mean steps with collision rates
  `0.00`, `0.01`, `0.02`;
- dense: `282.4`, `267.4`, `276.4` with collision rates `0.13`, `0.23`, `0.17`;
- fast: `279.5`, `288.8`, `286.8` with collision rates `0.13`, `0.08`, `0.11`.

**Conclusion:** The visual bootstrap now behaves well in the simulator and
shows no catastrophic seed-band overfit. Dense remains the hardest level, but
the DAgger model is a large improvement over the earlier pixel PPO run. The
next step is to improve dense-level robustness before optional PPO fine-tuning;
the model still has no real Subway Surfers competence.

**Checkpoint:** `results/visual_dagger_dense/visual_teacher_best.pt`.

## 2026-09-15 — Dense-focused DAgger v2

**Question:** Can recovery-state coverage improve the hard level without
memorizing the evaluation seed band?

**Change:** Kept the original visual checkpoint untouched and trained a
separate candidate with a larger dataset, three dense DAgger rounds, and the
same fixed validation/evaluation contract. The run kept the CPU/RAM guard.

**Held-out result:** On four unseen 100-seed bands (5000, 6000, 7000, and
8000), the candidate reached:

- standard: `295.0`–`300.0` mean steps, `0.00`–`0.03` collisions;
- dense: `284.2`–`292.2` mean steps, `0.08`–`0.11` collisions;
- fast: `292.4`–`299.4` mean steps, `0.02`–`0.05` collisions.

Compared with the previous dense-focused checkpoint, dense collision rate fell
from `0.13`–`0.23` to `0.08`–`0.11`, while standard and fast performance also
improved or stayed comparable. This is evidence of better simulator
generalization, not evidence of real-game competence.

**Selected visual candidate:**
`results/visual_dagger_dense_v2/visual_teacher_best.pt`.

## 2026-09-15 — PPO transfer guardrail

**Question:** Does ordinary PPO improve the visual DAgger policy when started
from its encoder and action head?

**Change:** Added a minimal PPO transfer/evaluation path using the DAgger
encoder, a separate value head, and a low learning rate. The run used four
mixed-level environment streams for 16,384 steps and wrote to a new result
directory; the DAgger checkpoint was not modified.

**Result:** Training completed without resource problems, but the first
quality probe regressed to `0.35` standard, `0.90` dense, and `0.70` fast
collision rates on 20 unseen episodes from seed 5000. Training telemetry also
showed mean episode length falling from about `300` to `192`.

**Conclusion:** Reject this PPO checkpoint. The experiment proves the transfer
plumbing works, not that unconstrained PPO is an improvement. Keep the DAgger
candidate selected until PPO includes a behavior-preserving constraint or a
better temporal representation.

The follow-up used a frozen visual encoder, a small teacher-agreement reward,
one PPO epoch, and an early-KL stop. It serialized and trained safely, but its
20-episode seed-5000 probe still reached `0.00` standard, `0.15` dense, and
`0.10` fast collisions. It is also rejected; longer training is not the next
fix while the policy has no temporal state.

## 2026-09-15 — Temporal visual experiment rejected

**Question:** Does a four-frame input provide the missing temporal context?

**Change:** Added a 12-channel four-frame CNN with DAgger recovery training.
The newest frame is initialized to reproduce the selected static v2 policy;
older frames begin as optional context. Two larger attempts hit the existing
RSS/RAM guard and stopped cleanly; the bounded candidate completed training.

**Held-out result:** On four 100-seed bands (5000/6000/7000/8000), the
candidate reached:

- standard: `297.6`–`300.0` mean steps, `0.00`–`0.01` collisions;
- dense: `246.0`–`265.2` mean steps, `0.29`–`0.33` collisions;
- fast: `274.7`–`280.4` mean steps, `0.15`–`0.19` collisions.

**Conclusion:** Reject the temporal candidate. More input history by itself
made the closed-loop policy worse even though validation accuracy was high;
the next experiment must inspect action timing and failure states rather than
assuming frame stacking will solve them.

**Rejected checkpoint:**
`results/visual_temporal_dagger_v4/temporal_visual_teacher_best.pt`.

## 2026-09-15 — Wider static DAgger comparison rejected

**Change:** Tried four dense recovery rounds with the original single-frame CNN
on training-only seeds, leaving v2 untouched.

**Held-out result:** Across starts 5000/6000/7000/8000, standard collisions were
`1`–`3%`, dense `12`–`24%`, and fast `11`–`13%`.

**Conclusion:** More single-frame recovery data was also worse and more
variable than v2. Keep `results/visual_dagger_dense_v2/visual_teacher_best.pt`
as the selected simulator checkpoint.

## 2026-09-15 — Critical-frame DAgger comparison rejected

**Question:** Does reserving half of each DAgger round for imminent hazards
fix the last-second errors found by the failure analysis?

**Change:** Kept v2 untouched and retrained a separate single-frame candidate.
The collector stratified examples by whether the current-lane obstacle was
within 2.5 simulator distance units, while preserving general recovery
frames. The run stayed behind the same CPU/RAM guard.

**Held-out result:** Across starts 5000/6000/7000/8000, standard collisions
were `0`–`1%`, dense `11`–`20%`, and fast `6`–`14%`.

**Conclusion:** The targeted sampler did not improve the hard-level band
consistently and was worse than v2 overall. Keep
`results/visual_dagger_dense_v2/visual_teacher_best.pt` selected. The next
experiment should change action timing or add a small, tested recovery layer;
more imitation samples alone have reached a diminishing-return point.

## 2026-09-15 — Failure-timing instrumentation

**Change:** Extended `analyze_visual_failures.py` to retain the six decisions
leading into each collision. The report now separates the obstacle seen at
each decision, the model action, the teacher action, and model confidence.

**Smoke finding:** A five-episode dense check exposed a representative late
flip: the model correctly chose `roll` for four consecutive frames, then
changed to high-confidence `jump` at distance `0–2` and collided. This is a
diagnostic sample, not a performance estimate; the 100-episode held-out
metrics above remain the selection gate.

**Next experiment:** Use these histories to test one action-timing/recovery
rule in the simulator, with v2 as the unchanged baseline and the same four
held-out seed bands.

## 2026-09-15 — Broad motion-hold timing rule rejected

**Question:** Can a short output hold stop last-second jump/roll flips?

**Change:** Added an optional inference-time wrapper that holds an accepted
`jump` or `roll` against an opposite motion output for one or two decisions.
Lane changes and `noop` remain available. The wrapper is off by default and
does not alter the checkpoint.

**Pilot result:** On 20 episodes from each held-out seed band, hold-1 produced
`15`–`35%` fast collisions. Hold-2 produced `15`–`40%` standard, `45`–`75%`
dense, and `20`–`40%` fast collisions.

**Conclusion:** Reject the broad hold rule. It suppresses legitimate motion
changes along with unstable flips. Keep it as a negative control only; the
next timing experiment must require evidence of a repeated prior action
before delaying a switch.

## 2026-09-15 — Repeated-motion flip guard rejected

**Question:** Does narrowing the timing rule to sustained motion outputs avoid
the broad hold rule's false positives?

**Change:** Added a second optional inference-time rule. It delays an
opposite `jump`/`roll` only after three consecutive accepted outputs of the
prior motion, and releases the switch after one or two repeated proposals.
The rule is off by default and does not modify v2.

**Pilot result:** The one-step delay produced `10`–`25%` dense and `15`–`40%`
fast collisions. The two-step delay produced `15`–`40%` standard,
`45`–`70%` dense, and `15`–`45%` fast collisions across 20 episodes per
held-out band.

**Conclusion:** Reject timing-only guards. Even the targeted version delays
legitimate compound-obstacle transitions. The next improvement must make
near-field obstacle appearance more separable or train a representation that
preserves it; no timing wrapper is selected.

## 2026-09-15 — Near-field visual representation selected

**Question:** Does giving the pixel policy a higher-resolution view of the
near track fix last-second visual confusion without changing the action space
or physics?

**Change:** Added a six-channel input: the original 84×84 RGB frame plus a
vertically enlarged crop of rows 32–83. The added branch starts with zero
weights copied from v2, so the new model initially reproduces v2 exactly.
Training used the existing teacher/DAgger pipeline and resource guard.

**Held-out result:** Across starts 5000/6000/7000/8000, the candidate reached:

- standard: `297.7`–`300.0` mean steps, `0`–`1%` collisions;
- dense: `286.8`–`296.0` mean steps, `4`–`10%` collisions;
- fast: `292.5`–`297.9` mean steps, `2`–`4%` collisions.

**Conclusion:** Select the near-field checkpoint for simulator viewing and
keep v2 as the rollback baseline. This is a representation improvement in
the toy environment, not evidence of real Subway Surfers competence.

**Selected checkpoint:**
`results/visual_nearfield_v1/nearfield_visual_teacher_best.pt`.

## 2026-09-15 — Surprise level generalization check

**Question:** Does the selected visual policy survive a level profile it never
saw during training or normal candidate selection?

**Change:** Added an evaluation-only `surprise` profile. It uses 1.12× motion,
a rounded spawn interval of 7, a 1.10× spawn distance, and a different mix
with more oncoming and compound patterns. It is excluded from `mixed`, the
training CLI choices, and the normal three-level audit.

**Solvability check:** The symbolic teacher completed 100 surprise episodes
with `0%` collisions.

**Held-out result:** The near-field checkpoint reached:

- surprise starts 5000/6000/7000/8000: mean steps `294.5`–`300.0`, collision
  rates `0`–`4%`.

**Conclusion:** Pass the surprise generalization check and keep the near-field
checkpoint selected. This remains a simulator result; it does not establish
competence in the real Subway Surfers app.

## 2026-09-15 — Rich MaleCNS graph run rejected

**Question:** Can the downloaded neuron-level MaleCNS slice, rather than the
selected CNN, learn the richer six-action runner?

**Change:** Added rich-environment support and the existing resource guard to
`train_male_cns_neuron_ppo.py`, then trained a fixed 196-neuron/247-edge graph
and a topology-preserving plastic-edge variant for 50,176 steps each. Both
used full RGB frames and `mixed` standard/dense/fast training levels; the
held-out `surprise` profile was not used for training.

**Held-out result:** On 20 seeds starting at `5000`, the fixed graph reached
`50.5/35.5/35.9` mean steps on standard/dense/fast and `30.6` on surprise.
The plastic-edge run produced the same values. Every profile had a `1.00`
collision rate. The symbolic teacher still reaches the 300-step timeout, so
the rich task is solvable.

**Decision:** Reject both rich MaleCNS checkpoints as controllers. They prove
that the real neuron graph loads and runs through the six-action interface,
but the current 21-receptor pooling loses the information needed for the rich
visual task. Keep the near-field CNN selected and retain the brain runs as
negative controls. The next brain experiment needs a measurable retinal
mapping improvement before another long PPO run.

## 2026-09-15 — Learned-retina MaleCNS controller accepted for simulator use

**Question:** Can a visual adapter learn the 21 receptor values while the
MaleCNS graph, rather than a CNN action head, makes the final decision?

**Change:** Kept the 196-neuron/247-edge topology fixed, added a differentiable
21-value learned retina on top of the frozen six-channel near-field visual
adapter, and trained a small action head from the graph's four DNc outputs.
The hand retina was also corrected to follow perspective lanes and to group
solid/jump/roll evidence; its masks are cached and vectorized. Training used
behavior cloning plus three dense-focused DAgger rounds under the resource
guard. The evaluation-only surprise profile remained out of training.

**Confirmation:** On 100 episodes from held-out start `5000`:

| level | mean steps | collision rate |
| --- | ---: | ---: |
| standard | 297.7 | 0.01 |
| dense | 284.6 | 0.10 |
| fast | 293.1 | 0.06 |
| surprise | 295.1 | 0.04 |

On the four-band 20-episode pilot, the ranges were 0–5% standard, 0–10%
dense, 5–15% fast, and 0–5% surprise. A graph ablation on the same dense/fast
bands fell to 33–42 mean steps with `1.00` collision rate. The graph-only
checkpoint is therefore the selected fly-path simulator candidate:
`results/fly_cns_retina_v1/fly_cns_retina_best.pt`.

**Boundary:** This is a local-renderer result. The CNN is a frozen pixel-to-
retina adapter, the graph is a reduced connectome slice, and the model has not
been tested on the real Subway Surfers app. No ADB input was sent.

## 2026-09-15 — Residual hybrid retained, not promoted

**Question:** Can a bounded MaleCNS residual improve the selected near-field
CNN without sacrificing its held-out safety?

**Change:** Initialized from the selected near-field checkpoint, froze its
visual policy, and trained a bounded residual from a learned retina through
the MaleCNS graph. A zero-residual control exactly reproduces the base logits.

**Result:** The residual was nonzero but changed only 26 of 5,867 audited
actions at scale `0.25`; its 20-episode held-out aggregates were effectively
the same as the base, with a few small regressions. Scaling it to `0.5` did
not improve dense/fast survival. Keep it as an ablation, not the selected
controller; the graph-only learned-retina path demonstrates stronger causal
use of the fly graph.

## 2026-09-15 — Soft pickup objective and promotion gate

**Question:** Can the simulator teach human-like pickup decisions without a
hard rule that forbids a coin whenever an obstacle is nearby?

**Change:** Added `human_teacher_action`, which gives each action a continuous
hazard cost and a pickup value. It can collect a coin on the current route,
jump through a pickup-lane hazard, or change lanes for a pickup when the
expected danger is small. A narrow visual-label guard falls back to the
verified recovery teacher only inside the imminent `2.5`-distance window. The
old `teacher_action` remains available as the survival-only control.

The rich step info now exposes collected pickups, pickup opportunities, and
near-misses. The fly evaluator reports mean steps, score, coins, coin capture,
near-misses, and collision rate. A baseline gate rejects a candidate with more
than a three-point collision regression, more than an eight-step survival drop,
or more than `0.25` fewer coins per episode on any evaluated seed band.

**Teacher solvability check:** On 100 episodes per level, the new teacher
completed every episode without collision and collected more coins than the old
teacher:

```text
standard: old 8.15 coins, new 8.99 coins, both 0.00 collision rate
dense:    old 10.65 coins, new 12.26 coins, both 0.00 collision rate
fast:     old 9.34 coins, new 10.27 coins, both 0.00 collision rate
surprise: old 9.20 coins, new 11.01 coins, both 0.00 collision rate
```

**Anti-overfitting split:** Demonstrations use seed ranges `0–119` and
`200–319`; validation uses `1000–1049`; DAgger rollouts use `2000–2299`; the
final audit uses separate bands beginning at `5000`, `6000`, `7000`, and `8000`.
The `surprise` profile remains evaluation-only.

**Model result:** The fresh human-label fly candidate and the low-learning-rate
fine-tune were not promoted. The fine-tune smoke audit on start `5000` reached
`0.00/0.15/0.10/0.05` collision rates on standard/dense/fast/surprise,
whereas the selected checkpoint's larger 100-episode audit reached
`0.01/0.10/0.06/0.04`. The new label objective increased pickup behavior in
some bands but did not yet provide a reliable safety-preserving gain.

**Decision:** Keep `results/fly_cns_retina_v1/fly_cns_retina_best.pt` selected.
Keep the human candidates as rejected controls. The pipeline changes are
accepted because they make the objective measurable, reproducible, and
gateable without hiding a regression.

**Verification:** `tests.py` passes 22 checks; relevant modules compile;
the selected checkpoint passes a baseline self-gate; the rejected candidate
correctly returns a non-zero status under `--gate`. No ADB input or live-phone
control was used.

## 2026-09-15 — Safety-preserving pickup distillation rejected

Question: Can the accepted fly controller absorb the human-like pickup
preference without changing its established hazard actions?

Change: Started from the selected checkpoint and distilled its logits while
training on the human pickup labels. The training target preserves every
non-noop action from the anchor; only a left/right pickup preference may
override an anchor noop. This keeps the experiment small and makes the safety
assumption explicit rather than relying on a generic fine-tune.

Four-band audit: The candidate was evaluated for 20 episodes at each of the
held-out starts 5000, 6000, 7000, and 8000, separately on every level. It
passed all four fast bands. It failed standard 7000 (coins 8.75 versus
baseline 9.05), dense 6000 (coins 10.60 versus 10.90), and surprise 7000
(collision rate 0.05 versus 0.00). The other bands passed the configured
safety, survival, and pickup gate.

Decision: Reject
results/fly_cns_retina_distill_v2_smoke/fly_cns_retina_best.pt and keep
results/fly_cns_retina_v1/fly_cns_retina_best.pt selected. The anchor
constraint reduced the first candidate's dense collision regression, but the
pickup improvement is still not stable across unseen layouts. No phone or ADB
control was used.

## 2026-09-15 — Read-only real-screen calibration

Observation: The connected Motorola Edge 60 Pro reported 1220x2712. The
captured Subway Surfers frame is the main menu with Tap to Play, and is saved
at results/android_observation/subway_main_20260915.png.

Pipeline check: Added the watch-only bridge for the selected custom fly
checkpoint. Three menu screenshots produced noop at confidence 0.886, noop at
0.996, and roll at 0.927. This confirms screenshot decoding, near-field
packing, checkpoint loading, and telemetry logging; the changing menu
prediction is expected out-of-distribution behavior, not gameplay.

Boundary: No tap, swipe, key event, launch, reset, purchase, or ad interaction
was sent. The next real-device requirement is a manually started gameplay
frame so the adapter can be inspected on the track, still in watch-only mode.

Follow-up: After the game was started manually, a ten-frame watch-only burst
at 0.15-second intervals produced roll, roll, right, jump, and other
predictions. The final saved frame is the game's results/reward screen
(score 615, 21 coins), showing that the run ended while no model action was
executed. This is not a gameplay score for the model. It confirms the need
for real-game visual adaptation and a gameplay-state gate before any future
supervised action test.
