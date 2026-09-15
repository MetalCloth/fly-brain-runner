# Real MaleCNS slice

This is the first point where the project uses real fly-connectome data. The
file [`data/male_cns_visual_subgraph.json`](../data/male_cns_visual_subgraph.json)
is a small manifest queried from the published MaleCNS `v1.0` dataset through
neuPrint.

That manifest is now loaded by `MaleCNSGraphExtractor`. The graph extractor
has zero trainable parameters; PPO trains the recurrent policy/value readout
after the fixed graph.

## What we selected

The slice has six published neuron types:

| type | broad role | neurons in dataset |
| --- | --- | ---: |
| `R1-R6` | visual sensory | 3,377 |
| `L1` | optic-lobe intrinsic | 1,776 |
| `L2` | optic-lobe intrinsic | 1,779 |
| `L3` | optic-lobe intrinsic | 1,772 |
| `DNc01` | descending output | 2 |
| `DNc02` | descending output | 2 |

The type-level slice therefore represents 8,708 annotated neurons, but the
controller will initially use only six graph nodes. That compression is
intentional: it lets us test the data path on a laptop before attempting a
neuron-by-neuron simulation.

The chosen path is easy to explain: visual input enters `R1-R6`, optic-lobe
types `L1/L2/L3` provide an intermediate layer, and `DNc01/DNc02` give us a
small descending-output endpoint. The manifest records 30 nonzero directed
type-to-type edges and the exact queries used to obtain them.

## Reproduce it

From `fly-brain-runner`:

```bash
./.venv/bin/python fetch_male_cns_subgraph.py
./.venv/bin/python -m json.tool data/male_cns_visual_subgraph.json >/dev/null
```

The fetcher uses Python's standard library. A `NEUPRINT_TOKEN` environment
variable is accepted if the server requires authentication; the current
read-only query worked without one. We do not download the 1.1 GB full edge
table for this milestone.

The upstream data and access instructions are documented by the
[MaleCNS download page](https://male-cns.janelia.org/download/) and the
[neuPrint Python quickstart](https://connectome-neuprint.github.io/neuprint-python/docs/quickstart.html).

## What this proves

- The project can reach the official dataset.
- The selected sensory/intermediate/descending types exist in the dataset.
- Real published edge weights can be stored in a small, inspectable artifact.
- A controller can be constrained by real wiring instead of the hand-designed
  `FlyGraphExtractor` topology.

## What it does not prove

This is not “the fly brain running.” It is a type-level wiring summary. It
does not contain membrane dynamics, synapse signs, delays, neuromodulation,
or a validated mapping from game pixels to fly photoreceptors. Those mappings
will be explicit engineering assumptions and will be tested against the CNN
controls.

## Controller result

On the partial-view runner, with seed `7` and 12,288 collected PPO steps, the
fixed MaleCNS controller reached:

```text
mean_steps=57.6  median_steps=50.0  mean_score=5.45  collision_rate=1.00
```

This matched the dense retina control at the same training budget. It is a
successful wiring/data integration, not a solved game. The graph reduction is
also not a neuron-level simulator: it uses `log1p` edge weights, source-column
normalization, three fixed message-passing hops, and lane/channel replication
so the tiny type graph can retain runner information.

## Next smallest experiment

Replace the six-node type collapse with a slightly richer documented
neuron-level subset, preserving the same pixel interface and PPO controls.
If that wiring still cannot learn, inspect the biological reduction and the
sensor/action mapping before touching the phone or the real game.

## Neuron-level reduction

The next reduction is stored in
[`data/male_cns_neuron_subgraph.json`](../data/male_cns_neuron_subgraph.json).
It queries individual `ConnectsTo` edges within the same six published types,
then keeps a small route-preserving slice:

```text
R1-R6  →  L1/L2/L3  →  DNc01/DNc02
```

The fetcher first retains the two-hop sensory-to-descending route, then ranks
neurons by the total weight of their internal edges. It keeps all four
descending neurons and at most 48 neurons per other type. The source query
contains 7,339 participating neurons and 17,640 body-level edges; the
selected manifest contains 196 neurons and 247 edges.

This is still a deliberately selected subgraph, not a complete brain model.
The controller maps the 21 pooled visual channels into the selected `R1-R6`
neurons, applies three message-passing hops, and reads the four selected DNc
neurons as 84 features. The fixed variant has no trainable graph parameters.
The plastic variant keeps the same topology but makes the 247 edge strengths
trainable, which is an explicit engineering relaxation of the fixed
connectome constraint.

Reproduce the selection and structural checks with:

```bash
./.venv/bin/python fetch_male_cns_neuron_subgraph.py --max-neurons-per-type 48
./.venv/bin/python male_cns_neuron_check.py
```

## Neuron-level controller results

All runs use the partial-view runner, 300-step episodes, spawn interval 8,
CPU, and RecurrentPPO. Evaluation is deterministic and resets the LSTM state
for each fixed seed.

| controller | training | evaluation | mean steps | collision rate |
| --- | ---: | ---: | ---: | ---: |
| fixed graph | 12,288 steps, seed 7 | 20 seeds | 57.6 | 1.00 |
| randomized graph | 12,288 steps, seed 7 | 100 seeds | 42.5 | 1.00 |
| plastic edges | 12,288 steps, seed 7 | 100 seeds | 76.7 | 1.00 |
| plastic edges | 50,176 steps, seed 7 | 100 seeds | 300.0 | 0.00 |

The 50,176-step plastic checkpoint also timed out at 300 steps on 20 unseen
seeds with speed 1.25, on 20 seeds with spawn interval 6, and on 20 seeds with
both speed 1.5 and spawn interval 6. This is a successful result on the local
runner, not evidence that the model understands the commercial game.

Training initialization still matters. Separate 50,176-step plastic runs with
seed `1` and seed `2` reached, respectively, `107.8` mean steps / `0.92`
collision rate and `177.2` mean steps / `0.76` collision rate over 100
evaluation seeds. Extending seed `2` to 100,352 collected steps improved it to
`208.8` mean steps, `238.0` median steps, and `0.54` collision rate. The
selected seed-7 checkpoint is therefore a robust policy across evaluation
conditions, but the current training recipe is not yet claimed to converge
reliably from every random initialization.

The short fixed-versus-randomized comparison does not establish that the
published topology is better than a random graph. The useful result is that
the real topology can be retained while a small, explicit set of edge gains is
learned. Biological neuron dynamics, synapse signs, delays, and the
pixel-to-photoreceptor mapping remain outside this experiment.

## Rich runner experiment

The next test kept the 196-neuron, 247-edge graph but moved it to the Layer 2
rich runner: full 84x84 RGB frames, six actions, long trains, roof routes,
gaps, compound hazards, and the mixed standard/dense/fast training profile.
The `surprise` profile remained evaluation-only. Both runs used seed `7` and
50,176 collected RecurrentPPO steps under the resource guard.

| controller | standard | dense | fast | surprise |
| --- | ---: | ---: | ---: | ---: |
| fixed graph | 50.5 / 1.00 | 35.5 / 1.00 | 35.9 / 1.00 | 30.6 / 1.00 |
| plastic edges | 50.5 / 1.00 | 35.5 / 1.00 | 35.9 / 1.00 | 30.6 / 1.00 |

Each cell is `mean_steps / collision_rate` over 20 seeds starting at `5000`.
The symbolic teacher solves these profiles, so the failure is in this visual
reduction/controller, not an impossible level. The rich graph checkpoints are
kept as reproducible negative controls; they do not replace the selected
near-field visual policy. More PPO time is not the next move: the retinal
pooling and graph-to-action information path need a new measurable design
first.

The rich plastic checkpoint is
`results/male_cns_neuron_rich_plastic_50k/male_cns_neuron_plastic_full_rich_recurrent_ppo.zip`.
It is a simulator artifact only and has not been tested on the phone app.

## Learned-retina graph controller

The pure PPO graph runs above were a useful negative control: the fixed
21-value hand retina did not preserve enough rich visual information. The
replacement keeps the graph topology but learns the pixel-to-retina mapping.
The frozen six-channel near-field CNN is only a visual adapter. It emits 21
learned receptor values; those values are injected into the 196-neuron graph,
and the action head receives only the resulting 84 DNc features.

Training uses the existing behavior-cloning/DAgger helpers, with the same
disjoint seed bands and evaluation-only surprise profile. The selected
checkpoint is `results/fly_cns_retina_v1/fly_cns_retina_best.pt`.

On 100 episodes from start `5000`, it reached `297.7/0.01` standard,
`284.6/0.10` dense, `293.1/0.06` fast, and `295.1/0.04` surprise, where each
cell is mean steps/collision rate. On the four-band 20-episode pilot, zeroing
the graph output caused 100% collisions and roughly 33–42 mean steps on
dense/fast. This establishes that the graph is materially used in the local
simulator, not that the reduced graph is a biological brain simulation or a
real-game controller.
