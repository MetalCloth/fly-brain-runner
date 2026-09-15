# Goal: Rich Subway-Style Visual RL Environment

## Objective

Build a small, repeatable, pixel-rendered endless-runner environment that
teaches the fly brain the decisions it will need in Subway Surfers. It is a
training laboratory, not a claim that the commercial app has been recreated.

The environment must expose a normal Gymnasium-style `reset()` and
`step(action)` loop, deterministic seeded runs, rewards, terminal collisions,
timeouts, and an 84x84 RGB observation option.

## Actions

The base controller keeps the original five actions:

| ID | Action | Meaning |
|---:|---|---|
| 0 | noop | keep running |
| 1 | left | move one lane left |
| 2 | right | move one lane right |
| 3 | jump | jump over a low obstacle |
| 4 | roll | slide under an overhead obstacle |

Rich mode adds:

| ID | Action | Meaning |
|---:|---|---|
| 5 | hoverboard | spend one board charge for temporary protection |

The Android bridge will continue to expose only the original five gameplay
actions until a real-game action contract is separately validated.

## Decision-equivalent obstacles

The simulator now has one canonical gameplay schema. `kind` answers “what must
the policy do?”; `appearance` answers “what does a human see?”; timing and
route metadata answer “how does this variant move?” This prevents every train,
bus, tunnel, or seasonal skin from becoming a new RL class by accident.

| Behavior (`kind`) | Policy rule | Example appearances/metadata |
|---|---|---|
| `solid` | avoid the lane; only pass with a valid roof route or high jump | block, train, bus; `roofable=True` for a normal roof-capable train |
| `jump` | jump or change lanes | barrier |
| `roll` | roll or change lanes | roll bar, low tunnel |
| `either` | jump, roll, or change lanes | mixed hurdle |
| `gap` | jump, jetpack, or stay elevated | track gap |
| `roof_route` | enter the route; it turns on the elevated state | ramp |

The fast oncoming-train case is deliberately not another behavior: it is
`solid` + `appearance="train"` + `speed_scale=1.8` + `roofable=False`. A bus
can use the same `solid` behavior with `appearance="bus"`. The visual renderer
can distinguish those objects while the numeric encoder keeps the policy
schema compact.

Long trains are rendered as several connected train segments. A ramp placed
before them starts a short roof-running state, so the agent must learn that a
train is sometimes a route rather than only a blocker. Spawn patterns also
include two-lane lane changes and consecutive hazards. Signal/pillar/wall
appearances remain visual variants mapped to the existing decision classes.
The old names (`block`, `train`, `oncoming_train`, `tunnel`, and `ramp`) remain
accepted as debug aliases, but observations always expose the canonical
behavior plus its appearance metadata.

## Pickups and equipment

The rich environment includes these pickup classes:

- `coin`: optional collection reward.
- `key`: optional collection reward.
- `jetpack`: temporary airborne state that clears ground obstacles.
- `super_sneakers`: temporary higher jumps that can clear trains.
- `coin_magnet`: collects coins and keys across lanes.
- `multiplier`: doubles obstacle and score reward temporarily.
- `pogo`: extended/high jump state.
- `speed_pad`: temporary speed increase, reducing reaction time.
- `mystery_box`: resolves to a random useful power-up.

Hoverboards are represented as an action because the player activates them
during a run. The board absorbs one collision and is then consumed. Cosmetic
board powers, character abilities, score boosters, headstarts, menu purchases,
ads, and seasonal event rules are not separate RL mechanics in this milestone;
they will map to a canonical effect or remain outside the run environment.

The core run-time power-ups are consistent with SYBO's support documentation:
[Power-Ups](https://sybo.helpshift.com/hc/en/5-subway-surfers/faq/209-power-ups/),
[basic controls](https://sybo.helpshift.com/hc/en/5-subway-surfers/faq/205-basics/),
and [hoverboard powers](https://sybo.helpshift.com/hc/en/5-subway-surfers/faq/208-hoverboards-their-powers/).

## Level variation and overfit checks

Rich mode has four reproducible level variants without changing the action or
observation contract:

| Variant | Difference |
|---|---|
| `standard` | original spawn timing and pattern mix |
| `dense` | more frequent spawns and more multi-lane patterns |
| `fast` | faster world motion with a different pattern mix |
| `mixed` | chooses one of the three above independently at each reset |
| `surprise` | held-out speed/spawn/pattern profile, never used for training |

The numeric PPO probes train on `mixed` or a level curriculum and evaluate on
disjoint fixed seed bands for `standard`, `dense`, and `fast`. A model is
selected by held-out survival across those levels, not by its training reward.
The variants are still simulator levels; they are not a recreation of Subway
Surfers maps.

## Visual bootstrap

The first pixel policy is initialized by behavior cloning from the symbolic
teacher, then improved with dense-focused DAgger rounds. DAgger runs the
current pixel policy, labels the states it actually visits with the teacher,
and adds those recovery frames to training. This addresses compounding visual
mistakes while keeping a fixed validation set and disjoint seed bands for
early stopping and selection. The current candidate appends a vertically
enlarged crop of the near track to the original RGB frame. Across four
held-out bands it reaches 0–1% standard, 4–10% dense, and 2–4% fast collision
rates, improving on the earlier static candidate's 8–11% dense range. It is
still a visual bootstrap, not PPO fine-tuning and not a real-game controller.

The selected near-field model also survives the isolated `surprise` profile at
0–4% collision rate across four held-out seed bands. This is a simulator
generalization check only.

## Risk-aware pickup behavior

Coins and equipment are objectives, not decorations, but survival remains the
dominant objective. `human_teacher_action` evaluates the available actions with
a continuous danger cost and pickup value. That means a coin can be worth
collecting when a lane change or jump makes the route safe, can be collected
before a distant hazard, and can be skipped when the estimated collision cost
is too high. There is no global “never collect near a train” label rule.

The visual label path adds only a narrow recovery guard for hazards within
`2.5` distance. The guard prevents late labels from teaching a student to
gamble after the collision window has opened; it does not remove the
coin-versus-risk trade-off earlier in the route.

Promotion uses multiple outcome metrics: mean survival steps, collision rate,
score, coins, coin capture, and near-misses. A candidate is compared with the
selected checkpoint on disjoint seed bands and is rejected for a collision
increase above three percentage points, a survival drop above eight steps, or a
loss above `0.25` coins per episode on any band.

## Fly-path result

The first rich MaleCNS PPO runs were rejected because the hand-coded retina
lost perspective and timing information. The current graph-path candidate uses
the frozen near-field visual adapter only to emit 21 learned receptors; the
196-neuron/247-edge MaleCNS graph feeds the action head directly. On 100
episodes from held-out start `5000`, it reached 297.7/1% standard, 284.6/10%
dense, 293.1/6% fast, and 295.1/4% surprise mean steps/collision rate. A
dense/fast graph ablation collapsed to 33–42 mean steps with 100% collisions.

This is the selected fly-path simulator checkpoint, not real-game competence:
`results/fly_cns_retina_v1/fly_cns_retina_best.pt`.

A four-frame temporal variant was implemented and audited, but it reached
29–33% dense collisions. It is retained as a negative experiment; temporal
context alone is not enough, and the static DAgger candidate remains selected.

## Reward and termination

- small positive reward for surviving a step;
- positive reward for clearing an obstacle;
- small reward for collecting coins/keys/power-ups;
- doubled obstacle reward while the multiplier is active;
- negative reward for a collision;
- hoverboard collision protection avoids termination once per charge;
- `terminated=True` for an unprotected collision;
- `truncated=True` at the configured episode time limit.

## Missing complexity and layer order

This is the boundary for the build. The commercial game has more variation
than an RL prototype needs on day one, so each layer must earn its place by
changing a decision the controller has to make.

| Layer | Adds | Status |
|---|---|---|
| 0 | Three lanes, isolated hazards, fixed-speed track | complete baseline |
| 1 | Decision classes, power-ups, collectibles, hoverboard save, pixel view | complete |
| 2 | Long trains, roof route, oncoming trains, gaps, compound patterns | complete |
| 3 | Board-specific effects, speed progression, train/scenery motion variants | next |
| 4 | Continuous lane movement, jump arcs, roll windows, camera/scale motion | later |
| 5 | Coin trails, route solvability checks, denser multi-lane combinations | later |
| 6 | Start/pause/death/revive/menu/ad state machine | outside run physics |
| 7 | Real-game screenshots, domain randomization, Android calibration | sim-to-real |

The following are deliberately not separate mechanics yet: character skins,
board cosmetics, seasonal map art, missions, daily challenges, scoreboards,
shop purchases, ads, and menu navigation. They matter for a phone supervisor,
but adding them to the run policy now would teach UI handling instead of
obstacle control.

## Acceptance checks

The goal is complete only when:

1. Rich mode passes Gymnasium's environment checker.
2. Seeded rich runs are repeatable.
3. Every obstacle class has a deterministic pass/fail test.
4. Every pickup class is representable in observations and rendering.
5. Hoverboard protection, high jump, jetpack, magnet, multiplier, speed, ramp,
   and mystery-box behavior have tests.
6. Pixel observations remain 84x84 RGB and stay inside the declared space.
7. The policy is evaluated on held-out seeds and multiple level variants,
   not only the training seed, with the promotion gate applied.
8. Results are reported as simulator results and are not presented as
   Subway-Surfers phone performance.
9. A fly-path candidate must pass a graph-ablation check; a graph merely
   present in the code is not counted as brain use.

## Next work after this milestone

1. Analyze the near-field candidate's remaining failures on held-out bands;
   timing-only recovery rules have been rejected by the held-out gate.
2. Add board-specific powers, moving scenery, and camera motion without
   changing the environment contract.
3. Retry temporal/PPO changes only after a measurable representation gain;
   temporal input and extra DAgger sampling have already been rejected.
4. Collect real-game screenshots separately for visual adaptation.
5. Keep the real Android app as a later supervised evaluation target.
6. Keep the rich MaleCNS graph runs as negative controls; use the learned-
   retina graph candidate for simulator work and do not connect it to the
   phone until a separate real-game visual adapter exists.
