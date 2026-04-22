# Canzuk Ltd

**Sovereign AI for the CANZUK bloc.** Built in the UK. Runs on infrastructure you control. No foreign API dependency, no telemetry, no licence tied to a parent company in another jurisdiction.

---

## What it is

**TARDIS** — a small, fast grammar model architected for the regime where worst-case complexity is intractable but typical-case structure is exploitable. The deliberate counter-thesis to frontier-scale general-purpose models: cover the same ground at a fraction of the parameter count, the cost, and the supply-chain footprint.

**Wren** — TARDIS's training architecture. Multiple "adept" substrates with diverse inductive biases (random forests, gradient-boosted trees, k-NN, ridge, with stacking and sweep-search extensions) trained against examiner folds, then promoted into a per-target ensemble. Pure-Go implementation, no GPU required, no external dependencies. Deploys on a phone, a laptop, an air-gapped tactical edge box, or a Civo cluster — the same binary, the same outputs.

---

## What it does

Currently demonstrated on:

- **Physics prediction.** Boom: Trajectory Unknown — Freelancer.com $7k asteroid-ejecta challenge. Composite NRMSE 0.285 across six targets, OOF validated on 2,930 training rows, pure stdlib Go, reproducible byte-for-byte under seed 42.
- **Cellular automata.** Many-step Game of Life prediction faster than direct simulation on typical configurations.
- **Boolean reasoning.** Structured SAT-style instances where the right inductive biases route around the worst case.
- **Inverse design.** Given output constraints, returns 20 diverse parameter sets that satisfy them — already operational in the Boom submission.

The **DrWhom** configuration extends the same architecture to cybersecurity work: vulnerability triage, fuzzing input generation, patch-effect prediction, smart-contract auditing.

---

## How it helps your business today

Three immediate vectors:

1. **Replace black-box vendor predictions with auditable in-house models.** Actuarial, supply-chain, demand forecasting, anomaly detection — TARDIS is small enough to swap in, fast enough to retrain weekly, and fully inspectable. No model card opacity. No "we cannot disclose the training data."
2. **Cybersecurity attack-surface coverage at small-model cost.** DrWhom finds the same classes of bugs that frontier-model tooling finds, on a stack that doesn't ship your codebase to a foreign cloud for inference.
3. **Design optimisation as a service.** Inverse design — *given the answer you want, find the inputs that produce it* — generalises to chemistry, materials, mechanical tolerancing, network configuration, drug screening. Demonstrated on physics, ports trivially.

---

## The strategic case

Every major component of the AI stack the UK currently runs on — model weights, training compute, inference endpoints, the toolchain that builds the toolchain — is owned by a small handful of US and Chinese firms. The same supply chain that ships those models also ships the vulnerabilities being weaponised against CANZUK critical infrastructure week after week. Continuing to build national capability on top of it cedes both the technical ground and the political leverage.

CANZUK has the engineering depth, the legal alignment between member states, and the historical trust to host its own sovereign AI capability. Canzuk Ltd is building it now, in Go, in the open, on hardware nation-states can audit.

Engaging now means a seat at the table when the question of who owns the next decade's defensive AI is decided. Not engaging means defaulting that decision to the same actors whose stacks are the reason we need this in the first place — or to threat actors whose stacks we can't trust by definition.

There is no third option in which someone else builds this for the bloc.

---

## Get in touch

**Matt Kilcoyne** — `mrjkilcoyne@gmail.com`

GitHub: [@mrjkilcoyne-lgtm](https://github.com/mrjkilcoyne-lgtm)
