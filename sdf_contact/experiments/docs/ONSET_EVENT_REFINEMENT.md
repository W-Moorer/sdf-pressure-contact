# Onset-event refinement

This update narrows the dynamics tuning focus to the **contact establishment stage**.

## What changed

- The default solver now has an onset-aware mode.
- When a step transitions from inactive contact to active contact, the solver locally refines **that onset step only** using a small number of uniform substeps.
- On the next few steps of the first contact cycle, the solver compares a one-step update against two half-steps and only keeps the refined path when the discrepancy in state/contact response exceeds configured tolerances.
- The rest of the contact segment is not uniformly refined.

## Why this matters

This targets the remaining flat-punch gap after support-area correction:

- static flat-punch force-vs-delta can already be made accurate,
- but the free-body curve can still miss the theoretical onset timing and the first-contact-cycle shape.

So the right next step is not more global refinement everywhere, but **better temporal resolution exactly at onset**.
