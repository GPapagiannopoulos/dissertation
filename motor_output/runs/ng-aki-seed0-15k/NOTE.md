# Monolithic seed 0 at 15,000 steps -- the saturation diagnostic, 2026-08-23

Kept deliberately. This run answered "does the monolithic arm need more than 15,000
steps?" for a SINGLE model: no. Validation loss bottoms at step 10,000 (0.12478) and
rises after, so the peak is interior and the budget was adequate.

It is superseded as a corpus member by `ng-aki-seed0`, re-run at 30,000 steps for
STEP PARITY with the LoRA arm. That re-run is justified by the ENSEMBLE argument, not
this one: at 15,000 steps with saves every 2,000 the monolithic arm has 8 checkpoints
against LoRA's 16, which would give the two arms unequal snapshot diversity underneath
the headline comparison.
