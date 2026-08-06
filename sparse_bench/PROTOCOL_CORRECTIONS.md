# CEARF-N protocol corrections (22 July 2026)

This file prevents superseded artifacts from being cited in the ADMA paper.

## Invalid historical Diginetica validation artifacts

The HID loader reconstructs each validation query by holding the last item out
of a training session. Earlier runners evaluated that query while its target
and incoming transition remained in the training index/model. Test labels were
not accessed, but validation was leaked, so component, epoch, beta, and router
selection from those runs is invalid.

Superseded artifacts include unqualified Diginetica `valid_memory.npz` caches,
the old Diginetica blocks in `cearfn_v2_*`, and the files explicitly renamed
`*.invalid-validation-leakage*` or `*.invalid-exclude-seen*`. They are retained
only for auditability and must not supply manuscript numbers.

## Locked replacement protocol

1. Cap validation deterministically at 5,000 queries.
2. For every selected `<source>_v` query, verify that the source sequence equals
   `context + target`, then replace that source training row by `context`.
3. Fit validation memory, PASGR/neighbourhood models, and neural baselines on
   these target-held-out sessions.
4. Select component cells and hyperparameters using validation metrics only.
   Exact component ties prefer fewer enabled switches.
5. Fit router families on 80% of validation and select the family on the
   disjoint 20%; exact ties prefer the simpler router.
6. After all CEARF-N decisions are locked, refit once on full training
   sessions and score the test set once.
7. Neural baselines lock their epoch on the leakage-safe validation split and
   evaluate that selected checkpoint directly for the primary paper numbers.
   A full-session refit path is retained only as a sensitivity check because
   NARM collapsed under refit despite a healthy selected checkpoint.
8. Diginetica remains repeat-aware (`exclude_seen=False`) in both phases.

Replacement caches contain `nested_valid` or `nested_test` in their names.
Paired inference clusters the three matched seeds within each test query.
Original source-session identifiers are absent from the HID test artifact, so
session-level clustering cannot be recovered and is disclosed as a limitation.
