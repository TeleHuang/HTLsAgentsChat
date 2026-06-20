# Paper 2 figure and experiment requirements update

Date: 2026-06-20
Owner: local Coding Bot
Audience: Manager Bot and distributed experiment agents

## Scope

This update supersedes the first draft figure-generation plan for Paper 2. The original Chinese paper draft is archived as:

`specs/paper-sync/paper2_chinese_draft_original_20260620.md`

The current task is to redraw the paper figures with publication-grade quality while avoiding misleading placeholder data.

## Plotting standard

- Use the local `nature-figure` skill with the Python backend.
- Export vector-first outputs: SVG and PDF with editable text; PNG is a preview only.
- Every generated figure must include source-data CSV files and a manifest stating whether the panel is real measurement, derived from reports, schematic, or blocked by environment.
- Missing metrics must be shown as `n/a` or omitted from the panel. Do not encode missing values as zero.

## User feedback on existing figures

- Fig. 1: Text exceeds boxes; redesign layout and wrapping.
- Fig. 2: Visually acceptable, but provenance is unclear. Prefer real source data if available.
- Fig. 3: Abnormal; rerun a tolerance sweep for `mindnlp`, `mindtorchV2`, and `torch4ms` from local `mindnlp/src`.
- Fig. 4: Flow lines are visually messy; simplify the agent workflow diagram.
- Fig. 5: Replace the 3D design-space plot with a 2D chart. The intended dimension is model x component x failure mode, but a 2D matrix or bubble chart is acceptable if it better supports interpretation.
- Fig. 6: The all-zero torch4ms pass rate is suspicious. Use `-demo` as the immediate source of truth for this figure and diagnose environment or plugin faults separately.
- Fig. 7: Acceptable for now; data volume is small but not immediately blocking.
- Fig. 8: Acceptable for now; a stronger version likely depends on GPU-vs-NPU experiments.
- Fig. 9: Current radar chart is misleading because missing metrics were treated as zero. Replace or redesign it so unavailable metrics are explicit.

## Immediate experiment decisions

- Use `-demo` for fresh quick evaluations of local adapters where possible.
- Candidate local packages under `/home/ma-user/work/mindnlp/src`:
  - `mindnlp`: currently import-blocked by missing `transformers`; classify as environment dependency until remediated.
  - `mindtorch`: importable as a torch-compatible namespace.
  - `mindtorch_v2`: importable as a torch-compatible namespace when aliased into `sys.modules["torch"]`.
  - `torch4ms`: importable in a clean subprocess; must run isolated because mixed imports can pollute results.
- Fig. 3 should be based on real tolerance reruns or clearly marked blocked/incomplete where adapter execution fails.
- Fig. 6 should use `-demo` summaries, not the suspicious CC-plugin torch4ms summary.
- Fig. 9 should be a metric-availability/scorecard-style visualization unless enough complete metrics exist for a radar plot.

## Open risks

- Current NPU environment reports MindSpore TBE warnings; distinguish environment/configuration failures from bridge compatibility failures.
- `mindnlp` requires optional dependencies for HuggingFace compatibility. Installing dependencies may be needed before using it as a real bridge candidate.
- Some local adapters are namespace replacements rather than runtime interceptors; their preambles must be evaluated in subprocess isolation to avoid cross-adapter contamination.

