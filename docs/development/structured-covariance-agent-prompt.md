# Structured covariance implementation agent prompt

Implement the complete feature in [Structured covariance terms implementation requirements](structured-covariance-requirements.md). Treat that file as the acceptance contract. Read it in full before editing.

The task spans Bambi and Formulae. The deliverables are focused implementation changes for a tandem Formulae PR and Bambi PR, complete tests, user documentation, examples, validation results, benchmarks, and separate PR-ready descriptions.

## Required outcome

Add `ar1`, `ou`, `cs`, `diag`, and `us` as native Bambi structured Gaussian effects.

Use the function-call interface with `by=` grouping:

~~~python
ar1(season, by=player)

ar1(season, by=["player", "pitcher_hand"])

us(1 + x, by=player)

us(0 + C(visit), by=patient)
~~~

`us`, `diag`, and `cs` accept ordinary coefficient designs with implicit intercepts. AR(1)/OU accept coordinates. CS is heterogeneous by default and also accepts `homogeneous=True`.

Use a supported Formulae interface for structured expressions, grouping lists, and `/` expansion. Inspect each remaining Formulae change and keep the focused tandem change only where the function-call API needs it. Grouping uses `by=`; do not add a second positional grouping API or support inner group-specific expressions. Preserve ordinary group-specific terms such as `(1 + x | group)`, ordinary calls, and arithmetic. Do not replace Formulae changes with string parsing or Bambi-specific parser workarounds.

These wrappers describe covariance among latent Gaussian effects in the linear predictor. Gaussian observation noise remains conditionally independent. A separate residual-covariance component is outside this PR. Maintainer alignment on this distinction remains pending; do not describe the discussion as settled approval.

Support multiple additive structured effects, ordinary random effects, HSGPs, and auxiliary parameter formulas. In Gaussian models, integrate out at most one eligible structured term while sampling all remaining effects. Do not treat extra crossed effects as a reason to reject otherwise valid partial marginalization.

~~~python
model.fit()  # structured_method="auto"
model.fit(structured_method="latent")
model.fit(structured_method="marginal")
model.fit(
    structured_method="marginal",
    marginalize="ar1(season, by=player)",
)
~~~

Integrate at most one eligible term and recover its states jointly. Broader joint Gaussian marginalization and coordination with its prototypes remain future work; do not add a general engine, sparse solver framework, or experimental dependency.

Automatic selection is deterministic and reported. Explicit selection must be honored or rejected clearly. All four NUTS backends are required: PyMC, nutpie, NumPyro, and BlackJAX. Support both latent and eligible marginal inference on each.

AR(1) preserves integer gaps and requires declared order for non-numeric coordinates. OU accepts numeric time and datetimes, using elapsed days by default with explicit unit overrides.

Support new groups and missing states within the fitted domain for all five structures. Support new AR(1)/OU times. Fixed-domain coefficient structures require categorical levels to be declared at model construction. New numeric slope values remain valid. Recover joint conditional latent draws after marginalized fits and preserve joint uncertainty in prediction.

Read the complete requirements for priors, naming, identification, eligibility, prediction controls, pointwise log likelihood, validation, performance, and exclusions. The examples here are only an orientation.

## Working rules

- Inspect the current checkout, applicable `AGENTS.md` files, contribution instructions, and installed environment before acting.
- The specification was prepared on `codex/covariance-requirements` from upstream main at `ef4097dc61589c74c6e6c3c48a989a9be6c1553b`. Reuse the intended branch or create a feature branch from main while preserving these requirements. Never reset unrelated work.
- Inspect the Formulae repository and its instructions independently. If it is not checked out, locate it or obtain a focused checkout through the normal repository workflow.
- Keep Bambi and Formulae changes separate. Use a local dependency override for paired development; release configuration must target released Formulae support.
- The previous `structured-group-covariance` branch was abandoned because its scope grew too large. It is optional reference material for specific questions, not an implementation to resume or reproduce. Keep this effort within the agreed feature scope and avoid importing its unrelated refactors or validation infrastructure.
- Apply KISS, YAGNI, and DRY with readability first. Reuse current interfaces. Do not redesign the backend or family system for this task.
- Use the user's preferred `uv` workflow where practical and honor the actual repository's pinned tooling and CI. Avoid changing an unrelated environment. Do not reformat notebooks in place.
- Consult the Bayesian workflow skill for probabilistic code and statistical validation. Use nutpie for primary validation, plus the required four-backend compatibility checks.
- Keep implementation comments about the actual software behavior. Do not add conversation-state comments.
- Prepare reviewable changes and PR descriptions. Commit, push, or publish only with explicit user authorization. Never mention Codex in git messages.

## Lead and subagents

The user will run the lead agent on Sol with High reasoning effort. The lead owns architecture, core implementation, integration, scope control, and final acceptance. Work through one coherent implementation slice at a time.

Subagents are authorized for bounded tasks under these rules:

- Keep at most one subagent active at a time. Subagents must not delegate further.
- Use a separate Sol High reviewer for the Formulae interface, Gaussian likelihood/gradient/recovery prototype, and prediction integration. Give the reviewer fresh context containing the relevant requirements, code, and diff, without relying on the implementer's account of correctness.
- Astra High is an optional choice for difficult mathematical review if available. Its availability is not a prerequisite; Sol High with independent numerical references is sufficient for the review workflow.
- Luna Medium may handle small, fully specified fixtures, documentation, or mechanical edits. Keep covariance design, filtering, marginalization, state recovery, and coupled backend changes with the lead.
- Give every delegated task a clear question or deliverable, relevant requirements, permitted files, and an acceptance check. Agents share the filesystem; do not overlap edits. The lead integrates and checks delegated changes.
- Reviewers are read-only by default. Ask for actionable findings with file/line references and concrete evidence. For mathematical changes, ask them to derive the expected likelihood or conditional covariance independently and check the small dense reference, rather than only inspecting whether the implementation's own tests pass.
- The lead is the sole owner of local compilation and test execution. Transfer execution explicitly if needed, with the previous owner stopped. No concurrent test or sampler processes across agents.
- Review completed slices, then repair substantive findings and rerun affected checks. The lead evaluates suggestions against the specification; reviews must not expand scope or trigger repeated broad refactors.
- If subagents or model overrides are unavailable, continue with the lead and explicit numerical checks, and report the review limitation. Do not block the implementation solely on orchestration features.

## Validation and reporting

The user's machine has very limited RAM. Local tests and prototypes must be small and lightweight:

- Start with tiny datasets and focused deterministic tests. Run one test process and one sampler at a time; use separate processes to release compiled graphs between backends.
- Do not run parallel pytest workers, concurrent sampling jobs, or parallel chains locally. Use one chain and one core with short runs for local sampler smoke checks.
- Keep draws, groups, time points, and retained prediction arrays small. Use only tiny dense reference matrices.
- Do not launch the full suite, repeated calibration runs, large benchmarks, or a Stan/glmmTMB installation locally by default. Keep heavier statistical validation separate for CI or a suitably provisioned environment, and report checks that remain pending.
- Short smoke runs establish execution and shape correctness, not convergence or posterior recovery. Preserve meaningful statistical validation as a separate requirement.
- Keep validation code focused. Reuse existing tests and small scripts rather than building a general validation framework.

First prove the difficult interfaces with small executable prototypes: Formulae syntax and metadata; grouped exact AR(1) likelihood/gradients with repeated states and another sampled effect; joint conditional recovery; all four sampler backends; Bambi build/fit/prediction integration.

Use independent dense Gaussian calculations as correctness oracles. Use matched-design glmmTMB checks for covariance and conditional prediction, rather than treating frequentist estimates as Bayesian posterior truth.

Run fast deterministic regressions in normal CI and use a focused slow workflow for sampling recovery, calibration, and benchmarks. Record exact commands, versions, seeds, diagnostic results, and performance measurements. Include compile and recovery costs, not just sampler time.

Before completion, audit every section of the requirements. Report what changed, checks passed or failed, measured computational benefit, dependency/release coordination, and any remaining blockers. An unsupported required sampler, omitted syntax, incomplete prediction path, or missing exact recovery means the feature is incomplete.

When blocked on a genuine design conflict, ask one focused question with a recommendation. Resolve questions answerable from code by inspecting the code. Keep progress reports short and concrete.

## Implementation steps

1. Read the full specification and inspect both repositories, instructions, environments, and existing interfaces.
2. Implement and review the focused Formulae extension and parser/design regression tests.
3. Prove and independently review grouped AR(1) marginalization, gradients, recovery, Bambi integration, and all four samplers.
4. Implement all five latent structures, priors, coefficient designs, grouping, nesting, and auxiliary formulas.
5. Implement method selection and exact one-term marginalization for all eligible structures.
6. Complete and review prediction, recovery, aliases, metadata, log likelihood, serialization, and lifecycle regressions.
7. Run all required deterministic, sampling, calibration, and performance checks and resolve failures.
8. Deliver the implementation, documentation, validation evidence, and separate Formulae/Bambi review summaries.
