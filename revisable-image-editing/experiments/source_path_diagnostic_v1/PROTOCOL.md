# Frozen design: source-path diagnostic v1, 14 October 2026

## Question and evidence boundary

Does a fixed intervention on source-conditioning attention improve editing across
natural task families, and does native selection of those interventions beat equal-budget
resampling and generic instruction clarification? Existing cutout training fitted its
synthetic targets without broad natural-edit gains. Cross-path interference is a new
hypothesis, not an established explanation. This diagnostic has zero gradient updates,
does not reproduce the paper checkpoint, and cannot establish method novelty.

## Precise intervention

The pinned editor uses contextualized Qwen2.5-VL tokens and clean VAE reference tokens.
Simply scaling a conditioning embedding is inappropriate because per-token RMSNorm
largely removes that scaling. Instead add a **key logit bias before softmax**:

`attention(Q,K,V) = softmax(QK^T/sqrt(d) + existing_mask + key_bias) V`.

The semantic intervention adds `log(0.5)` at retained conditioning positions whose input
IDs were native image tokens. The reference intervention adds `log(0.5)` at the appended
VAE source-reference key positions. Instructions, noisy-output key positions, learned
weights, positions, source pixels and the original source-image reference remain unchanged.

The bias multiplies unnormalized attention mass by 0.5; normalized mass is not exactly
halved. It broadcasts over **all queries/heads in all 60 blocks, at all 40 timesteps,
in both positive and negative CFG branches**. The semantic sequence is contextualized:
textual positions still contain source information, so this does NOT remove all semantic
source conditioning. Modifying source-key attention also affects the conditioning/reference
streams. Interpret it as a precisely defined token-position intervention, not isolation
of independent pure-semantic and pure-detail mechanisms.

Unity hooks return without modifying original arguments or masks. Existing padding is
retained. GPU canary requires exact decoded-RGB equality between no attention hooks and
unity hooks, then verifies 4,800 hook calls for each altered generation. Both branches'
visual masks and the actual packed source-reference length are recorded.

## Data and supervision

Use only raw photographs. CPU preparation selects/copies 2,080 nonduplicate candidates:
32 control sources plus 256 candidate sources per family. Ordering is fixed before model
outcomes. Reject encoded duplicates, decoded-RGB duplicates and dHash distance <=4
against the portable 23,119-entry declared protected registry and earlier candidates.
This excludes the previously trained 4,096 raw photos. It is not an unseen-pretraining claim.

Native Qwen2.5-VL, using the exact editor text-encoder weights and explicit modality IDs,
proposes grounded original/desired-state contracts. Admit the first 32 eligible sources
per family from the frozen pool. Families: addition, removal, replacement, attribute
(color/material), background, style, text and pose/action. One source belongs to one
family. Save all ineligible sources and schema failures. Any quota failure stops the
diagnostic without substitutions or shrinking the category denominator. Compound editing
and extraction retention are outside this first cohort and must be tested separately
before a broad submission claim.

No benchmark prompt, caption, edited target, human preference or external model supplies
proposals, gates, selection labels, rewards or curriculum. The background/foreground
data bank and original extraction adapter are not used.

## Calibration and canary

For 32 independent control photographs, overlay a red square and a protected yellow
circle. Construct four known variants: correct blue square, unchanged red square,
incorrect green square, and blue square with damaged purple circle. These are 128
internally known tests, assessed in both image orders. All five native checks must be
true in both orders for acceptance. Unknown/invalid results abstain.

Require >=90% correct-case acceptance, >=90% total classification accuracy and <=5%
false acceptance on the 96 bad controls. This tests interface/calibration on controlled
interventions, not arbitrary natural semantic correctness. Stop if the gate fails.

Select one eligible addition source for five 40-step, native-geometry canary generations:
unhooked baseline, unity baseline, semantic attenuation, reference attenuation and both.
Require exact unity-baseline pixel parity, finite output, expected hooks and sufficient
GPU memory. Record actual memory peaks and elapsed times. A failed canary cannot be
silently resumed with changed code/settings. No reduced-resolution canary substitutes
for the deployment-geometry check.

## Matched natural generations and selection

For each of 256 cases generate exactly ten distinct candidates:

| Candidate | Semantic scale | Reference scale | Seed | Instruction |
| --- | ---: | ---: | ---: | --- |
| base42 | 1 | 1 | 42 | original |
| semantic42 | 0.5 | 1 | 42 | original |
| reference42 | 1 | 0.5 | 42 | original |
| both42 | 0.5 | 0.5 | 42 | original |
| base101/base202/base303 | 1 | 1 | 101/202/303 | original |
| clarify101/clarify202/clarify303 | 1 | 1 | 101/202/303 | one fixed clarification template |

All use 40 steps, true CFG4, blank negative prompt, one image and native aspect/area.
Three selection pools each contain FOUR candidates, including the same base42:

- Intervention: base42, semantic42, reference42, both42.
- Resampling: base42, base101, base202, base303.
- Clarification: base42, clarify101, clarify202, clarify303.

Thus candidate generation/verification budgets per pool are equal. The baseline, its
native checks, and other duplicate work are reused across pools. This is a comparison
of four deterministic intervention candidates against four random-seed candidates;
the fixed seed42 factorial separately isolates individual interventions.

Each candidate is judged by the frozen native component with SOURCE and CANDIDATE in
both display orders against the ORIGINAL request. Checks: transition execution, old-state
resolution when applicable, required instance retention, applicable protected content
and new defects. Source facts requested to change are never immutable requirements.
Every check must be true in both orders. Select the first passing candidate in the stated
pool order. If no candidate passes, retain base42 and label it a fallback. No externally
judged or oracle-best candidate is used. Preserve all cases and failures in evaluation.

## Independent evaluation

Generation/selection must finish before external scoring. GPT-4.1-2025-04-14 evaluates
SOURCE and anonymous A/B candidates on five fields: semantics, instance fidelity,
preservation, quality and overall. Each comparison is judged in both A/B orders:

- semantic vs base; reference vs base; both vs base;
- both vs semantic; both vs reference;
- internally selected intervention vs base, selected resample and selected clarification.

Add one identical-image control per family: 4,104 requests total. Judge availability is
checked first. The four-worker CPU scorer records calls before sending, uses no retries,
authenticates reused receipts and caps modeled expenditure at $40. No credentials or
API output enters the native worker. API interruption requires review, not duplicate calls.

Win=1, tie=0.5, loss=0. Average both orders, group physical sources, then take an equal-
family macro. Compute 5,000 source-cluster bootstrap draws. Per-family/secondary intervals
are descriptive and unadjusted. No externally selected winning setting is promoted.
Unassessable fields or failed identical controls block a decision; they are not discarded.

## Preregistered go/no-go

All THREE primary selected-intervention comparisons (base, selected resample, selected
clarification) must have semantic credit >=0.55 and a lower 95% bound >0.50. For each,
preservation and quality lower bounds must be >=0.48, and >=6/8 family semantic means
must be >=0.50. Native intervention selection acceptance coverage must be >=25%.
Coverage includes unchanged baseline acceptances; report it separately from how often
an intervention was selected. No aggregate can hide missing families or fallbacks.

Passing means eligible for a separate mechanism confirmation, NOT automatic training.
Failure means do_not_train. The next learned mechanism would still need fixed-gate,
standard-distillation and resampling controls, independent data and multiround evidence.

## Migration and execution integrity

The target is one dedicated approximately 48GB GPU with component CPU offloading,
unchanged BF16 weights and no speed/quantization adapter. Direct execution is refused on
the originating shared host. Model/source hashes, code/config identity, runtime versions,
device type and each image/observation are authenticated. Resume reuses committed work;
uncommitted local work may repeat only its original deterministic query/seed. A different
GPU/runtime or changed method needs a new namespace. Generation and API scoring are
separate commands so no GPU is occupied during API waiting. No job is submitted here.
