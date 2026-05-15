# Benchmark Tables For Meeting

This file is for professor-facing tables where we need:

- **official baseline numbers** from public sources
- **our method rows** with clearly labeled target ranges until the actual runs finish

Do not present any `target` or `TBD` value as a measured result.

## Official Qwen Baseline

The following baseline values come from the official Qwen-Image technical report:

- `DPG = 88.32`
- `GenEval = 0.91`
- `GSO = 15.11`
- `GEdit-EN = 7.56`
- `GEdit-CN = 7.52`
- `ImgEdit = 4.27`
- `OneIG-EN = 0.539`
- `OneIG-ZH = 0.548`

Notes:

- The OneIG values are explicit in Table 5 and Table 6 of the report.
- The DPG, GenEval, GSO, GEdit, and ImgEdit values are read from the summary benchmark figure in the report.
- These are official baseline values for the Qwen-Image family, not our local fine-tuned checkpoint.

Source:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324

## Editing-Focused Table For Slides

Use this if the meeting is about the self-evolving image editing direction.

| Method | Status | GEdit-EN | ImgEdit | Note |
| --- | --- | --- | --- | --- |
| Qwen-Image official baseline | official | `7.56` | `4.27` | public report value |
| Our local supervised baseline | estimate | `7.30` | `4.12` | conservative local reproduction estimate |
| Naive self-training | estimate | `7.18` | `4.06` | likely unstable or slightly worse |
| Current hybrid reward | target | `7.8 - 8.2` | `4.4 - 4.7` | expected modest gain if filtering helps |
| Final hard-gated + relative ranker | target | `8.0 - 8.6` | `4.6 - 5.0` | current main method |
| `+` counterfactual reward | target | `8.2 - 8.8` | `4.7 - 5.1` | optional follow-up |

## Generation-Sanity Table For Slides

Use this only if you want to show that editing improvements should not destroy general generation quality.

| Method | Status | DPG | GenEval | OneIG-EN | OneIG-ZH |
| --- | --- | --- | --- | --- | --- |
| Qwen-Image official baseline | official | `88.32` | `0.91` | `0.539` | `0.548` |
| Our local supervised baseline | estimate | `87.90` | `0.895` | `0.531` | `0.540` |
| Current hybrid reward | target | `>= 87.5` | `>= 0.89` | `>= 0.52` | `>= 0.53` |
| Final hard-gated + relative ranker | target | `>= 87.5` | `>= 0.89` | `>= 0.52` | `>= 0.53` |

Interpretation:

- I would not frame the paper around generation gains.
- The generation table is mainly a regression guardrail.
- The main gains should be argued on `GEdit-EN` and `ImgEdit`.

## Safe Script For Explaining The Table

Use wording like:

> The Qwen row is the official public baseline from the technical report. The rows for our methods are not claimed results yet; they are the experiment slots and target ranges for the first GPU runs. The point is that the method space is already narrowed and the experiments are clearly defined.

Avoid wording like:

- "we already achieved"
- "our score is"
- "we improved by"

until the actual benchmark runs are completed.
