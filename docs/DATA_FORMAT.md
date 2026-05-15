# Data Format

## DiffSynth training manifest

Each record must contain:

- `prompt`
- `image`
- `edit_image`

Example:

```json
[
  {
    "prompt": "Replace the red mug with a blue ceramic mug.",
    "image": "data/processed/train/target/example_0001.png",
    "edit_image": "data/processed/train/source/example_0001.png"
  },
  {
    "prompt": "Use Figure 2 as the dress color reference for Figure 1.",
    "image": "data/processed/train/target/example_0002.png",
    "edit_image": [
      "data/processed/train/source/example_0002_a.png",
      "data/processed/train/source/example_0002_b.png"
    ]
  }
]
```

## Field meaning

- `image`: target edited image
- `edit_image`: source or conditioning image or images

## Manifest builder input

The manifest builder accepts:

- JSON list
- JSON object
- JSONL
- CSV

Use CLI flags to map source fields into the canonical schema.

## Validation guarantees

The validator checks:

- prompt non-emptiness
- path existence
- image readability
- single `image`
- one-or-more `edit_image`

It also saves a preview contact sheet to [outputs/logs/manifest_preview.jpg](/Users/ritesh.thawkar/Ritesh/neurips-project/outputs/logs/manifest_preview.jpg).

