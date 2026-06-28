# visualization — interpretability overlays

`interpret_concept_overlay.py` renders, for a single whole-slide image, how each
diagnostic concept is distributed across patches — the model's interpretable
evidence for its 1p/19q prediction. Each diagnostic patch is coloured by its
per-concept **concept share** (the concept's fraction of that patch's attention).

The script imports the model and rendering helpers from the repository's main
`src/` package (it adds the repo root to `sys.path` automatically), so run it
from anywhere after cloning.

## Usage

```bash
python visualization/interpret_concept_overlay.py --help
```

- `--label TP` (true-positive / 1p/19q-codeleted) renders the oligodendroglioma
  concepts; `--label TN` (true-negative) renders the astrocytoma concept.
- `--wsi_id` selects the slide; checkpoint / anchors / feature / raw-slide paths
  default to the repo's canonical locations (a `results/morphopath` checkpoint and
  the bundled `src/conch_loc.pt` / `src/conch_score.pt`) and are overridable via
  CLI flags. Rendering requires the **Arial / Helvetica** font.
