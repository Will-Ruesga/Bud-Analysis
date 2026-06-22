# viewer

Interactive predicted-vs-true explorer for a trained run, served with Flask.
Mirrors the structure of the cooking-recipes app (`app.py` + `routes/` +
`services/` + `static/` + `templates/`).

## What it does

- Plots **predicted vs true** (Plotly) over a grey ±0.1 tolerance band, coloured by class.
- **Filter** by view (checkboxes, plus *top only* / *all*) and by fork.
- **Aggregate** the points:
  - *None* — one point per view.
  - *Per fork* — MIL mean over the selected views (one point per `(flower, fork)`).
  - *Per flower* — mean over forks and views (one point per flower).
- Pick the **head** (aggregator) and the **split** (train/val/test/all).
- **Click a point** → that flower's points are ringed in the plot, and every view of
  the flower (grouped by fork, with each view's prediction and the fork's MIL mean)
  is shown below the graph.

Predictions are computed live: the run's trained head is applied to each image's
cached embedding, so per-view values are available for any aggregation.

## Run

The viewer ships inside the core repo and imports `core`; Flask is a core
dependency, so any venv with `base` installed can run it:

```bash
cd base/viewer
source <any-task-venv>/bin/activate     # any venv with base installed
python app.py                           # http://127.0.0.1:5000
```

It auto-discovers runs under `../../output/` (Bud-Analysis/output) that have a trained head.

## Layout

```
viewer/
├── app.py                 # Flask app, registers blueprints
├── routes/
│   ├── web.py             # the page
│   └── api.py             # /api/runs, /api/data, /api/image
├── services/
│   └── run_loader.py      # load run → per-view predictions (head over embeddings)
├── templates/index.html
└── static/{app.js, css/style.css}
```
