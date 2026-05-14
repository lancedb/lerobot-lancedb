# Building and deploying the docs

The site at <https://lancedb.github.io/lerobot-lancedb/> is built from this
`docs/` folder with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

## Local preview

Install the docs extras and run the dev server:

```bash
pip install -e .[docs]
mkdocs serve
```

Open <http://127.0.0.1:8000/>. Edits to anything under `docs/` or to
`mkdocs.yml` hot-reload automatically.

## One-off build

```bash
mkdocs build --strict
```

Output lands under `site/`. `--strict` turns warnings into errors so CI
catches broken links / missing nav entries.

## Deploy

GitHub Pages is auto-deployed by
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on every push
to `main`. The workflow:

1. Installs `pip install -e .[docs]`.
2. Runs `mkdocs build --strict`.
3. Publishes `site/` to the `gh-pages` branch via `peaceiris/actions-gh-pages`.

The first deploy requires Pages to be enabled with source set to the
`gh-pages` branch (Repository **Settings → Pages → Build and deployment →
Source = Deploy from a branch, branch = gh-pages /(root)**).

## Manual deploy (fallback)

If the workflow ever fails and you need to push docs by hand:

```bash
pip install -e .[docs]
mkdocs gh-deploy --force
```

This builds locally and pushes to the `gh-pages` branch using your
existing git credentials.

## File layout

```
docs/
  index.md          Front page (intro + nav + headline benchmark)
  conversion.md     Both converter CLIs + Python API
  frames-format.md  LeRobotLanceDataset (per-frame JPEG / PNG)
  video-format.md   LeRobotLanceVideoDataset (per-file mp4 blobs)
  examples.md       Tour of examples/
  benchmarks.md     Full benchmark tables + methodology
  README.md         You are here
mkdocs.yml          Site config (theme, nav, markdown extensions)
.github/workflows/docs.yml   Deploy workflow
```

When adding a new page: drop the `.md` under `docs/`, list it under `nav:`
in `mkdocs.yml`, and either re-run `mkdocs build` locally or push to
`main`. The workflow will pick it up.
