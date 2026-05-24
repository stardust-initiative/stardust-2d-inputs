# stardust-2d-inputs

The **input-data engine** for the Stardust 2-D atmospheric model: a lean
runtime *loader + provenance registry*, plus the offline *generators* that
produce the model's input files.

It owns all of the 2-D model's input data. Data lives in **stores** — a
private GCS bucket (everything) and a public Zenodo deposit (the verified
subset) — never in this repository. The loader is **backend-agnostic**: the
same code runs against a local directory, the private bucket, or the public
deposit, with the backend chosen by configuration.

This repository consolidates and replaces the former `climate_database_files`
and `Materials_optical_tables` repositories.

> **Status — `v0.x`, in development.** The runtime `core/` and the `era5` /
> `transport` / `optical` generators are implemented. The registry carries the
> transport-paper input set: the `transport-paper-v1` release pins the 11
> public datasets; `private-superset` pins the full private set. Data stores:
> the public Zenodo deposit is available.

## Installation

The package is `stardust_2d_inputs`. Python ≥ 3.9.

```bash
# lean runtime core (loader + registry) — what a simulation needs
pip install -e .

# + the private GCS data backend
pip install -e '.[gcs]'

# + the offline data generators (heavy deps; populated in a later phase)
pip install -e '.[generators]'

# + development / test tooling
pip install -e '.[dev]'
```

The runtime **core** has deliberately lean dependencies (`xarray`, `numpy`,
`netCDF4`, `pooch`) so that importing it at simulation runtime is cheap. The
GCS backend (`gcsfs`) and the generators (`cdsapi`, `zarr`, `scipy`, …) are
optional extras.

## Usage

### Configure

Copy the template and edit it; the real `config.json` is gitignored and is
**never** committed.

```bash
cp config.example.json config.json
```

`config.json` selects the storage backend (`local`, `gcs`, or `zenodo`) and
points the loader at the registry. No credentials are stored in it — the GCS
backend uses ambient IAM credentials, Zenodo is public.

**Pointing a model run at the engine.** A consuming model (e.g. the 2-D
climate model) usually runs from its own working directory, so it locates the
engine config via the `STARDUST_2D_INPUTS_CONFIG` environment variable:

```bash
export STARDUST_2D_INPUTS_CONFIG=/path/to/stardust-2d-inputs/config.json
```

Resolution order is `STARDUST_2D_INPUTS_CONFIG` → a `config.json` in the current
directory. Choose the backend in that config by use case:

| backend | use | release |
|---|---|---|
| `gcs` | internal / full-physics runs — the complete private superset | `private-superset` |
| `zenodo` | public reproduction — the published subset only | `transport-paper-v1` |
| `local` | offline development against a local content-addressed store | either |

(The committed `config.example.json` defaults to `zenodo` + `transport-paper-v1`,
the private-repo runtime default.)


#### Setup — external user (public, Zenodo)

No credentials and no GCS/generator extras — Zenodo is public.

```bash
# 1. install just the runtime core
pip install -e .

# 2. write config.json with the zenodo backend (the public repo ships this as its
#    config.example.json default); set record_id to the published Zenodo record
cat > config.json <<'JSON'
{
  "backend": "zenodo",
  "registry_path": "registry/registry.json",
  "releases_dir": "releases",
  "default_release": "transport-paper-v1",
  "zenodo": { "record_id": "<published record id>" }
}
JSON

# 3. point runs at it
export STARDUST_2D_INPUTS_CONFIG="$PWD/config.json"

# 4. verify (fetches over public HTTPS, hash-verified)
python -c "from stardust_2d_inputs.core.loader import load; print(load('monthly_zonal_o3', release='transport-paper-v1').sizes)"
```

### Load a dataset

```python
from stardust_2d_inputs import load

# latest verified version of a key
ds = load("Monthly_Zonal_Variables")

# pin a release manifest — the input set a given paper was run against
ds = load("Monthly_Zonal_Variables", release="transport-paper-v1")

# cherry-pick one key to an explicit older version (bug bisection)
ds = load("Monthly_Zonal_o3", release="transport-paper-v1", version="v1")
```

`load` returns an `xarray.Dataset`. The resolution rule is, in order of
precedence: **explicit `version`** → **`release` pin** → **latest `verified`
version**. A dataset on `probation` loads with a warning; one with no
verified version must be pinned explicitly.

### Run-stamping

Every `(key, version)` the loader resolves is recorded, so a simulation can
stamp its output with the exact inputs it consumed:

```python
from stardust_2d_inputs.core.loader import resolved_inputs, reset_resolved_inputs

reset_resolved_inputs()
# ... run, calling load() as needed ...
inputs = resolved_inputs()   # -> [{key, version, content_hash, status, ...}]
```

## Repository layout

```
stardust_2d_inputs/
  core/              # the ONLY part imported at simulation runtime
    loader.py        # load(key, release=, version=) -> xarray.Dataset
    backends.py      # local | gcs (private) | zenodo (public) storage
    registry.py      # the provenance registry + release manifests
    provenance.py    # generator-output stamp helper
    config.py        # config load + backend selection
  generators/        # offline tooling — [generators] extra (later phase)
    era5/  transport/  optical/
registry/registry.json   # the provenance registry — git-versioned
releases/                # per-release  key -> version  manifests
config.example.json      # config template; the real config is gitignored
tests/                   # pytest suite (runs against a local backend)
```

### The registry and releases

`registry/registry.json` is the git-versioned source of truth: one entry per
dataset key, each with one or more immutable, content-addressed versions and
a `status` (`verified` / `probation` / `deprecated`). Its git history is the
database change log. A *release manifest* in `releases/` pins a coherent
`key → version` set — the input set a publication was run against.

## Dependencies

| Scope | Packages |
|---|---|
| Core (runtime) | `xarray`, `numpy`, `netCDF4`, `pooch` |
| `[gcs]` extra | `gcsfs` |
| `[generators]` extra | `cdsapi`, `zarr`, `gcsfs`, `scipy` |
| `[dev]` extra | `pytest`, `pre-commit` |

## Development

- **Branching / commits** — see [AGENTS.md](AGENTS.md): feature branches,
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/),
  rebase-only linear history, no direct push to `main`.
- **Tests** — `pytest` from the repository root. The suite generates tiny
  fixture NetCDF files at runtime and exercises `core/` against a local
  backend; no network and no real data are needed.
- **Pre-commit** — install the hooks with `pre-commit install`; they run a
  secret scanner and a formatter on every commit. CI is the backstop.

## License

[MIT](LICENSE) for code. Data artifacts distributed via the public deposit
carry CC-BY-4.0.

## Contact

- Repository / code — Dorri Halbertal, <d.halbertal@stardust-initiative.com>

See [SECURITY.md](SECURITY.md) for the private-disclosure channel.
