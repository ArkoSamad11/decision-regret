# Architecture — hosting split and storage

```
Browser
  │  same-origin /api/*
  ▼
Vercel (Next.js)  ──rewrite──▶  Railway (FastAPI + DuckDB, Docker)
```

## Hosting split

Vercel hosts the frontend only. `web/next.config.mjs` rewrites `/api/:path*` to
`${API_ORIGIN}/:path*`. Three consequences, all deliberate (SPEC.md §14):

- **No CORS preflight on the interactive path.** The browser only ever talks to
  its own origin, so no `OPTIONS` round-trip sits in front of moment selection.
- **The backend URL stays out of the client bundle.** `API_ORIGIN` is read at
  request time in a server component, never shipped to the browser. This is why
  `MomentPicker` is the only `"use client"` component in the tree — everything
  else stays a server component so the origin has no route into client code.
- **The API can move without a frontend redeploy.** Changing hosts is an env-var
  change on Vercel, not a rebuild.

The API is stateless: no session, no writes, no attached volume. Every request
is a read against a DuckDB file that was baked into the image at build time.

## Storage: a DuckDB file in the image, not a database service

The served data is ~12 MB, read-only, and regenerated wholesale by
`make serve-db` whenever the model changes. It is never written at runtime.

A Postgres service would add a network hop, a connection pool, a migration
story, and a bill, and would buy nothing: there is no concurrent write to
serialize and no row that outlives a deploy. Baking the file into the image also
makes the artifact honest — the image and the numbers it serves are versioned
together, so a given container can only ever serve the run that produced it.
`run.json` carries the `config_hash` and `run_id`, and `/health` reports both.

The tradeoff: new data requires a rebuild and redeploy, not an `INSERT`. For a
tournament dataset that is finished and will not change, this is the right side
of the trade.

## What the image contains

Built from the **repo root**, not `api/`, because it needs `configs/` and
`artifacts/` alongside the package:

```bash
docker build -f api/Dockerfile -t xdr-api .
```

Baked in: the installed `xdr` package, `configs/`, and from `artifacts/` the
DuckDB file, `run.json`, and the evaluation/transfer reports.

Deliberately excluded: `artifacts/support_index.joblib` (~80 MB, 87% of the
artifacts payload). The k-NN support gate runs at database *build* time, on a
workstation, and its verdicts are already recorded per option inside
`xdr.duckdb`. Nothing on the serving path opens the index. See `.dockerignore`.

### Two known weight problems, not yet addressed

1. **The image installs far more than it serves.** `xdr.serve.app` imports
   `xdr.serve.store`, which imports `lightgbm`, `pandas`, `joblib`, and
   `sklearn` at module level — and `xdr/__init__.py` imports `sklearn.neighbors`
   deliberately (a Windows OpenMP workaround, see DECISIONS.md). So the whole
   training stack ships even though serving is pure DuckDB reads. Fixing it
   means moving the build-time imports inside `build_database`; that is a real
   refactor with a real regression risk, and it has not been done.
2. **Image size is not separately measured**, only the compressed upload bundle
   (~4.3 MB — see "Deploying" below). No local container runtime was available
   to inspect the built image directly; Railway builds and runs it, and `/health`
   confirms the running container serves real data, but the on-disk image size
   against SPEC.md §14's host-limit concern has not been checked independently.
   SPEC's related concern about the CPU-only torch wheel is moot — torch is not
   a dependency, because the DeepSets encoder (M6) was deferred.

## Deploying

**API (Railway).** `railway.json` selects the Dockerfile builder and points at
`api/Dockerfile`. `artifacts/` is gitignored, so **run `make reproduce` locally
first** — there is nothing to bake in otherwise, and the failure is quiet.

**The build context is not simply "the repo root."** `railway up`'s default
upload respects `.gitignore` (not a Railway-specific `.railwayignore` — one was
tried and had no effect on this CLI version), which excludes `artifacts/`
exactly like git does. Deploying from the repo root as-is silently produces a
~126 KB bundle missing the one directory the image needs, and the build fails
on `COPY artifacts artifacts` with `"/artifacts": not found`. The reliable
approach: build a **minimal staging directory** containing only `api/`
(`pyproject.toml`, `src/`, `Dockerfile`), `configs/`, `artifacts/` (minus
`support_index.joblib`), and a copy of `railway.json` at its root (without it,
Railway's auto-detect builder — Railpack — runs instead of Docker and fails
outright, since it doesn't recognize an `api/` subdirectory as an app root),
then `railway up` **from inside that staging directory**. This produced a
~4.3 MB compressed bundle and a real build. Running `railway up <path>` instead
of `cd`-ing into the staging directory first creates a *new*, disconnected
Railway project named after the path's directory name, rather than deploying
to an already-linked project — link state is tied to the CWD, not passed
through via a path argument.

The container listens on `$PORT`, which Railway injects; the Dockerfile falls
back to 8000 so local `docker run` still works. A public domain is not created
by default — `railway domain` generates one after the first successful deploy.

**Frontend (Vercel).** Root directory `web/`. Set `API_ORIGIN` to the Railway
service's public URL. `web/vercel.json` pins the framework, region, and security
headers.

### Verifying a deploy — do not trust a green healthcheck

`railway.json` points the healthcheck at `/health`, which returns **200 even
when no artifacts are present**, with `{"status": "no_artifacts"}`. That is the
honest answer for the route, but it means Railway will happily mark a container
healthy when it has nothing to serve, and the dashboard will render its empty
state against a "working" API.

After deploying, check the body, not the status code:

```bash
curl -s https://<service>.up.railway.app/health
# want: {"status":"ok","model_version":"<hash>","run_id":"<id>"}
# not:  {"status":"no_artifacts","model_version":null,"run_id":null}
```

`no_artifacts` on a deploy that was supposed to have data means the bake failed
or `XDR_ROOT` is not pointing where the files landed — see DECISIONS.md, "repo
root resolution under a non-editable install."

## Status

**Live.** The API is deployed on Railway and verified against real data:
`/health`, `/matches`, `/matches/{id}/moments`, `/calibration`, and `/transfer`
all return real Euro 2024 / Women's Euro 2025 numbers from the actual
`xdr.duckdb` baked into the image, not `no_artifacts`. The first two deploy
attempts failed for reasons now fixed and documented above and in
DECISIONS.md: a missing `artifacts/` in the build context, and a
`multimethod`/`pandera` import error that only surfaced on a genuinely clean
install. Frontend deployment to Vercel is the remaining step.
