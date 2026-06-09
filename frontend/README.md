# DuxxDB Cloud — Frontend (React + TypeScript + Vite)

Production frontend for the DuxxDB managed cloud, replacing the embedded
zero-build SPAs (which remain as a no-dependency fallback served by the Rust
binaries).

- **Console** (`/console`) → control-plane API: organizations, projects, API
  keys (issue / rotate / revoke / mint JWT), members & invitations, project
  placement, usage.
- **Studio** (`/studio`) → data-plane read API, authenticated by a workspace
  JWT: overview, memory search, cost, evals, datasets, replay, traces, audit.

## Develop

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Run the backends so the dev proxy can reach them (shared JWT secret):

```bash
# control plane + its API on :7070
$env:DUXX_CONTROL_JWT_SECRET = "dev-secret"
duxx-control serve 127.0.0.1:7070

# data plane: RESP on :6399, Studio read-API on :7072
duxx-server --jwt-secret dev-secret --studio-addr 127.0.0.1:7072 --tenants-dir ./ws
```

The Vite dev server proxies (see `vite.config.ts`):

| Frontend path | → Backend |
|---|---|
| `/api/control/*` | `http://localhost:7070/*` |
| `/api/studio/*`  | `http://localhost:7072/studio/*` |

So there are **no CORS issues in development**.

## Build

```bash
npm run build        # type-checks (tsc -b) then bundles to dist/
npm run preview
```

## Production configuration

Point the app at the real API origins via env vars at build time, and either
enable CORS on the backends or (recommended) serve both behind one reverse
proxy / API gateway:

```bash
VITE_CONTROL_URL=https://control.yourcloud.com
VITE_STUDIO_URL=https://node-1.yourcloud.com/studio
```

## Structure

```
src/
  lib/      api.ts (typed clients + backend types), config.ts, toast.tsx
  components/ ui.tsx (Card, Pill, Table, Async, useAsync, …)
  pages/    Console.tsx (orgs/projects/keys/members), Studio.tsx (8 read tabs)
  App.tsx   top nav + routes
```
