import { CONTROL_URL, STUDIO_URL } from './config'

// ---------------------------------------------------------------------------
// Types — mirror the Rust backend contracts (duxx-control + duxx-server/studio)
// ---------------------------------------------------------------------------

export interface Org {
  id: string
  name: string
}
export interface Project {
  id: string
  org_id: string
  name: string
}
export interface ApiKey {
  id: string
  /** Present only on issue/rotate; redacted in list responses. */
  secret?: string
  org_id?: string
  project_id: string
  env: string
  role: string
  name: string
  revoked: boolean
}
export interface Member {
  id: string
  org_id: string
  email: string
  role: string
  status: string
}
export interface Usage {
  requests: number
  tokens_in: number
  tokens_out: number
  cost_usd: number
}
export interface Placement {
  project_id: string
  node: string
  mode: string
}

// Studio (data-plane) read models
export interface Overview {
  tenant: string
  role: string
  memory_count: number
  prompts: string[]
}
export interface MemoryHit {
  id: number
  score: number
  text: string
}
export interface CostBucket {
  key: string
  count: number
  tokens_in: number
  tokens_out: number
  total_usd: number
  mean_usd: number
}
export interface CostSummary {
  tenant: string
  total_usd: number
  by_model: CostBucket[]
}
export interface EvalRun {
  id: string
  name: string
  model: string
  status: string
  dataset_name: string
}
export interface ReplaySession {
  trace_id: string
  captured_at_unix_ns: number
}
export type Span = Record<string, unknown>
export interface AuditEvent {
  ts_unix_ns: number
  principal: string
  command: string
  outcome: string
}
export interface DocumentMeta {
  id: string
  uri: string
  content_type: string
  version: number
  chunks: number
}

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------

export const errMsg = (e: unknown): string =>
  e instanceof Error ? e.message : String(e)

interface ReqOpts {
  method?: string
  body?: unknown
  token?: string
}

async function request<T>(url: string, opts: ReqOpts = {}): Promise<T> {
  const headers: Record<string, string> = {}
  if (opts.body !== undefined) headers['content-type'] = 'application/json'
  if (opts.token) headers['authorization'] = `Bearer ${opts.token}`
  const res = await fetch(url, {
    method: opts.method ?? 'GET',
    headers,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  })
  const text = await res.text()
  const data = (text ? JSON.parse(text) : {}) as unknown
  if (!res.ok) {
    const msg =
      (data as { error?: string }).error ?? `${res.status} ${res.statusText}`
    throw new Error(msg)
  }
  return data as T
}

// ---------------------------------------------------------------------------
// Control-plane API
// ---------------------------------------------------------------------------

export const control = {
  health: () => request<{ status: string }>(`${CONTROL_URL}/healthz`),

  listOrgs: () =>
    request<{ orgs: Org[] }>(`${CONTROL_URL}/v1/orgs`).then((r) => r.orgs),
  createOrg: (name: string) =>
    request<Org>(`${CONTROL_URL}/v1/orgs`, { method: 'POST', body: { name } }),

  listProjects: (org_id: string) =>
    request<{ projects: Project[] }>(`${CONTROL_URL}/v1/projects/list`, {
      method: 'POST',
      body: { org_id },
    }).then((r) => r.projects),
  createProject: (org_id: string, name: string) =>
    request<Project>(`${CONTROL_URL}/v1/projects`, {
      method: 'POST',
      body: { org_id, name },
    }),

  listKeys: (project_id: string) =>
    request<{ keys: ApiKey[] }>(`${CONTROL_URL}/v1/keys/list`, {
      method: 'POST',
      body: { project_id },
    }).then((r) => r.keys),
  issueKey: (project_id: string, env: string, role: string, name: string) =>
    request<ApiKey>(`${CONTROL_URL}/v1/keys`, {
      method: 'POST',
      body: { project_id, env, role, name },
    }),
  revokeKey: (key_id: string) =>
    request<{ revoked: boolean }>(`${CONTROL_URL}/v1/keys/revoke`, {
      method: 'POST',
      body: { key_id },
    }),
  rotateKey: (key_id: string) =>
    request<ApiKey>(`${CONTROL_URL}/v1/keys/rotate`, {
      method: 'POST',
      body: { key_id },
    }),
  mintToken: (api_key_secret: string, ttl_secs: number) =>
    request<{ jwt: string }>(`${CONTROL_URL}/v1/tokens`, {
      method: 'POST',
      body: { api_key_secret, ttl_secs },
    }),

  placeProject: (project_id: string, node: string, mode: string) =>
    request<Placement>(`${CONTROL_URL}/v1/placements`, {
      method: 'POST',
      body: { project_id, node, mode },
    }),
  authEntries: (node: string) =>
    request<{ entries: string[] }>(`${CONTROL_URL}/v1/auth-entries`, {
      method: 'POST',
      body: { node },
    }).then((r) => r.entries),

  usage: (project_id: string) =>
    request<Usage>(`${CONTROL_URL}/v1/usage/query`, {
      method: 'POST',
      body: { project_id },
    }),

  listMembers: (org_id: string) =>
    request<{ members: Member[] }>(`${CONTROL_URL}/v1/members/list`, {
      method: 'POST',
      body: { org_id },
    }).then((r) => r.members),
  inviteMember: (org_id: string, email: string, role: string) =>
    request<{ member: Member; invite_token: string }>(
      `${CONTROL_URL}/v1/members/invite`,
      { method: 'POST', body: { org_id, email, role } },
    ),
  removeMember: (member_id: string) =>
    request<{ removed: boolean }>(`${CONTROL_URL}/v1/members/remove`, {
      method: 'POST',
      body: { member_id },
    }),
}

// ---------------------------------------------------------------------------
// Studio (data-plane) API — Bearer-JWT authenticated, workspace-scoped
// ---------------------------------------------------------------------------

export const studio = {
  overview: (token: string) =>
    request<Overview>(`${STUDIO_URL}/overview`, { token }),
  memory: (token: string, q: string, k: number) =>
    request<{ hits: MemoryHit[] }>(
      `${STUDIO_URL}/memory?q=${encodeURIComponent(q)}&k=${k}`,
      { token },
    ).then((r) => r.hits),
  cost: (token: string) => request<CostSummary>(`${STUDIO_URL}/cost`, { token }),
  evals: (token: string) =>
    request<{ runs: EvalRun[] }>(`${STUDIO_URL}/evals`, { token }).then(
      (r) => r.runs,
    ),
  datasets: (token: string) =>
    request<{ datasets: string[] }>(`${STUDIO_URL}/datasets`, { token }).then(
      (r) => r.datasets,
    ),
  documents: (token: string) =>
    request<{ documents: DocumentMeta[] }>(`${STUDIO_URL}/documents`, {
      token,
    }).then((r) => r.documents),
  replay: (token: string) =>
    request<{ sessions: ReplaySession[] }>(`${STUDIO_URL}/replay`, {
      token,
    }).then((r) => r.sessions),
  traces: (token: string, trace_id: string) =>
    request<{ spans: Span[] }>(
      `${STUDIO_URL}/traces?trace_id=${encodeURIComponent(trace_id)}`,
      { token },
    ).then((r) => r.spans),
  audit: (token: string, limit: number) =>
    request<{ events: AuditEvent[] }>(`${STUDIO_URL}/audit?limit=${limit}`, {
      token,
    }).then((r) => r.events),
}
