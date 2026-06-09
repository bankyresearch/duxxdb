import { useEffect, useState } from 'react'
import { studio, errMsg } from '../lib/api'
import { useToast } from '../lib/toast'
import { Async, Empty, Pill, Table, fmtNs, useAsync } from '../components/ui'

const TABS = [
  'overview',
  'memory',
  'cost',
  'evals',
  'datasets',
  'documents',
  'replay',
  'traces',
  'audit',
] as const
type Tab = (typeof TABS)[number]

const STORE_KEY = 'duxx_jwt'

export default function Studio() {
  const toast = useToast()
  const [token, setToken] = useState(localStorage.getItem(STORE_KEY) ?? '')
  const [active, setActive] = useState('') // the connected token
  const [who, setWho] = useState('')
  const [tab, setTab] = useState<Tab>('overview')

  const connect = async (t: string) => {
    const tok = t.trim()
    if (!tok) return
    try {
      const o = await studio.overview(tok)
      setWho(`${o.tenant} · ${o.role}`)
      setActive(tok)
      localStorage.setItem(STORE_KEY, tok)
      setTab('overview')
    } catch (e) {
      setActive('')
      toast(errMsg(e), true)
    }
  }

  // Auto-connect once if a token is already stored.
  useEffect(() => {
    if (token) void connect(token)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="studio">
      <div className="auth">
        <input
          value={token}
          onChange={(e) => setToken(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && connect(token)}
          placeholder="Paste a workspace JWT (minted in the Console)"
        />
        <button onClick={() => connect(token)}>Connect</button>
        {active && <Pill kind="good">{who}</Pill>}
      </div>

      {!active ? (
        <Empty>Connect with a workspace JWT to view this workspace.</Empty>
      ) : (
        <>
          <div className="tabs wrap">
            {TABS.map((t) => (
              <button
                key={t}
                className={tab === t ? 'active' : ''}
                onClick={() => setTab(t)}
              >
                {t[0].toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
          <TabView tab={tab} token={active} />
        </>
      )}
    </div>
  )
}

function TabView({ tab, token }: { tab: Tab; token: string }) {
  switch (tab) {
    case 'overview':
      return <OverviewView token={token} />
    case 'memory':
      return <MemoryView token={token} />
    case 'cost':
      return <CostView token={token} />
    case 'evals':
      return <EvalsView token={token} />
    case 'datasets':
      return <DatasetsView token={token} />
    case 'documents':
      return <DocumentsView token={token} />
    case 'replay':
      return <ReplayView token={token} />
    case 'traces':
      return <TracesView token={token} />
    case 'audit':
      return <AuditView token={token} />
  }
}

function OverviewView({ token }: { token: string }) {
  const state = useAsync(() => studio.overview(token), [token])
  return (
    <Async state={state}>
      {(o) => (
        <div>
          <div>
            <span className="stat">
              <b>{o.memory_count}</b>
              <span>memories</span>
            </span>
            <span className="stat">
              <b>{o.prompts.length}</b>
              <span>prompts</span>
            </span>
            <span className="stat">
              <b className="mono">{o.tenant}</b>
              <span>workspace</span>
            </span>
          </div>
          <h3>Prompts</h3>
          {o.prompts.length === 0 ? (
            <Empty>none</Empty>
          ) : (
            <ul className="list">
              {o.prompts.map((p) => (
                <li key={p} className="mono">
                  {p}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Async>
  )
}

function MemoryView({ token }: { token: string }) {
  const [q, setQ] = useState('')
  const [k, setK] = useState(10)
  const [submitted, setSubmitted] = useState<{ q: string; k: number } | null>(
    null,
  )
  return (
    <div>
      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) =>
              e.key === 'Enter' && q.trim() && setSubmitted({ q: q.trim(), k })
            }
            placeholder="Search this workspace's memory…"
          />
        </div>
        <div className="field" style={{ maxWidth: 90 }}>
          <input
            type="number"
            value={k}
            onChange={(e) => setK(Number(e.target.value) || 10)}
          />
        </div>
        <button onClick={() => q.trim() && setSubmitted({ q: q.trim(), k })}>
          Recall
        </button>
      </div>
      {!submitted ? (
        <div className="muted">Enter a query to search.</div>
      ) : (
        <MemoryResults token={token} q={submitted.q} k={submitted.k} />
      )}
    </div>
  )
}

function MemoryResults({
  token,
  q,
  k,
}: {
  token: string
  q: string
  k: number
}) {
  const state = useAsync(() => studio.memory(token, q, k), [token, q, k])
  return (
    <Async state={state}>
      {(hits) => (
        <Table
          rows={hits}
          empty="no hits"
          cols={[
            { key: 'id', label: 'id', render: (h) => h.id },
            {
              key: 'score',
              label: 'score',
              render: (h) => h.score.toFixed(4),
            },
            { key: 'text', label: 'text', render: (h) => h.text },
          ]}
        />
      )}
    </Async>
  )
}

function CostView({ token }: { token: string }) {
  const state = useAsync(() => studio.cost(token), [token])
  return (
    <Async state={state}>
      {(c) => (
        <div>
          <span className="stat">
            <b>${Number(c.total_usd).toFixed(4)}</b>
            <span>total spend</span>
          </span>
          <h3>By model</h3>
          <Table
            rows={c.by_model}
            empty="no spend recorded"
            cols={[
              { key: 'key', label: 'model', render: (b) => b.key },
              { key: 'count', label: 'calls', render: (b) => b.count },
              { key: 'ti', label: 'tokens in', render: (b) => b.tokens_in },
              { key: 'to', label: 'tokens out', render: (b) => b.tokens_out },
              {
                key: 'total',
                label: 'total $',
                render: (b) => Number(b.total_usd).toFixed(4),
              },
              {
                key: 'mean',
                label: 'mean $',
                render: (b) => Number(b.mean_usd).toFixed(4),
              },
            ]}
          />
        </div>
      )}
    </Async>
  )
}

function EvalsView({ token }: { token: string }) {
  const state = useAsync(() => studio.evals(token), [token])
  return (
    <Async state={state}>
      {(runs) => (
        <Table
          rows={runs}
          empty="no eval runs"
          cols={[
            { key: 'id', label: 'id', render: (r) => r.id },
            { key: 'name', label: 'name', render: (r) => r.name },
            { key: 'model', label: 'model', render: (r) => r.model },
            { key: 'status', label: 'status', render: (r) => r.status },
            {
              key: 'dataset',
              label: 'dataset',
              render: (r) => r.dataset_name,
            },
          ]}
        />
      )}
    </Async>
  )
}

function DatasetsView({ token }: { token: string }) {
  const state = useAsync(() => studio.datasets(token), [token])
  return (
    <Async state={state}>
      {(ds) =>
        ds.length === 0 ? (
          <Empty>none</Empty>
        ) : (
          <ul className="list">
            {ds.map((d) => (
              <li key={d} className="mono">
                {d}
              </li>
            ))}
          </ul>
        )
      }
    </Async>
  )
}

function DocumentsView({ token }: { token: string }) {
  const state = useAsync(() => studio.documents(token), [token])
  return (
    <Async state={state}>
      {(docs) => (
        <Table
          rows={docs}
          empty="no documents ingested"
          cols={[
            { key: 'uri', label: 'source', render: (d) => d.uri },
            { key: 'ct', label: 'type', render: (d) => d.content_type },
            { key: 'version', label: 'version', render: (d) => d.version },
            { key: 'chunks', label: 'chunks', render: (d) => d.chunks },
          ]}
        />
      )}
    </Async>
  )
}

function ReplayView({ token }: { token: string }) {
  const state = useAsync(() => studio.replay(token), [token])
  return (
    <Async state={state}>
      {(sessions) => (
        <Table
          rows={sessions}
          empty="no replay sessions"
          cols={[
            { key: 'trace', label: 'trace_id', render: (s) => s.trace_id },
            {
              key: 'at',
              label: 'captured',
              render: (s) => fmtNs(s.captured_at_unix_ns),
            },
          ]}
        />
      )}
    </Async>
  )
}

function TracesView({ token }: { token: string }) {
  const [tid, setTid] = useState('')
  const [submitted, setSubmitted] = useState('')
  return (
    <div>
      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <input
            value={tid}
            onChange={(e) => setTid(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && setSubmitted(tid.trim())}
            placeholder="trace_id…"
          />
        </div>
        <button onClick={() => setSubmitted(tid.trim())}>Load trace</button>
      </div>
      {!submitted ? (
        <div className="muted">Enter a trace_id to view its spans.</div>
      ) : (
        <TraceSpans token={token} traceId={submitted} />
      )}
    </div>
  )
}

function TraceSpans({ token, traceId }: { token: string; traceId: string }) {
  const state = useAsync(() => studio.traces(token, traceId), [token, traceId])
  return (
    <Async state={state}>
      {(spans) =>
        spans.length === 0 ? (
          <Empty>no spans for that trace_id</Empty>
        ) : (
          <pre>{JSON.stringify(spans, null, 2)}</pre>
        )
      }
    </Async>
  )
}

function AuditView({ token }: { token: string }) {
  const state = useAsync(() => studio.audit(token, 200), [token])
  return (
    <Async state={state}>
      {(events) => (
        <Table
          rows={events}
          empty="no audit events"
          cols={[
            { key: 'ts', label: 'time', render: (e) => fmtNs(e.ts_unix_ns) },
            {
              key: 'principal',
              label: 'principal',
              render: (e) => e.principal,
            },
            { key: 'command', label: 'command', render: (e) => e.command },
            {
              key: 'outcome',
              label: 'outcome',
              render: (e) =>
                e.outcome === 'ok' ? (
                  <Pill kind="good">ok</Pill>
                ) : (
                  <Pill kind="bad">{e.outcome}</Pill>
                ),
            },
          ]}
        />
      )}
    </Async>
  )
}
