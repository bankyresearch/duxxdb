import { useEffect, useState, type ReactNode } from 'react'
import { errMsg } from '../lib/api'

export const Card = ({
  title,
  children,
}: {
  title?: string
  children: ReactNode
}) => (
  <div className="card">
    {title && <h3 className="card-title">{title}</h3>}
    {children}
  </div>
)

export const Pill = ({
  kind,
  children,
}: {
  kind?: 'good' | 'bad' | 'warn'
  children: ReactNode
}) => <span className={'pill' + (kind ? ' ' + kind : '')}>{children}</span>

export const Empty = ({ children }: { children: ReactNode }) => (
  <div className="empty">{children}</div>
)

export function copy(text: string) {
  void navigator.clipboard.writeText(text)
}

export function shortId(id: string, n = 12): string {
  return id.length > n ? id.slice(0, n) + '…' : id
}

/** Format unix-nanoseconds as a local datetime. */
export function fmtNs(ns: number): string {
  if (!ns) return ''
  const ms = Math.floor(ns / 1_000_000)
  return new Date(ms).toLocaleString()
}

interface AsyncState<T> {
  loading: boolean
  data?: T
  error?: string
}

/** Run an async fetch on mount / when `deps` change, tracking load + error. */
export function useAsync<T>(
  fn: () => Promise<T>,
  deps: unknown[],
): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: true })
  useEffect(() => {
    let live = true
    setState({ loading: true })
    fn()
      .then((data) => live && setState({ loading: false, data }))
      .catch((e) => live && setState({ loading: false, error: errMsg(e) }))
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

/** Render the loading / error / data states of a useAsync result. */
export function Async<T>({
  state,
  children,
}: {
  state: AsyncState<T>
  children: (data: T) => ReactNode
}) {
  if (state.loading) return <div className="muted">Loading…</div>
  if (state.error) return <div className="err">{state.error}</div>
  if (state.data === undefined) return null
  return <>{children(state.data)}</>
}

export function Table<T>({
  rows,
  cols,
  empty = 'none',
}: {
  rows: T[]
  cols: { key: string; label: string; render: (row: T) => ReactNode }[]
  empty?: string
}) {
  if (rows.length === 0) return <Empty>{empty}</Empty>
  return (
    <table>
      <thead>
        <tr>
          {cols.map((c) => (
            <th key={c.key}>{c.label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {cols.map((c) => (
              <td key={c.key} className="mono">
                {c.render(row)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
