import { useCallback, useEffect, useState } from 'react'
import {
  control,
  errMsg,
  type ApiKey,
  type Member,
  type Org,
  type Project,
  type Usage,
} from '../lib/api'
import { useToast } from '../lib/toast'
import { Empty, Pill, copy, shortId } from '../components/ui'

export default function Console() {
  const toast = useToast()
  const [orgs, setOrgs] = useState<Org[]>([])
  const [org, setOrg] = useState<Org | null>(null)
  const [tab, setTab] = useState<'projects' | 'members'>('projects')
  const [newOrg, setNewOrg] = useState('')

  const loadOrgs = useCallback(async () => {
    try {
      setOrgs(await control.listOrgs())
    } catch (e) {
      toast(errMsg(e), true)
    }
  }, [toast])
  useEffect(() => {
    void loadOrgs()
  }, [loadOrgs])

  const createOrg = async () => {
    if (!newOrg.trim()) return
    try {
      const o = await control.createOrg(newOrg.trim())
      setNewOrg('')
      await loadOrgs()
      setOrg(o)
      setTab('projects')
      toast('Organization created')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>Organizations</h2>
        <div className="newrow">
          <input
            value={newOrg}
            onChange={(e) => setNewOrg(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && createOrg()}
            placeholder="New org name"
          />
          <button onClick={createOrg}>Create</button>
        </div>
        <ul className="list">
          {orgs.length === 0 && <li className="muted">no orgs yet</li>}
          {orgs.map((o) => (
            <li
              key={o.id}
              className={org?.id === o.id ? 'active' : ''}
              onClick={() => {
                setOrg(o)
                setTab('projects')
              }}
            >
              <span>{o.name}</span>
              <span className="mono muted small">{shortId(o.id)}</span>
            </li>
          ))}
        </ul>
      </aside>

      <main>
        {!org ? (
          <Empty>Select or create an organization to begin.</Empty>
        ) : (
          <>
            <div className="org-head">
              <h2 className="m0">{org.name}</h2>
              <span className="mono muted small">{org.id}</span>
            </div>
            <div className="tabs">
              <button
                className={tab === 'projects' ? 'active' : ''}
                onClick={() => setTab('projects')}
              >
                Projects
              </button>
              <button
                className={tab === 'members' ? 'active' : ''}
                onClick={() => setTab('members')}
              >
                Members
              </button>
            </div>
            {tab === 'projects' ? (
              <ProjectsTab org={org} />
            ) : (
              <MembersTab org={org} />
            )}
          </>
        )}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

function ProjectsTab({ org }: { org: Org }) {
  const toast = useToast()
  const [projects, setProjects] = useState<Project[]>([])
  const [sel, setSel] = useState<Project | null>(null)
  const [name, setName] = useState('')

  const load = useCallback(async () => {
    try {
      setProjects(await control.listProjects(org.id))
    } catch (e) {
      toast(errMsg(e), true)
    }
  }, [org.id, toast])
  useEffect(() => {
    setSel(null)
    void load()
  }, [load])

  const create = async () => {
    if (!name.trim()) return
    try {
      const p = await control.createProject(org.id, name.trim())
      setName('')
      await load()
      setSel(p)
      toast('Project created')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }

  return (
    <>
      <div className="card">
        <div className="newrow">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && create()}
            placeholder="New project name"
          />
          <button onClick={create}>Create project</button>
        </div>
        {projects.length === 0 ? (
          <Empty>No projects yet.</Empty>
        ) : (
          <ul className="list">
            {projects.map((p) => (
              <li
                key={p.id}
                className={sel?.id === p.id ? 'active' : ''}
                onClick={() => setSel(p)}
              >
                <span>{p.name}</span>
                <span className="mono muted small">{shortId(p.id)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      {sel && <ProjectDetail project={sel} />}
    </>
  )
}

function ProjectDetail({ project }: { project: Project }) {
  const toast = useToast()
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [usage, setUsage] = useState<Usage | null>(null)
  const [env, setEnv] = useState('prod')
  const [role, setRole] = useState('service')
  const [keyName, setKeyName] = useState('')
  const [secret, setSecret] = useState<string | null>(null)
  const [jwt, setJwt] = useState<string | null>(null)
  const [node, setNode] = useState('127.0.0.1:6380')
  const [mode, setMode] = useState('shared')
  const [entries, setEntries] = useState<number | null>(null)

  const load = useCallback(async () => {
    try {
      const [k, u] = await Promise.all([
        control.listKeys(project.id),
        control.usage(project.id),
      ])
      setKeys(k)
      setUsage(u)
    } catch (e) {
      toast(errMsg(e), true)
    }
  }, [project.id, toast])
  useEffect(() => {
    setSecret(null)
    setJwt(null)
    void load()
  }, [load])

  const issue = async () => {
    try {
      const k = await control.issueKey(
        project.id,
        env,
        role,
        keyName.trim() || 'key',
      )
      setSecret(k.secret ?? null)
      setJwt(null)
      setKeyName('')
      await load()
      toast('Key issued — copy the secret now')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }
  const revoke = async (id: string) => {
    if (!confirm('Revoke this key? Clients using it are denied immediately.'))
      return
    try {
      await control.revokeKey(id)
      await load()
      toast('Key revoked')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }
  const rotate = async (id: string) => {
    if (!confirm('Rotate this key? The current secret stops working immediately.'))
      return
    try {
      const k = await control.rotateKey(id)
      setSecret(k.secret ?? null)
      setJwt(null)
      await load()
      toast('Key rotated')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }
  const mint = async () => {
    if (!secret) return
    try {
      const r = await control.mintToken(secret, 900)
      setJwt(r.jwt)
      toast('Workspace JWT minted')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }
  const place = async () => {
    if (!node.trim()) return
    try {
      await control.placeProject(project.id, node.trim(), mode)
      const e = await control.authEntries(node.trim())
      setEntries(e.length)
      toast('Project placed on ' + node.trim())
    } catch (e) {
      toast(errMsg(e), true)
    }
  }

  return (
    <div className="card">
      <h3 className="card-title">Project · {project.name}</h3>
      <div className="grid2">
        <div>
          <label>Issue API key</label>
          <div className="row">
            <div className="field">
              <label>Env</label>
              <select value={env} onChange={(e) => setEnv(e.target.value)}>
                <option>prod</option>
                <option>staging</option>
                <option>dev</option>
              </select>
            </div>
            <div className="field">
              <label>Role</label>
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                {['service', 'developer', 'observer', 'evaluator', 'admin', 'owner'].map(
                  (r) => (
                    <option key={r}>{r}</option>
                  ),
                )}
              </select>
            </div>
          </div>
          <div className="field mt8">
            <input
              value={keyName}
              onChange={(e) => setKeyName(e.target.value)}
              placeholder="Key name (e.g. agent-runtime)"
            />
          </div>
          <button onClick={issue}>Issue key</button>
          {secret && (
            <div className="reveal mt10">
              <div>
                Secret (shown once): <span className="mono">{secret}</span>
              </div>
              <div className="row mt8">
                <button className="ghost sm" onClick={() => copy(secret)}>
                  Copy secret
                </button>
                <button className="sm" onClick={mint}>
                  Mint workspace JWT →
                </button>
              </div>
              {jwt && (
                <div className="reveal accent mt10">
                  Workspace JWT (15 min): <span className="mono">{jwt}</span>
                  <div className="mt8">
                    <button className="sm" onClick={() => copy(jwt)}>
                      Copy for Studio
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div>
          <label>Placement</label>
          <div className="row">
            <div className="field">
              <input
                value={node}
                onChange={(e) => setNode(e.target.value)}
                placeholder="node host:port"
              />
            </div>
            <div className="field maxw140">
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option>shared</option>
                <option>dedicated</option>
              </select>
            </div>
            <button className="ghost" onClick={place}>
              Place
            </button>
          </div>
          {entries !== null && (
            <div className="small muted mt8">
              {entries} auth-key line(s) for <span className="mono">{node}</span>
            </div>
          )}
          <label className="mt16">Usage</label>
          <div>
            <span className="stat">
              <b>{usage?.requests ?? 0}</b>
              <span>requests</span>
            </span>
            <span className="stat">
              <b>{(usage?.tokens_in ?? 0) + (usage?.tokens_out ?? 0)}</b>
              <span>tokens</span>
            </span>
            <span className="stat">
              <b>${Number(usage?.cost_usd ?? 0).toFixed(4)}</b>
              <span>spend</span>
            </span>
          </div>
        </div>
      </div>

      <h3>Keys</h3>
      {keys.length === 0 ? (
        <Empty>No keys issued.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Role</th>
              <th>Env</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id}>
                <td>{k.name || <span className="muted">(unnamed)</span>}</td>
                <td>
                  <Pill>{k.role}</Pill>
                </td>
                <td>
                  <Pill>{k.env}</Pill>
                </td>
                <td>
                  {k.revoked ? (
                    <Pill kind="bad">revoked</Pill>
                  ) : (
                    <Pill kind="good">active</Pill>
                  )}
                </td>
                <td>
                  <div className="actions">
                    {!k.revoked && (
                      <>
                        <button
                          className="ghost sm"
                          onClick={() => rotate(k.id)}
                        >
                          Rotate
                        </button>
                        <button
                          className="danger sm"
                          onClick={() => revoke(k.id)}
                        >
                          Revoke
                        </button>
                      </>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Members
// ---------------------------------------------------------------------------

function MembersTab({ org }: { org: Org }) {
  const toast = useToast()
  const [members, setMembers] = useState<Member[]>([])
  const [email, setEmail] = useState('')
  const [role, setRole] = useState('developer')
  const [invite, setInvite] = useState<{ email: string; token: string } | null>(
    null,
  )

  const load = useCallback(async () => {
    try {
      setMembers(await control.listMembers(org.id))
    } catch (e) {
      toast(errMsg(e), true)
    }
  }, [org.id, toast])
  useEffect(() => {
    setInvite(null)
    void load()
  }, [load])

  const doInvite = async () => {
    if (!email.trim()) return
    try {
      const r = await control.inviteMember(org.id, email.trim(), role)
      setInvite({ email: email.trim(), token: r.invite_token })
      setEmail('')
      await load()
      toast('Invite created')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }
  const remove = async (id: string, em: string) => {
    if (!confirm('Remove ' + em + ' from this organization?')) return
    try {
      await control.removeMember(id)
      await load()
      toast('Member removed')
    } catch (e) {
      toast(errMsg(e), true)
    }
  }

  return (
    <div className="card">
      <label>Invite a member</label>
      <div className="row">
        <div className="field">
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="email@company.com"
          />
        </div>
        <div className="field maxw170">
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            {['developer', 'admin', 'observer', 'evaluator', 'owner'].map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </div>
        <button onClick={doInvite}>Send invite</button>
      </div>
      {invite && (
        <div className="reveal mt10">
          Invite token for {invite.email}:{' '}
          <span className="mono">{invite.token}</span>
          <div className="mt8">
            <button className="ghost sm" onClick={() => copy(invite.token)}>
              Copy invite
            </button>
          </div>
        </div>
      )}
      <h3>Members</h3>
      {members.length === 0 ? (
        <Empty>No members yet.</Empty>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Email</th>
              <th>Role</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {members.map((m) => (
              <tr key={m.id}>
                <td>{m.email}</td>
                <td>
                  <Pill>{m.role}</Pill>
                </td>
                <td>
                  {m.status === 'Active' ? (
                    <Pill kind="good">active</Pill>
                  ) : (
                    <Pill kind="warn">invited</Pill>
                  )}
                </td>
                <td>
                  <div className="actions">
                    <button
                      className="danger sm"
                      onClick={() => remove(m.id, m.email)}
                    >
                      Remove
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
