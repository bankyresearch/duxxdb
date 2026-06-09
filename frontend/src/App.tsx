import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import Console from './pages/Console'
import Studio from './pages/Studio'

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">DuxxDB Cloud</span>
        <nav className="topnav">
          <NavLink to="/console">Console</NavLink>
          <NavLink to="/studio">Studio</NavLink>
        </nav>
        <span className="spacer" />
        <span className="sub">managed AI-agent database</span>
      </header>
      <Routes>
        <Route path="/" element={<Navigate to="/console" replace />} />
        <Route path="/console" element={<Console />} />
        <Route path="/studio" element={<Studio />} />
        <Route path="*" element={<Navigate to="/console" replace />} />
      </Routes>
    </div>
  )
}
