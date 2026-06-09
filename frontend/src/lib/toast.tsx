import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from 'react'

interface ToastItem {
  id: number
  msg: string
  bad?: boolean
}

type Push = (msg: string, bad?: boolean) => void

const ToastCtx = createContext<Push>(() => {})

let seq = 0

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const push = useCallback<Push>((msg, bad) => {
    const id = ++seq
    setItems((t) => [...t, { id, msg, bad }])
    setTimeout(() => setItems((t) => t.filter((x) => x.id !== id)), 2600)
  }, [])
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div key={t.id} className={'toast' + (t.bad ? ' bad' : '')}>
            {t.msg}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  )
}

export const useToast = (): Push => useContext(ToastCtx)
