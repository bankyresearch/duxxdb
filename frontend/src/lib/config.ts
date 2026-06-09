// API base URLs. Default to the dev-server proxy paths (see vite.config.ts).
// Override in production via VITE_CONTROL_URL / VITE_STUDIO_URL.
export const CONTROL_URL =
  (import.meta.env.VITE_CONTROL_URL as string | undefined) ?? '/api/control'
export const STUDIO_URL =
  (import.meta.env.VITE_STUDIO_URL as string | undefined) ?? '/api/studio'
