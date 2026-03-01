export async function register() {
  // next-auth v4 calls localStorage.getItem during SSR, but Next.js 15 provides
  // an incomplete localStorage mock in the Node.js environment. This polyfill
  // ensures the mock has all required methods before next-auth tries to use them.
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    const g = global as typeof globalThis & { localStorage?: Storage };
    if (!g.localStorage || typeof g.localStorage.getItem !== 'function') {
      (g as any).localStorage = {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {},
        clear: () => {},
        length: 0,
        key: () => null,
      };
    }
  }
}
