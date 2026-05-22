// Prerender disabled because pages fetch the backend at request time
// and use sessionStorage; the static adapter still serves them as SPA
// via the fallback (index.html) route.
export const prerender = false;
export const ssr = false;
