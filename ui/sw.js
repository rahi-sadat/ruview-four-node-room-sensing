// RuView live-demo service worker reset.
// This project view changes often during pose debugging, so the safest behavior
// is to clear old offline caches and let the browser fetch fresh UI modules.

self.addEventListener('install', (event) => {
  event.waitUntil(clearAllCaches());
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    clearAllCaches()
      .then(() => self.clients.claim())
      .then(() => self.registration.unregister())
  );
});

// No respondWith(): every request passes through to the network/browser normally.
self.addEventListener('fetch', () => {});

async function clearAllCaches() {
  if (!self.caches?.keys) return;
  const keys = await caches.keys();
  await Promise.all(keys.map(key => caches.delete(key)));
}
