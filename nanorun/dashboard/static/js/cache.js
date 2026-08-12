// Versioned persistent dashboard cache with a bounded IndexedDB LRU.

const DASHBOARD_CACHE_DB = 'nanorun-dashboard-cache';
const DASHBOARD_CACHE_VERSION = 1;
const DASHBOARD_CACHE_STORE = 'entries';
const DASHBOARD_CACHE_MAX_BYTES = 250 * 1024 * 1024;

const DashboardCache = (() => {
    const memory = new Map();
    let dbPromise = null;
    let persistent = true;
    let cachedTotalBytes = null;
    let mutationChain = Promise.resolve();

    function estimateSize(value) {
        try { return new Blob([JSON.stringify(value)]).size; }
        catch { return 0; }
    }

    function open() {
        if (!persistent || !window.indexedDB) {
            persistent = false;
            return Promise.resolve(null);
        }
        if (dbPromise) return dbPromise;
        dbPromise = new Promise(resolve => {
            let request;
            try { request = indexedDB.open(DASHBOARD_CACHE_DB, DASHBOARD_CACHE_VERSION); }
            catch {
                persistent = false;
                resolve(null);
                return;
            }
            request.onupgradeneeded = () => {
                const db = request.result;
                if (db.objectStoreNames.contains(DASHBOARD_CACHE_STORE)) {
                    db.deleteObjectStore(DASHBOARD_CACHE_STORE);
                }
                const store = db.createObjectStore(DASHBOARD_CACHE_STORE, { keyPath: 'key' });
                store.createIndex('accessed_at', 'accessed_at');
            };
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => {
                persistent = false;
                resolve(null);
            };
            request.onblocked = () => {
                persistent = false;
                resolve(null);
            };
        });
        return dbPromise;
    }

    async function get(key) {
        const db = await open();
        if (!db) return memory.get(key);
        return new Promise(resolve => {
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readwrite');
            const store = tx.objectStore(DASHBOARD_CACHE_STORE);
            const request = store.get(key);
            request.onsuccess = () => {
                const entry = request.result;
                if (entry) {
                    entry.accessed_at = Date.now();
                    store.put(entry);
                }
                resolve(entry ? entry.value : memory.get(key));
            };
            request.onerror = () => resolve(undefined);
        });
    }

    async function entries(prefix) {
        const db = await open();
        if (!db) {
            return Array.from(memory.entries())
                .filter(([key]) => key.startsWith(prefix))
                .map(([key, value]) => ({ key, value }));
        }
        return new Promise(resolve => {
            const result = [];
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readonly');
            const request = tx.objectStore(DASHBOARD_CACHE_STORE).openCursor();
            request.onsuccess = () => {
                const cursor = request.result;
                if (!cursor) {
                    const seen = new Set(result.map(entry => entry.key));
                    memory.forEach((value, key) => {
                        if (key.startsWith(prefix) && !seen.has(key)) result.push({ key, value });
                    });
                    resolve(result);
                    return;
                }
                if (cursor.value.key.startsWith(prefix)) {
                    result.push({ key: cursor.value.key, value: cursor.value.value });
                }
                cursor.continue();
            };
            request.onerror = () => resolve(result);
        });
    }

    async function evictOldest(db, bytesNeeded) {
        return new Promise(resolve => {
            let freed = 0;
            const target = Math.max(bytesNeeded, DASHBOARD_CACHE_MAX_BYTES * 0.1);
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readwrite');
            const index = tx.objectStore(DASHBOARD_CACHE_STORE).index('accessed_at');
            const request = index.openCursor();
            request.onsuccess = () => {
                const cursor = request.result;
                if (!cursor || freed >= target) return;
                freed += Number(cursor.value.size || 0);
                cursor.delete();
                cursor.continue();
            };
            tx.oncomplete = () => resolve(freed);
            tx.onerror = () => resolve();
        });
    }

    function existingSize(db, key) {
        return new Promise(resolve => {
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readonly');
            const request = tx.objectStore(DASHBOARD_CACHE_STORE).get(key);
            request.onsuccess = () => resolve(Number(request.result?.size || 0));
            request.onerror = () => resolve(0);
        });
    }

    async function totalSize(db) {
        return new Promise(resolve => {
            let total = 0;
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readonly');
            const request = tx.objectStore(DASHBOARD_CACHE_STORE).openCursor();
            request.onsuccess = () => {
                const cursor = request.result;
                if (!cursor) {
                    resolve(total);
                    return;
                }
                total += Number(cursor.value.size || 0);
                cursor.continue();
            };
            request.onerror = () => resolve(total);
        });
    }

    function putEntry(db, entry) {
        return new Promise((resolve, reject) => {
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readwrite');
            tx.objectStore(DASHBOARD_CACHE_STORE).put(entry);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
            tx.onabort = () => reject(tx.error);
        });
    }

    async function write(key, value) {
        const db = await open();
        if (!db) {
            memory.set(key, value);
            return;
        }
        const size = estimateSize(value);
        const entry = { key, value, size, accessed_at: Date.now() };
        if (cachedTotalBytes === null) cachedTotalBytes = await totalSize(db);
        const replacedSize = await existingSize(db, key);
        const projectedSize = cachedTotalBytes - replacedSize + size;
        if (projectedSize > DASHBOARD_CACHE_MAX_BYTES) {
            const freed = await evictOldest(
                db, projectedSize - DASHBOARD_CACHE_MAX_BYTES
            );
            cachedTotalBytes = Math.max(0, cachedTotalBytes - Number(freed || 0));
        }
        let storedSize = await existingSize(db, key);
        try {
            await putEntry(db, entry);
            cachedTotalBytes = Math.max(0, cachedTotalBytes - storedSize) + size;
        } catch (error) {
            if (error?.name !== 'QuotaExceededError') {
                memory.set(key, value);
                return;
            }
            // Quota handling is intentionally one retry: clear an LRU slice and
            // degrade to memory if the browser still cannot accept the write.
            const freed = await evictOldest(db, Math.max(size, DASHBOARD_CACHE_MAX_BYTES * 0.1));
            cachedTotalBytes = Math.max(0, cachedTotalBytes - Number(freed || 0));
            storedSize = await existingSize(db, key);
            try {
                await putEntry(db, entry);
                cachedTotalBytes = Math.max(0, cachedTotalBytes - storedSize) + size;
            } catch { memory.set(key, value); }
        }
    }

    function set(key, value) {
        mutationChain = mutationChain.then(() => write(key, value)).catch(() => {
            memory.set(key, value);
        });
        return mutationChain;
    }

    async function discard(key) {
        memory.delete(key);
        const db = await open();
        if (!db) return;
        const removedSize = await existingSize(db, key);
        await new Promise(resolve => {
            const tx = db.transaction(DASHBOARD_CACHE_STORE, 'readwrite');
            tx.objectStore(DASHBOARD_CACHE_STORE).delete(key);
            tx.oncomplete = resolve;
            tx.onerror = resolve;
        });
        if (cachedTotalBytes !== null) {
            cachedTotalBytes = Math.max(0, cachedTotalBytes - removedSize);
        }
    }

    function remove(key) {
        mutationChain = mutationChain.then(() => discard(key)).catch(() => {});
        return mutationChain;
    }

    return { open, get, entries, set, remove, get persistent() { return persistent; } };
})();
