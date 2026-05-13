// Centralized state manager for nanorun dashboard
// All persisted and session state lives here.

function createStateManager(schema) {
    const STORAGE_KEY = 'nanorun_state';
    let cache = {};
    const listeners = {};

    // Load persisted state from localStorage
    try {
        cache = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    } catch { cache = {}; }

    function get(key, subkey) {
        const def = schema[key];
        if (!def) return undefined;
        if (def.keyed && subkey !== undefined) {
            const bucket = cache[key];
            if (!bucket || typeof bucket !== 'object') return structuredClone(def.default);
            return bucket[subkey] !== undefined ? bucket[subkey] : structuredClone(def.default);
        }
        return cache[key] !== undefined ? cache[key] : structuredClone(def.default);
    }

    function set(key, value, subkey) {
        const def = schema[key];
        if (!def) return;
        if (def.keyed && subkey !== undefined) {
            if (!cache[key] || typeof cache[key] !== 'object') cache[key] = {};
            cache[key][subkey] = value;
        } else {
            cache[key] = value;
        }
        if (def.persist) save();
        notify(key, value);
    }

    function update(patch) {
        let needsSave = false;
        for (const [k, v] of Object.entries(patch)) {
            const def = schema[k];
            if (!def) continue;
            cache[k] = v;
            if (def.persist) needsSave = true;
            notify(k, v);
        }
        if (needsSave) save();
    }

    function save() {
        const persisted = {};
        for (const [k, def] of Object.entries(schema)) {
            if (def.persist && cache[k] !== undefined) {
                persisted[k] = cache[k];
            }
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    }

    function on(key, fn) {
        if (!listeners[key]) listeners[key] = [];
        listeners[key].push(fn);
    }

    function off(key, fn) {
        if (!listeners[key]) return;
        listeners[key] = listeners[key].filter(f => f !== fn);
    }

    function notify(key, val) {
        (listeners[key] || []).forEach(fn => fn(val));
    }

    return { get, set, update, on, off };
}

const State = createStateManager({
    // Global preferences (persist always)
    theme: { default: 'dark', persist: true },
    sidebarCollapsed: { default: false, persist: true },
    bucket: { default: [], persist: true },

    // Navigation state (persist so reloads restore position)
    view: { default: 'experiments', persist: true },
    selectedExp: { default: null, persist: true },
    selectedRun: { default: null, persist: true },
    chartView: { default: null, persist: true },
    tab: { default: 'diff', persist: true },

    // Per-experiment heatmap defaults (keyed by code_hash)
    heatmapDefaults: { default: {}, persist: true, keyed: true },

    // Session-only state (not persisted — resets on reload)
    experimentList: { default: [] },
    experimentData: { default: null },
    selectedExpIds: { default: null },
    selectedCellExpIds: { default: null },
    runsSort: { default: { col: 'started_at', dir: 'desc' } },
    runsEnvFilters: { default: {} },
    runsLimit: { default: 20 },
    chartXRange: { default: null },
    heatmapSelectedVars: { default: [] },
    currentTotalSteps: { default: 0 },
    selectedQueuedScript: { default: null },
    deleteInProgress: { default: false },
});
