// Durable dashboard snapshot + Server-Sent Events state reducer.

let _dashboardExperimentSummaries = new Map();
let _dashboardQueueData = null;
let _dashboardLastEventId = 0;
let _dashboardQueryCursor = null;
let _dashboardEventSource = null;
let _dashboardSnapshotReady = false;
let _dashboardResetInProgress = false;
const _dashboardEntityRevisions = new Map();
const _dashboardMetricRevisions = new Map();
const _dashboardCurveRevisions = new Map();
let _dashboardDiscoveryMembership = new Set();
let _dashboardPriorDiscoveryMembership = new Set();
let _dashboardTracks = [];
const _dashboardLatencySamples = [];
window.nanorunDashboardSSE = {
    connected: false,
    latencySamplesMs: _dashboardLatencySamples,
};

function _recordDashboardEventLatency(payload) {
    const committedAt = Date.parse(payload.committed_at || '');
    if (!Number.isFinite(committedAt)) return;
    _dashboardLatencySamples.push(Math.max(0, Date.now() - committedAt));
    if (_dashboardLatencySamples.length > 200) _dashboardLatencySamples.shift();
}

function dashboardGroupIdentity(exp) {
    return exp.group || {
        code_hash: exp.code_hash || null,
        track: exp.track || '',
        gpus: exp.gpus || 1,
        gpu_type: exp.gpu_type || 'H100',
    };
}

function dashboardGroupKey(exp) {
    const group = dashboardGroupIdentity(exp);
    return JSON.stringify([
        group.code_hash || `_no_hash_${exp.id}`,
        group.track || '',
        group.gpus || 1,
        group.gpu_type || 'H100',
    ]);
}

function getDashboardExperimentGroups() {
    const track = document.getElementById('track-filter')?.value || '';
    const status = document.getElementById('status-filter')?.value || '';
    const search = (document.getElementById('search-filter')?.value || '').trim().toLowerCase();
    const source = _dashboardDiscoveryMembership.size > 0
        ? Array.from(_dashboardDiscoveryMembership)
            .map(id => _dashboardExperimentSummaries.get(Number(id)))
            .filter(Boolean)
        : Array.from(_dashboardExperimentSummaries.values());
    const summaries = source
        .filter(exp => !track || exp.track === track)
        .filter(exp => !status || exp.status === status)
        .filter(exp => {
            if (!search) return true;
            return [exp.name, exp.script, exp.track, exp.code_hash]
                .some(value => String(value || '').toLowerCase().includes(search));
        });

    const grouped = new Map();
    summaries.forEach(exp => {
        const key = dashboardGroupKey(exp);
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(exp);
    });

    const results = [];
    grouped.forEach(groupExps => {
        groupExps.sort((a, b) => String(b.started_at || '').localeCompare(String(a.started_at || '')));
        const primary = groupExps[0];
        const losses = groupExps.map(exp => exp.loss).filter(value => value != null);
        const trainTimes = groupExps.map(exp => exp.train_time_ms).filter(value => value != null);
        const lossMetrics = new Set(groupExps.filter(exp => exp.loss != null).map(exp => exp.loss_metric));
        const statuses = groupExps.map(exp => exp.status);
        const withMetrics = groupExps.find(exp => exp.current_step != null);
        const aggregateStatus = statuses.includes('running')
            ? 'running'
            : (statuses.includes('completed') ? 'completed' : (statuses[0] || 'unknown'));
        results.push({
            id: primary.id,
            experiment_ids: groupExps.map(exp => exp.id),
            name: primary.name,
            track: primary.track,
            script: primary.script,
            code_hash: primary.code_hash,
            status: aggregateStatus,
            gpus: primary.gpus,
            gpu_type: primary.gpu_type,
            env_vars: primary.env_vars,
            started_at: primary.started_at,
            n_runs: groupExps.length,
            is_sweep: new Set(groupExps.map(exp => JSON.stringify(exp.env_vars || {}))).size > 1,
            current_step: withMetrics?.current_step ?? null,
            total_steps: withMetrics?.total_steps ?? null,
            val_loss: losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : null,
            loss: losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : null,
            loss_metric: lossMetrics.size === 1 ? Array.from(lossMetrics)[0] : null,
            train_time_ms: trainTimes.length ? trainTimes.reduce((a, b) => a + b, 0) / trainTimes.length : null,
            val_losses: losses,
            losses,
            train_times: trainTimes,
            group: dashboardGroupIdentity(primary),
            group_key: dashboardGroupKey(primary),
            revision: Math.max(...groupExps.map(exp => exp.revision || 0)),
        });
    });
    results.sort((a, b) => String(b.started_at || '').localeCompare(String(a.started_at || '')));
    return results.slice(0, 100);
}

function getDashboardQueueData() {
    return _dashboardQueueData;
}

function _installDashboardSnapshot(snapshot) {
    _dashboardLastEventId = Math.max(_dashboardLastEventId, Number(snapshot.last_event_id || 0));
    _dashboardQueueData = snapshot.queue || { running: null, running_list: [], queued: [], state: 'active' };
    _sessionData = snapshot.sessions || [];
    _hubData = snapshot.hub || {};
    _dashboardTracks = snapshot.tracks || _dashboardTracks;
    _dashboardSnapshotReady = true;
    DashboardCache.set('dashboard:shell', {
        queue: _dashboardQueueData,
        sessions: _sessionData,
        hub: _hubData,
        tracks: _dashboardTracks,
        event_cursor: _dashboardLastEventId,
        discovery_membership: Array.from(_dashboardDiscoveryMembership),
    });
}

async function loadDashboardSnapshot() {
    const response = await fetch('/api/dashboard/snapshot');
    if (!response.ok) throw new Error(`Snapshot failed (${response.status})`);
    const snapshot = await response.json();
    _installDashboardSnapshot(snapshot);
    return snapshot;
}

async function restoreDashboardCache() {
    const [shell, cachedGroups, summaryEntries, detailEntries] = await Promise.all([
        DashboardCache.get('dashboard:shell'),
        DashboardCache.get('sidebar:groups'),
        DashboardCache.entries('experiment:'),
        DashboardCache.entries('curve:'),
    ]);
    if (shell) {
        _dashboardQueueData = shell.queue || _dashboardQueueData;
        _sessionData = shell.sessions || _sessionData;
        _hubData = shell.hub || _hubData;
        _dashboardTracks = shell.tracks || [];
        _dashboardLastEventId = Number(shell.event_cursor || 0);
        _dashboardDiscoveryMembership = new Set(
            (shell.discovery_membership || []).map(Number)
        );
    }
    summaryEntries.forEach(({ value: summary }) => {
        if (!summary?.id) return;
        const id = Number(summary.id);
        _dashboardExperimentSummaries.set(id, summary);
        _dashboardEntityRevisions.set(`experiment:${id}`, Number(summary.revision || 0));
        _dashboardMetricRevisions.set(id, Number(summary.metrics_revision || 0));
    });
    const details = new Map();
    detailEntries.forEach(({ value }) => {
        if (value?.id) {
            const id = Number(value.id);
            const revision = Number(value.metrics_revision || 0);
            const prior = details.get(id);
            if (!prior || Number(prior.metrics_revision || 0) <= revision) {
                details.set(id, value);
                _dashboardCurveRevisions.set(id, revision);
            }
        }
    });
    if (summaryEntries.length || shell) _dashboardSnapshotReady = true;
    return { shell, groups: cachedGroups || [], details };
}

function _knownExperimentCacheState() {
    const experiments = {};
    const metrics = {};
    _dashboardExperimentSummaries.forEach((summary, id) => {
        experiments[id] = Number(summary.revision || 0);
        if (_dashboardCurveRevisions.has(Number(id))) {
            metrics[id] = Number(_dashboardCurveRevisions.get(Number(id)) || 0);
        }
    });
    return { experiments, metrics };
}

async function streamExperimentQueries(queries, { signal = null, onFrame = null } = {}) {
    const response = await fetch('/api/experiments/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/x-ndjson' },
        body: JSON.stringify({ queries, cache_state: _knownExperimentCacheState() }),
        signal,
    });
    if (!response.ok) {
        let message = `Experiment query failed (${response.status})`;
        try { message = (await response.json()).error || message; } catch {}
        throw new Error(message);
    }
    const frames = [];
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffered = '';
    while (true) {
        const { value, done } = await reader.read();
        buffered += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffered.split('\n');
        buffered = done ? '' : lines.pop();
        for (const line of lines) {
            if (!line.trim()) continue;
            const frame = JSON.parse(line);
            frames.push(frame);
            await applyExperimentQueryFrame(frame);
            if (onFrame) await onFrame(frame);
        }
        if (done) break;
    }
    if (buffered.trim()) {
        const frame = JSON.parse(buffered);
        frames.push(frame);
        await applyExperimentQueryFrame(frame);
        if (onFrame) await onFrame(frame);
    }
    return frames;
}

async function applyExperimentQueryFrame(frame) {
    if (frame.type === 'metadata') {
        _dashboardQueryCursor = Number(frame.event_cursor || 0);
        _dashboardLastEventId = Math.max(
            _dashboardLastEventId, Number(frame.event_cursor || 0)
        );
        return;
    }
    if (frame.type === 'group_page') {
        if (frame.query !== 'sidebar') return;
        const ids = (frame.groups || []).flatMap(group => group.experiment_ids || [group.id]).map(Number);
        _dashboardPriorDiscoveryMembership = new Set(_dashboardDiscoveryMembership);
        _dashboardDiscoveryMembership = new Set(ids);
        renderExperimentListData(frame.groups || []);
        DashboardCache.set('sidebar:groups', frame.groups || []);
        return;
    }
    if (frame.type === 'experiment') {
        const incoming = frame.experiment;
        const id = Number(incoming.id);
        const previous = _dashboardExperimentSummaries.get(id);
        const previousRevision = Number(previous?.revision || -1);
        if (Number(incoming.revision || 0) < previousRevision) return;
        const merged = {
            ...(previous || {}),
            ...incoming,
            loss_curves: {
                ...(previous?.loss_curves || {}),
                ...(incoming.loss_curves || {}),
            },
        };
        _dashboardExperimentSummaries.set(id, merged);
        _dashboardEntityRevisions.set(`experiment:${id}`, Number(merged.revision || 0));
        _dashboardMetricRevisions.set(id, Number(merged.metrics_revision || 0));
        const cachedSummary = { ...merged };
        delete cachedSummary.loss_curves;
        delete cachedSummary.loss_curve;
        delete cachedSummary.curve_max_points;
        DashboardCache.set(`experiment:${id}`, cachedSummary);
        if (incoming.loss_curves) {
            _dashboardCurveRevisions.set(id, Number(merged.metrics_revision || 0));
            DashboardCache.set(
                `curve:${id}:${Number(merged.metrics_revision || 0)}`, merged
            );
        }
        return;
    }
    if (frame.type === 'complete' && frame.query === 'sidebar') {
        const authoritative = new Set((frame.experiment_ids || []).map(Number));
        const removed = Array.from(_dashboardPriorDiscoveryMembership)
            .filter(id => !authoritative.has(Number(id)));
        _dashboardDiscoveryMembership = authoritative;
        for (const id of removed) {
            _dashboardExperimentSummaries.delete(Number(id));
            _dashboardEntityRevisions.delete(`experiment:${id}`);
            _dashboardMetricRevisions.delete(Number(id));
            await DashboardCache.remove(`experiment:${id}`);
        }
        const shell = await DashboardCache.get('dashboard:shell') || {};
        shell.discovery_membership = Array.from(authoritative);
        shell.event_cursor = _dashboardLastEventId;
        await DashboardCache.set('dashboard:shell', shell);
    }
}

function _eventRevision(event, payload) {
    return Number(payload.revision || payload.summary?.revision || 0);
}

function _eventCursor(event, payload) {
    return Number(event.lastEventId || payload.event_cursor || 0);
}

function _acceptEntityRevision(entityKey, revision) {
    const previous = _dashboardEntityRevisions.has(entityKey)
        ? Number(_dashboardEntityRevisions.get(entityKey)) : -1;
    if (revision <= previous) return false;
    _dashboardEntityRevisions.set(entityKey, revision);
    return true;
}

function _onExperimentEvent(eventType, event) {
    const payload = JSON.parse(event.data || '{}');
    const revision = _eventRevision(event, payload);
    const experimentId = Number(payload.experiment_id);
    if (!experimentId || !_acceptEntityRevision(`experiment:${experimentId}`, revision)) return;

    if (eventType === 'experiment.deleted') {
        _dashboardExperimentSummaries.delete(experimentId);
        _dashboardMetricRevisions.delete(experimentId);
        _dashboardCurveRevisions.delete(experimentId);
        DashboardCache.remove(`experiment:${experimentId}`);
    } else if (payload.summary) {
        _dashboardExperimentSummaries.set(experimentId, payload.summary);
        _dashboardMetricRevisions.set(
            experimentId, Number(payload.metrics_revision || payload.summary.metrics_revision || 0)
        );
        DashboardCache.set(`experiment:${experimentId}`, payload.summary);
        if (_dashboardQueueData) {
            const running = (_dashboardQueueData.running_list || [])
                .filter(item => Number(item.id) !== experimentId);
            if (payload.summary.status === 'running') {
                const prior = (_dashboardQueueData.running_list || [])
                    .find(item => Number(item.id) === experimentId) || {};
                running.push({ ...prior, ...payload.summary });
            }
            _dashboardQueueData.running_list = running;
            _dashboardQueueData.running = running.length === 1 ? running[0] : null;
            renderQueueData(_dashboardQueueData);
        }
    }
    refreshExperiments();
    handleSelectedExperimentEvent(eventType, payload, revision);
}

async function _onQueueEvent(event) {
    const payload = JSON.parse(event.data || '{}');
    const revision = _eventCursor(event, payload);
    const sessionName = payload.session_name;
    if (!sessionName || !_acceptEntityRevision(`queue:${sessionName}`, revision)) return;
    try {
        const response = await fetch(`/api/queue/${encodeURIComponent(sessionName)}`);
        if (!response.ok) return;
        const patch = await response.json();
        if (_dashboardEntityRevisions.get(`queue:${sessionName}`) !== revision) return;
        const current = _dashboardQueueData || { running_list: [], queued: [], state: patch.state };
        const runningList = (current.running_list || [])
            .filter(item => item.session_name !== sessionName)
            .concat(patch.running_list || []);
        const queued = (current.queued || [])
            .filter(item => item.session_name !== sessionName)
            .concat(patch.queued || []);
        _dashboardQueueData = {
            running: runningList.length === 1 ? runningList[0] : null,
            running_list: runningList,
            queued,
            state: patch.state,
        };
        renderQueueData(_dashboardQueueData);
    } catch (error) {
        console.warn('Targeted queue refresh failed:', error);
    }
}

async function _onSessionEvent(event) {
    const payload = JSON.parse(event.data || '{}');
    const revision = _eventCursor(event, payload);
    if (payload.hub) {
        const revisionKey = 'session:__hub__';
        if (!_acceptEntityRevision(revisionKey, revision)) return;
        _hubData = payload.hub;
        if (!_sessionPopoverOpen) renderSessionChips();
        return;
    }
    const sessionName = payload.session_name;
    if (!sessionName || !_acceptEntityRevision(`session:${sessionName}`, revision)) return;
    if (payload.deleted) {
        _sessionData = _sessionData.filter(session => session.name !== sessionName);
        renderSessionChips();
        return;
    }
    try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionName)}`);
        if (!response.ok) return;
        const data = await response.json();
        if (_dashboardEntityRevisions.get(`session:${sessionName}`) !== revision) return;
        const index = _sessionData.findIndex(session => session.name === sessionName);
        if (index >= 0) _sessionData[index] = data.session;
        else _sessionData.push(data.session);
        if (!_sessionPopoverOpen) renderSessionChips();
    } catch (error) {
        console.warn('Targeted session refresh failed:', error);
    }
}

async function _recoverDashboardSnapshot() {
    if (_dashboardResetInProgress) return;
    _dashboardResetInProgress = true;
    if (_dashboardEventSource) _dashboardEventSource.close();
    try {
        await loadDashboardSnapshot();
        const experiments = await refreshExperiments();
        renderQueueData(_dashboardQueueData);
        renderSessionChips();
        const selected = State.get('selectedExp');
        const match = experiments.find(exp =>
            exp.code_hash === selected || String(exp.id) === String(selected)
        );
        if (match) {
            await selectExperiment(match.code_hash || match.id, match.experiment_ids || [match.id]);
        }
        connectDashboardEvents(_dashboardQueryCursor ?? _dashboardLastEventId);
    } catch (error) {
        console.error('Dashboard reset recovery failed:', error);
        setTimeout(_recoverDashboardSnapshot, 1000);
    } finally {
        _dashboardResetInProgress = false;
    }
}

function connectDashboardEvents(afterEventId) {
    if (_dashboardEventSource) _dashboardEventSource.close();
    _dashboardLastEventId = Number(afterEventId || 0);
    _dashboardEventSource = new EventSource(`/api/events?after=${_dashboardLastEventId}`);
    _dashboardEventSource.onopen = () => {
        window.nanorunDashboardSSE.connected = true;
    };
    _dashboardEventSource.onerror = () => {
        window.nanorunDashboardSSE.connected = false;
    };
    ['experiment.created', 'experiment.updated', 'experiment.deleted', 'metrics.changed']
        .forEach(eventType => {
            _dashboardEventSource.addEventListener(eventType, event => {
                const payload = JSON.parse(event.data || '{}');
                _recordDashboardEventLatency(payload);
                const cursor = _eventCursor(event, payload);
                if (cursor <= _dashboardLastEventId) return;
                _dashboardLastEventId = cursor;
                _onExperimentEvent(eventType, event);
            });
        });
    _dashboardEventSource.addEventListener('queue.changed', event => {
        const payload = JSON.parse(event.data || '{}');
        _recordDashboardEventLatency(payload);
        const revision = _eventCursor(event, payload);
        if (revision <= _dashboardLastEventId) return;
        _dashboardLastEventId = revision;
        _onQueueEvent(event);
    });
    _dashboardEventSource.addEventListener('session.changed', event => {
        const payload = JSON.parse(event.data || '{}');
        _recordDashboardEventLatency(payload);
        const revision = _eventCursor(event, payload);
        if (revision <= _dashboardLastEventId) return;
        _dashboardLastEventId = revision;
        _onSessionEvent(event);
    });
    _dashboardEventSource.addEventListener('dashboard.reset', event => {
        const payload = JSON.parse(event.data || '{}');
        _dashboardLastEventId = Math.max(_dashboardLastEventId, _eventCursor(event, payload));
        _recoverDashboardSnapshot();
    });
}
