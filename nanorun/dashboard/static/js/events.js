// Durable dashboard snapshot + Server-Sent Events state reducer.

let _dashboardExperimentSummaries = new Map();
let _dashboardQueueData = null;
let _dashboardLastEventId = 0;
let _dashboardEventSource = null;
let _dashboardSnapshotReady = false;
let _dashboardResetInProgress = false;
const _dashboardEntityRevisions = new Map();
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
    const summaries = Array.from(_dashboardExperimentSummaries.values())
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
    _dashboardLastEventId = Number(snapshot.last_event_id || 0);
    _dashboardExperimentSummaries = new Map();
    _dashboardEntityRevisions.clear();
    (snapshot.experiment_summaries || []).forEach(summary => {
        summary.revision = Number(summary.revision || _dashboardLastEventId);
        _dashboardExperimentSummaries.set(Number(summary.id), summary);
        _dashboardEntityRevisions.set(`experiment:${summary.id}`, summary.revision);
    });
    _dashboardQueueData = snapshot.queue || { running: null, running_list: [], queued: [], state: 'active' };
    _sessionData = snapshot.sessions || [];
    _hubData = snapshot.hub || {};
    _dashboardSnapshotReady = true;
}

async function loadDashboardSnapshot() {
    const response = await fetch('/api/dashboard/snapshot');
    if (!response.ok) throw new Error(`Snapshot failed (${response.status})`);
    const snapshot = await response.json();
    _installDashboardSnapshot(snapshot);
    return snapshot;
}

function _eventRevision(event, payload) {
    return Number(event.lastEventId || payload.revision || 0);
}

function _acceptEntityRevision(entityKey, revision) {
    const previous = Number(_dashboardEntityRevisions.get(entityKey) || 0);
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
    } else if (payload.summary) {
        payload.summary.revision = revision;
        _dashboardExperimentSummaries.set(experimentId, payload.summary);
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
    const revision = _eventRevision(event, payload);
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
    const revision = _eventRevision(event, payload);
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
        connectDashboardEvents(_dashboardLastEventId);
    } catch (error) {
        console.error('Dashboard reset recovery failed:', error);
        setTimeout(_recoverDashboardSnapshot, 1000);
    } finally {
        _dashboardResetInProgress = false;
    }
}

function connectDashboardEvents(afterEventId) {
    if (_dashboardEventSource) _dashboardEventSource.close();
    _dashboardEventSource = new EventSource(`/api/events?after=${Number(afterEventId || 0)}`);
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
                const revision = _eventRevision(event, payload);
                if (revision <= _dashboardLastEventId) return;
                _dashboardLastEventId = revision;
                _onExperimentEvent(eventType, event);
            });
        });
    _dashboardEventSource.addEventListener('queue.changed', event => {
        const payload = JSON.parse(event.data || '{}');
        _recordDashboardEventLatency(payload);
        const revision = _eventRevision(event, payload);
        if (revision <= _dashboardLastEventId) return;
        _dashboardLastEventId = revision;
        _onQueueEvent(event);
    });
    _dashboardEventSource.addEventListener('session.changed', event => {
        const payload = JSON.parse(event.data || '{}');
        _recordDashboardEventLatency(payload);
        const revision = _eventRevision(event, payload);
        if (revision <= _dashboardLastEventId) return;
        _dashboardLastEventId = revision;
        _onSessionEvent(event);
    });
    _dashboardEventSource.addEventListener('dashboard.reset', event => {
        const payload = JSON.parse(event.data || '{}');
        _dashboardLastEventId = Math.max(_dashboardLastEventId, _eventRevision(event, payload));
        _recoverDashboardSnapshot();
    });
}
