// Main application logic for nanorun dashboard

// --- Theme management ---

let THEMES = [];
let _themeMenuBuilt = false;

function applyTheme(name) {
    const v = document.querySelector('meta[name="version"]')?.content || Date.now();
    document.getElementById('theme-link').href = `/static/themes/${name}.css?v=${v}`;
    State.set('theme', name);
    document.getElementById('theme-menu')?.classList.add('hidden');
    document.querySelectorAll('.theme-option').forEach(el => {
        el.classList.toggle('active', el.dataset.theme === name);
    });
}

function toggleThemeMenu(e) {
    e.stopPropagation();
    const menu = document.getElementById('theme-menu');
    const wasHidden = menu.classList.contains('hidden');
    menu.classList.toggle('hidden');
    if (wasHidden) {
        if (!_themeMenuBuilt) buildThemeMenu();
        document.addEventListener('click', _closeThemeMenu);
    } else {
        document.removeEventListener('click', _closeThemeMenu);
    }
}

function _closeThemeMenu() {
    document.getElementById('theme-menu').classList.add('hidden');
    document.removeEventListener('click', _closeThemeMenu);
}

async function buildThemeMenu() {
    const menu = document.getElementById('theme-menu');
    if (!THEMES.length) {
        try { THEMES = await (await fetch('/api/themes')).json(); }
        catch { THEMES = ['dark']; }
    }
    const current = State.get('theme');
    menu.innerHTML = THEMES.map(name =>
        `<div class="theme-option${name === current ? ' active' : ''}" data-theme="${name}" onclick="applyTheme('${name}')"><span class="theme-sun">☼</span>&nbsp; ${name.replace(/-/g, ' ')}</div>`
    ).join('');
    _themeMenuBuilt = true;
    for (const name of THEMES) {
        try {
            const resp = await fetch(`/static/themes/${name}.css?v=${document.querySelector('meta[name="version"]')?.content || ''}`);
            const css = await resp.text();
            const accent = css.match(/--accent:\s*([^;]+)/)?.[1]?.trim();
            const bg = css.match(/--bg-card:\s*([^;]+)/)?.[1]?.trim();
            const el = menu.querySelector(`[data-theme="${name}"]`);
            if (el && accent && bg) {
                el.style.color = accent;
                el.style.background = bg;
            }
        } catch {}
    }
}

// --- View switching ---

function switchView(viewName) {
    State.set('view', viewName);
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });
    document.querySelectorAll('.panel-view').forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}-view`);
    });
    if (viewName === 'queue') {
        refreshQueue();
    } else {
        refreshExperiments();
    }
}

// --- Mobile nav ---

function switchMobilePanel(panelName) {
    const sidebar = document.getElementById('experiments-panel');
    const detail = document.querySelector('.detail-panel');
    sidebar.classList.remove('mobile-active');
    detail.classList.remove('mobile-active');
    if (panelName === 'experiments') {
        sidebar.classList.add('mobile-active');
        switchView('experiments');
    } else if (panelName === 'queue') {
        sidebar.classList.add('mobile-active');
        switchView('queue');
    } else if (panelName === 'detail') {
        detail.classList.add('mobile-active');
    }
    document.querySelectorAll('.mobile-nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.panel === panelName);
    });
}

// --- Sidebar toggle ---

function toggleSidebar() {
    if (isMobile()) return;
    const panel = document.getElementById('experiments-panel');
    panel.classList.toggle('collapsed');
    State.set('sidebarCollapsed', panel.classList.contains('collapsed'));
}

// --- Diff / Notes loading ---

let currentNotesRaw = null;
let currentDiffNotesTab = State.get('tab') || 'diff';

async function loadDiff(codeHash) {
    const contentEl = document.getElementById('diff-content');
    if (!codeHash) {
        contentEl.innerHTML = '<p class="placeholder">No code hash available</p>';
        return;
    }
    try {
        const response = await fetch(`/api/diff/${codeHash}`);
        if (!response.ok) {
            contentEl.innerHTML = '<p class="placeholder">No diff available (root node or diff not generated)</p>';
            return;
        }
        const diffText = await response.text();
        contentEl.innerHTML = renderDiff(diffText);
    } catch (e) {
        contentEl.innerHTML = `<p class="placeholder">Error loading diff: ${e.message}</p>`;
    }
}

async function loadNotes(scriptPath) {
    const contentEl = document.getElementById('notes-content');
    const notesTab = document.querySelector('.diff-notes-tab[data-tab="notes"]');
    if (!scriptPath) {
        contentEl.innerHTML = '';
        notesTab.style.display = 'none';
        if (currentDiffNotesTab === 'notes') switchDiffNotesTab('diff');
        return;
    }
    try {
        const response = await fetch(`/api/notes/${encodeURIComponent(scriptPath)}`);
        if (!response.ok || response.status === 204) {
            contentEl.innerHTML = '';
            notesTab.style.display = 'none';
            if (currentDiffNotesTab === 'notes') switchDiffNotesTab('diff');
            return;
        }
        const notesText = await response.text();
        contentEl.innerHTML = renderNotes(notesText);
        notesTab.style.display = '';
    } catch {
        contentEl.innerHTML = '';
        notesTab.style.display = 'none';
        if (currentDiffNotesTab === 'notes') switchDiffNotesTab('diff');
    }
}

function renderNotes(notesText) {
    if (!notesText || notesText.trim() === '') {
        currentNotesRaw = null;
        return '<p class="placeholder">Empty notes file</p>';
    }
    currentNotesRaw = notesText;
    const html = marked.parse(notesText);
    return `<div class="notes-wrapper">
        <button class="notes-copy-btn" onclick="copyNotes(this)" title="Copy notes">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
        </button>
        <div class="notes-markdown">${html}</div>
    </div>`;
}

function copyNotes(btn) {
    if (!currentNotesRaw) return;
    copyToClipboard(currentNotesRaw, btn);
}

function switchDiffNotesTab(tabName) {
    currentDiffNotesTab = tabName;
    State.set('tab', tabName);
    document.querySelectorAll('.diff-notes-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.getElementById('diff-tab-content').classList.toggle('active', tabName === 'diff');
    document.getElementById('notes-tab-content').classList.toggle('active', tabName === 'notes');
}

// --- Tracks loading ---

async function loadTracks() {
    const response = await fetch('/api/tracks');
    const data = await response.json();
    const select = document.getElementById('track-filter');
    data.tracks.forEach(track => {
        const option = document.createElement('option');
        option.value = track.name;
        option.textContent = track.name;
        select.appendChild(option);
    });
}

// --- Experiments list ---

async function refreshExperiments() {
    const searchFilter = document.getElementById('search-filter').value.trim();
    const statusFilter = document.getElementById('status-filter').value;
    const trackFilter = document.getElementById('track-filter').value;

    const params = new URLSearchParams();
    if (trackFilter) params.set('track', trackFilter);
    if (statusFilter) params.set('status', statusFilter);
    if (searchFilter) params.set('search', searchFilter);

    const url = params.toString() ? `/api/experiments?${params}` : '/api/experiments';
    const response = await fetch(url);
    let data = await response.json();

    const listEl = document.getElementById('experiments-list');

    if (data.experiments.length === 0) {
        listEl.innerHTML = '<p class="placeholder">No experiments found. Run "nanorun run &lt;script&gt;" to start an experiment.</p>';
        return [];
    }

    listEl.innerHTML = data.experiments.map(exp => {
        const scriptPath = (exp.script || '').replace(/'/g, "\\'");
        return `
        <div class="experiment-card ${exp.status}" onclick="selectExperiment('${exp.code_hash || exp.id}', ${JSON.stringify(exp.experiment_ids || [exp.id])})" data-id="${exp.code_hash || exp.id}">
            <div class="exp-header">
                <div class="exp-name-group">
                    <span class="exp-name copyable" onclick="event.stopPropagation(); copyToClipboard('${scriptPath}', this)" title="Click to copy: ${exp.script || 'n/a'}"><span class="exp-track">${exp.track || 'untracked'}</span>/${exp.name.substring(0, 25)}${exp.name.length > 25 ? '…' : ''}</span>${exp.code_hash ? `<span class="exp-hash">${exp.code_hash.substring(0, 8)}</span>` : ''}
                </div>
                <div class="exp-badges">
                    ${exp.status !== 'completed' ? `<span class="status-badge ${exp.status}">${exp.status}</span>` : ''}
                </div>
            </div>
            <div class="exp-metrics">
                <span class="val-loss">${formatValLoss(exp.val_loss, exp.train_time_ms)}${exp.n_runs > 1 ? ` <span class="n-runs-inline">(n=${exp.n_runs})</span>` : ''}</span>
                ${exp.status === 'running' ? `
                <div class="exp-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${exp.total_steps ? (exp.current_step / exp.total_steps * 100) : 0}%"></div>
                    </div>
                    <span class="progress-text">${exp.current_step || 0}/${exp.total_steps || '?'}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `}).join('');

    // Re-highlight selected
    const sel = State.get('selectedExp');
    if (sel) {
        document.querySelectorAll('.experiment-card').forEach(el => {
            el.classList.toggle('selected', el.dataset.id == sel);
        });
    }

    renderBucketCard();
    return data.experiments;
}

// --- selectExperiment (main orchestrator) ---

async function selectExperiment(codeHashOrId, experimentIds) {
    State.set('selectedQueuedScript', null);
    const prevExp = State.get('selectedExp');
    const isSameExperiment = prevExp === codeHashOrId;

    State.update({
        selectedExp: codeHashOrId,
        selectedExpIds: experimentIds,
    });

    if (!isSameExperiment) {
        State.update({
            selectedRun: null,
            selectedCellExpIds: null,
            heatmapSelectedVars: [],
            chartView: null,
        });
    }

    // Highlight selected card
    document.querySelectorAll('.experiment-card').forEach(el => {
        el.classList.toggle('selected', el.dataset.id == codeHashOrId);
    });

    // Fetch details for all experiments in the group
    const allData = await Promise.all(
        experimentIds.map(id => fetch(`/api/experiment/${id}`).then(r => r.json()))
    );
    const validData = allData.filter(d => !d.error);
    if (validData.length === 0) {
        document.getElementById('experiment-details').innerHTML = '<p class="error">No valid experiments found</p>';
        return;
    }

    const primary = validData[0];
    const isMultiple = validData.length > 1;

    // Store experiment data in state
    State.set('experimentData', validData);
    State.set('currentTotalSteps', Math.max(...validData.map(d => d.total_steps || 0)));

    // Load heatmap defaults for this code hash
    const currentCodeHash = primary.code_hash;
    if (!isSameExperiment && currentCodeHash) {
        let heatmapDefs = State.get('heatmapDefaults', currentCodeHash);
        const partialVars = detectPartialVars(validData);
        if (partialVars.length > 0 && primary.script) {
            const needsAutoDetect = partialVars.some(k => !heatmapDefs[k] || heatmapDefs[k] === '(default)');
            if (needsAutoDetect) {
                fetch(`/api/env-defaults/${primary.script}`).then(r => r.json()).then(data => {
                    if (data.defaults) {
                        let changed = false;
                        for (const key of partialVars) {
                            if (data.defaults[key] !== undefined && heatmapDefs[key] !== data.defaults[key]) {
                                heatmapDefs[key] = data.defaults[key];
                                changed = true;
                            }
                        }
                        if (changed) {
                            State.set('heatmapDefaults', heatmapDefs, currentCodeHash);
                            if (State.get('chartView') === 'heatmap') {
                                switchChartView('heatmap', false);
                            }
                        }
                    }
                }).catch(() => {});
            }
        }
    }

    // Check if sweep
    const envVarStrings = new Set(validData.map(d => JSON.stringify(d.env_vars || {})));
    const isSweep = envVarStrings.size > 1;

    // Compute aggregated stats (exclude running from averages)
    const completedData = validData.filter(d => d.status !== 'running');
    const valLosses = completedData.map(d => d.final_val_loss).filter(v => v != null);
    const trainTimes = completedData.map(d => d.final_train_time_ms).filter(v => v != null);
    const meanValLoss = valLosses.length ? valLosses.reduce((a, b) => a + b, 0) / valLosses.length : null;
    const meanTrainTime = trainTimes.length ? trainTimes.reduce((a, b) => a + b, 0) / trainTimes.length : null;

    const isBucket = State.get('selectedExp') === 'bucket';

    document.getElementById('detail-title').textContent = isBucket ? '★ Bucket' : (primary.script ? primary.script.split('/').pop() : primary.name);

    const copyTitleInput = document.getElementById('chart-copy-title');
    if (copyTitleInput) copyTitleInput.value = primary.script ? primary.script.split('/').pop() : primary.name;

    // Meta row
    document.getElementById('meta-row-container').innerHTML = isBucket ? `
        <div class="meta-row">
            <span class="meta-item bucket-label">${validData.length} run${validData.length !== 1 ? 's' : ''} collected</span>
        </div>
    ` : `
        <div class="meta-row">
            <a class="meta-item meta-script" href="#" onclick="event.preventDefault(); revealInFinder(${primary.id})" title="Reveal in Finder: ${primary.script || 'n/a'}">${primary.track || 'untracked'}/${primary.script ? primary.script.split('/').pop() : 'n/a'}</a>
            <span class="meta-sep">·</span>
            <span class="meta-item">${primary.gpus}x ${primary.gpu_type || 'H100'}</span>
            <span class="meta-sep">·</span>
            ${primary.code_hash
                ? `<span class="meta-item meta-hash" title="${primary.code_hash}">${primary.code_hash.substring(0, 8)}</span>`
                : `<span class="meta-item meta-hash-missing">no code hash</span>`}
            ${isMultiple ? `<span class="meta-sep">·</span><span class="meta-item">${isSweep ? 'sweep' : 'n'}=${validData.length}</span>` : ''}
        </div>
    `;

    // Details panel
    const detailsEl = document.getElementById('experiment-details');
    if (!isSameExperiment) {
        State.update({ runsEnvFilters: {}, runsSort: { col: 'started_at', dir: 'desc' }, chartXRange: null });
    }
    const anyHasEnvVars = validData.some(d => d.env_vars && Object.keys(d.env_vars).length > 0);
    runsIsSweep = isSweep || isBucket || anyHasEnvVars;

    detailsEl.innerHTML = `
        <div class="stats-runs-block">
            <div class="runs-header-row">
                <h4 class="runs-header-clickable" onclick="copyRunsAsMarkdown()" title="Click to copy as markdown">Runs</h4>
                <div class="stats-hero">
                    <div class="stat-box">
                        <span class="stat-label">${isMultiple ? 'Mean Loss' : 'Val Loss'}</span>
                        <span class="stat-value ${meanValLoss && meanValLoss < 3.3 ? 'good' : ''}">${meanValLoss ? meanValLoss.toFixed(4) : 'n/a'}</span>
                        ${isMultiple && valLosses.length > 1 ? `<span class="stat-range">(${Math.min(...valLosses).toFixed(4)}-${Math.max(...valLosses).toFixed(4)})</span>` : ''}
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">${isMultiple ? 'Mean Time' : 'Train Time'}</span>
                        <span class="stat-value">${meanTrainTime ? (meanTrainTime / 1000).toFixed(1) + 's' : 'n/a'}</span>
                        ${isMultiple && trainTimes.length > 1 ? `<span class="stat-range">(${(Math.min(...trainTimes) / 1000).toFixed(1)}s-${(Math.max(...trainTimes) / 1000).toFixed(1)}s)</span>` : ''}
                    </div>
                </div>
            </div>
            <div id="runs-filters"></div>
            <div id="runs-table-container" class="runs-table"></div>
        </div>
    `;

    renderRunsTable();

    // Chart views
    const eligibleViews = getEligibleViews(validData);
    const metaChartBlock = document.querySelector('.meta-chart-block');

    if (eligibleViews.length === 0) {
        metaChartBlock.classList.add('no-views');
    } else {
        metaChartBlock.classList.remove('no-views');
        updateViewSwitcher(eligibleViews);
        const savedChartView = State.get('chartView');
        const viewToShow = (savedChartView && eligibleViews.includes(savedChartView))
            ? savedChartView
            : eligibleViews[0];
        switchChartView(viewToShow);
    }

    // Run/cell selection across refreshes
    const validIds = new Set(validData.map(d => d.id));
    let cellIds = State.get('selectedCellExpIds');
    if (cellIds) {
        cellIds = cellIds.filter(id => validIds.has(id));
        State.set('selectedCellExpIds', cellIds.length > 0 ? cellIds : null);
    }
    const runId = State.get('selectedRun');
    if (runId && !validIds.has(runId)) {
        State.set('selectedRun', null);
    }

    // Metrics table
    const currentCellIds = State.get('selectedCellExpIds');
    const currentRunId = State.get('selectedRun');
    if (currentCellIds) {
        showAveragedMetricsForCell(currentCellIds);
    } else if (currentRunId) {
        const exp = validData.find(d => d.id === currentRunId);
        if (exp) updateMetricsTable(exp.loss_curve, false, currentRunId);
    } else if (isMultiple) {
        const allCurves = validData.map(d => d.loss_curve);
        const averaged = computeAveragedMetrics(allCurves);
        updateMetricsTable(averaged, true, null);
    } else {
        updateMetricsTable(primary.loss_curve, false, null);
    }

    updateRunRowHighlights();
    updateHeatmapHighlights();

    // Diff and notes
    if (!isBucket) {
        loadDiff(primary.code_hash);
        loadNotes(primary.script);
    }

    // Mobile auto-navigate
    if (!isSameExperiment && isMobile()) switchMobilePanel('detail');
}

// --- Queue ---

async function refreshQueueCount() {
    try {
        const response = await fetch('/api/queue');
        const data = await response.json();
        const queueCount = data.queued ? data.queued.length : 0;
        const queueBtn = document.querySelector('.view-btn[data-view="queue"]');
        if (queueBtn) queueBtn.textContent = queueCount > 0 ? `Queue (${queueCount})` : 'Queue';
    } catch {}
}

async function refreshQueue() {
    try {
        const response = await fetch('/api/queue');
        const data = await response.json();

        const contentEl = document.getElementById('queue-content');
        const stateBadge = document.getElementById('queue-state-badge');

        stateBadge.textContent = data.state;
        stateBadge.className = `queue-state-badge ${data.state}`;

        let html = '';
        const selectedExpIds = State.get('selectedExpIds');
        const selectedQueued = State.get('selectedQueuedScript');

        const runningList = data.running_list || (data.running ? [data.running] : []);
        runningList.forEach(r => {
            const progress = r.total_steps ? (r.current_step / r.total_steps * 100) : 0;
            const envStr = Object.entries(r.env_vars || {}).map(([k, v]) => `${k}=${v}`).join(', ');
            const track = r.track || 'untracked';
            const scriptName = r.script ? r.script.split('/').pop() : r.name;
            const sessionLabel = r.session_name ? `<span class="queue-item-session">${r.session_name}</span>` : '';
            const isSelected = selectedExpIds && selectedExpIds.includes(r.id);
            html += `
                <div class="queue-running${isSelected ? ' selected' : ''}" onclick="goToExperiment(${r.id})" title="Click to view in Experiments">
                    <div class="queue-running-top">
                        <div class="queue-running-label">Running</div>
                        <div class="queue-item-right">
                            ${r.gpus ? `<span class="queue-item-gpus">${r.gpus}x ${r.gpu_type || 'GPU'}</span>` : ''}
                            ${sessionLabel}
                        </div>
                    </div>
                    <div class="queue-running-name"><span class="queue-item-track">${track}</span>/${scriptName}</div>
                    ${envStr ? `<div class="queue-item-env">${envStr}</div>` : ''}
                    <div class="queue-running-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                        <span class="progress-text">${r.current_step || 0}/${r.total_steps || '?'}</span>
                    </div>
                    ${r.val_loss ? `<div class="queue-running-loss">Val Loss: ${r.val_loss.toFixed(4)}</div>` : ''}
                </div>
            `;
        });

        if (data.queued && data.queued.length > 0) {
            html += `<div class="queue-section-label">Queued (${data.queued.length})</div>`;
            data.queued.forEach((item) => {
                const scriptName = item.script.split('/').pop();
                const track = item.track || 'untracked';
                const envStr = Object.entries(item.env_vars || {}).map(([k, v]) => `${k}=${v}`).join(', ');
                const sessionLabel = item.session_name ? `<span class="queue-item-session">${item.session_name}</span>` : '';
                const isSelected = (selectedQueued && selectedQueued === item.script) || (selectedExpIds && selectedExpIds.includes(item.id));
                html += `
                    <div class="queue-item${isSelected ? ' selected' : ''}" onclick="goToQueuedItem('${item.script.replace(/'/g, "\\'")}')" title="Click to view script diff">
                        <span class="queue-item-index">${item.session_index || '?'}</span>
                        <div class="queue-item-info">
                            <div class="queue-item-name"><span class="queue-item-track">${track}</span>/${scriptName}</div>
                            ${envStr ? `<div class="queue-item-env">${envStr}</div>` : ''}
                        </div>
                        <div class="queue-item-right">
                            <span class="queue-item-gpus">${item.gpus}x ${item.gpu_type}</span>
                            ${sessionLabel}
                        </div>
                    </div>
                `;
            });
        } else if (runningList.length === 0) {
            html += `<div class="queue-empty">Queue is empty</div>`;
        }

        contentEl.innerHTML = html;

        const queueBtn = document.querySelector('.view-btn[data-view="queue"]');
        const queueCount = (data.queued ? data.queued.length : 0) + runningList.length;
        if (queueBtn) queueBtn.textContent = queueCount > 0 ? `Queue (${queueCount})` : 'Queue';
    } catch (e) {
        console.error('Failed to refresh queue:', e);
    }
}

async function goToQueuedItem(scriptPath) {
    if (!scriptPath) return;
    State.set('selectedQueuedScript', scriptPath);
    State.update({ selectedExp: null, selectedExpIds: null });
    refreshQueue();
    const detailsEl = document.getElementById('experiment-details');
    detailsEl.innerHTML = '';
    document.getElementById('detail-title').textContent = scriptPath.split('/').pop();
    document.getElementById('meta-row-container').innerHTML = `
        <div class="meta-row">
            <span class="meta-item">${scriptPath}</span>
            <span class="meta-sep">·</span>
            <span class="meta-item" style="color: var(--warning)">queued</span>
        </div>
    `;
    const chartBlock = document.querySelector('.meta-chart-block');
    if (chartBlock) chartBlock.classList.add('no-views');
    const metricsTable = document.getElementById('metrics-table');
    if (metricsTable) metricsTable.innerHTML = '';
    try {
        await fetch(`/api/env-defaults/${encodeURIComponent(scriptPath)}`);
    } catch {}
    loadNotes(scriptPath);
    try {
        const resp = await fetch(`/api/experiments?search=${encodeURIComponent(scriptPath.split('/').pop().replace('.py', ''))}&limit=1&aggregate=false`);
        const data = await resp.json();
        if (data.experiments && data.experiments.length > 0 && data.experiments[0].code_hash) {
            loadDiff(data.experiments[0].code_hash);
        }
    } catch {}
    switchDiffNotesTab('diff');
}

async function goToExperiment(expId) {
    if (!expId) return;
    try {
        const resp = await fetch(`/api/experiment/${expId}`);
        const exp = await resp.json();
        if (exp.error) return;
        const codeHash = exp.code_hash;
        if (codeHash) {
            const listResp = await fetch(`/api/experiments?search=${codeHash}&limit=1`);
            const listData = await listResp.json();
            const match = listData.experiments.find(e => e.code_hash === codeHash);
            if (match) {
                await selectExperiment(match.code_hash, match.experiment_ids || [match.id]);
            } else {
                await selectExperiment(codeHash, [expId]);
            }
        } else {
            await selectExperiment(expId, [expId]);
        }
    } catch {
        await selectExperiment(expId, [expId]);
    }
    if (State.get('view') === 'queue') refreshQueue();
}

// --- Session chips ---

const MAX_SESSION_CHIPS = 3;
let _sessionData = [];
let _sessionPopoverOpen = null;
let _hubData = {};

async function refreshSessionChips() {
    try {
        const resp = await fetch('/api/sessions');
        const data = await resp.json();
        _sessionData = data.sessions || data;
        _hubData = data.hub || {};
        _sessionData.sort((a, b) => {
            const order = { disconnected: 0, connecting: 1, connected: 2 };
            return (order[a.status] ?? 1) - (order[b.status] ?? 1);
        });
        if (_sessionPopoverOpen) return;
        renderSessionChips();
    } catch {}
}

function renderSessionChips() {
    const container = document.getElementById('session-chips');
    if (!container) return;
    const shown = _sessionData.slice(0, MAX_SESSION_CHIPS);
    const overflow = _sessionData.length - shown.length;
    container.innerHTML = shown.map(s => {
        const statusClass = s.status || 'disconnected';
        let suffix = '';
        if (s.status === 'connecting') suffix = ' <span class="chip-suffix connecting">reconnecting...</span>';
        return `<div class="session-chip ${statusClass}" onclick="onSessionChipClick('${s.name}')" data-session="${s.name}"><span class="chip-dot"></span>${s.name}${suffix}</div>`;
    }).join('') + (overflow > 0 ? `<span class="session-chip chip-overflow">+${overflow}</span>` : '');

    const hubContainer = document.getElementById('hub-chips');
    if (!hubContainer) return;
    const hubStatus = _hubData.status || 'unknown';
    const hubClass = hubStatus === 'connected' ? 'connected' : 'disconnected';
    const hubSuffix = hubStatus === 'error' ? ' <span class="chip-suffix connecting">paused</span>' : '';
    hubContainer.innerHTML = `<div class="session-chip ${hubClass}" onclick="onHubChipClick()" data-hub="hub"><span class="chip-icon">↓</span>logs${hubSuffix}</div>`;
}

async function onSessionChipClick(name) { openSessionPopover(name); }
function onHubChipClick() { openHubPopover(); }

function openHubPopover() {
    closeSessionPopover();
    _sessionPopoverOpen = 'hub';
    const chip = document.querySelector('[data-hub="hub"]');
    if (!chip) return;
    const popover = document.createElement('div');
    popover.className = 'session-popover';
    popover.id = 'session-popover';
    popover.onclick = e => e.stopPropagation();
    const h = _hubData;
    let body = `<div class="sp-header"><span class="sp-name">logs</span><span class="sp-close" onclick="closeSessionPopover()">✕</span></div><div class="sp-info">HF Hub log sync</div>`;
    if (h.status === 'connected') {
        const syncLabel = h.last_sync_at ? `Last sync: ${h.last_sync_at}` : 'Syncing...';
        body += `<div class="sp-info" style="color:var(--accent)">${syncLabel}</div>`;
    } else {
        if (h.last_error) body += `<div class="sp-error">${h.last_error}</div>`;
    }
    body += `<div class="sp-actions"><button class="sp-btn sp-btn-primary" onclick="doHubReconnect()">Reconnect</button></div>`;
    popover.innerHTML = body;
    chip.style.position = 'relative';
    chip.appendChild(popover);
    setTimeout(() => document.addEventListener('click', _popoverOutsideClick), 0);
}

async function doHubReconnect() {
    closeSessionPopover();
    await fetch('/api/hub/reconnect', { method: 'POST' });
    setTimeout(refreshSessionChips, 2000);
    setTimeout(refreshSessionChips, 5000);
}

function openSessionPopover(name) {
    closeSessionPopover();
    const s = _sessionData.find(x => x.name === name);
    if (!s) return;
    _sessionPopoverOpen = name;
    const chip = document.querySelector(`[data-session="${name}"]`);
    const popover = document.createElement('div');
    popover.className = 'session-popover';
    popover.id = 'session-popover';
    const gpuLabel = s.gpu_count > 1 ? `${s.gpu_count}× ${s.gpu_type}` : s.gpu_type;
    let body = `<div class="sp-header"><span class="sp-name">${s.name}</span><span class="sp-close" onclick="closeSessionPopover()">✕</span></div><div class="sp-info"><span style="color:var(--text-primary)">${s.host}</span> <span style="color:#888">${gpuLabel}</span></div>`;
    if (s.status === 'connected') {
        body += `<div class="sp-status-detail" id="sp-detail-${name}">Loading...</div>`;
        body += `<div class="sp-actions"><button class="sp-btn sp-btn-secondary" onclick="doDaemonRestart('${name}')">Daemon Restart</button></div>`;
    } else {
        if (s.last_error) body += `<div class="sp-error">${s.last_error}</div>`;
        body += `<div class="sp-actions"><button class="sp-btn sp-btn-primary" onclick="doReconnect('${name}')">Reconnect</button><button class="sp-btn sp-btn-secondary" onclick="doDaemonRestart('${name}')">Daemon Restart</button><button class="sp-btn sp-btn-danger" onclick="doRemoveSession('${name}')">Remove</button></div>`;
    }
    popover.innerHTML = body;
    popover.onclick = e => e.stopPropagation();
    chip.style.position = 'relative';
    chip.appendChild(popover);
    if (s.status === 'connected') loadDaemonStatus(name);
    setTimeout(() => document.addEventListener('click', _popoverOutsideClick), 0);
}

function _popoverOutsideClick(e) {
    const popover = document.getElementById('session-popover');
    if (popover && !popover.contains(e.target) && !e.target.closest('.session-chip')) {
        closeSessionPopover();
    }
}

function closeSessionPopover() {
    _sessionPopoverOpen = null;
    const el = document.getElementById('session-popover');
    if (el) el.remove();
    document.removeEventListener('click', _popoverOutsideClick);
}

async function loadDaemonStatus(name) {
    const el = document.getElementById(`sp-detail-${name}`);
    if (!el) return;
    try {
        const resp = await fetch(`/api/sessions/${name}/daemon-status`);
        if (!resp.ok) { el.innerHTML = '<span style="color:var(--error)">Could not reach daemon</span>'; return; }
        const d = await resp.json();
        if (!d) { el.innerHTML = '<span style="color:var(--error)">Could not reach daemon</span>'; return; }
        const statusColor = d.status === 'running' ? 'var(--accent)' : d.status === 'paused' ? 'var(--warning)' : 'var(--text-secondary)';
        let html = `<div><span style="color:${statusColor}">Daemon ${d.status || 'unknown'}</span></div>`;
        if (d.current_experiment_id) {
            html += `<div>Experiment: <span style="color:var(--info)">#${d.current_experiment_id}</span></div>`;
            if (d.current_window) html += `<div>Window: <span style="color:var(--warning)">${d.current_window}</span></div>`;
            if (d.current_run_id) html += `<div>Run ID: <span style="color:var(--text-secondary)">${d.current_run_id}</span></div>`;
        }
        html += `<div>Queue: <span style="color:var(--info)">${d.queue_length || 0} pending</span></div>`;
        const gpuProcs = d.gpu_processes || [];
        if (gpuProcs.length) {
            const totalMem = gpuProcs.reduce((s, p) => s + p.memory_mb, 0);
            html += `<div>GPU: <span style="color:var(--accent)">${gpuProcs.length} process(es), ${totalMem}MB</span></div>`;
            for (const p of gpuProcs) {
                html += `<div style="padding-left:0.8rem;color:var(--text-secondary)">PID ${p.pid}: ${p.name} (${p.memory_mb}MB)</div>`;
            }
        }
        el.innerHTML = html;
    } catch { el.innerHTML = '<span style="color:var(--error)">Error loading status</span>'; }
}

async function doReconnect(name) {
    closeSessionPopover();
    await fetch(`/api/sessions/${name}/reconnect`, { method: 'POST' });
    setTimeout(refreshSessionChips, 500);
    setTimeout(refreshSessionChips, 2000);
    setTimeout(refreshSessionChips, 5000);
    setTimeout(refreshSessionChips, 10000);
}

async function doRemoveSession(name) {
    closeSessionPopover();
    const resp = await fetch(`/api/sessions/${name}`, { method: 'DELETE' });
    const data = await resp.json();
    if (!data.success) {
        console.error('Failed to remove session:', data.error);
        return;
    }
    refreshSessionChips();
}

async function doDaemonRestart(name) {
    closeSessionPopover();
    await fetch(`/api/sessions/${name}/daemon-restart`, { method: 'POST' });
    setTimeout(async () => {
        await fetch(`/api/sessions/${name}/reconnect`, { method: 'POST' });
        setTimeout(refreshSessionChips, 2000);
        setTimeout(refreshSessionChips, 5000);
        setTimeout(refreshSessionChips, 10000);
    }, 6000);
    setTimeout(refreshSessionChips, 1000);
}

// --- revealInFinder ---

async function revealInFinder(expId) {
    try {
        const response = await fetch(`/api/reveal/${expId}`, { method: 'POST' });
        const data = await response.json();
        if (!data.success) console.error('Failed to reveal:', data.error);
    } catch (e) {
        console.error('Failed to reveal:', e);
    }
}

// --- Auto-refresh / init ---

let refreshInterval = null;

async function startAutoRefresh() {
    // Restore view
    const savedView = State.get('view');
    if (savedView === 'queue') {
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === 'queue');
        });
        document.querySelectorAll('.panel-view').forEach(view => {
            view.classList.toggle('active', view.id === 'queue-view');
        });
        refreshQueue();
    }

    const experiments = await refreshExperiments();
    refreshQueueCount();

    // Restore tab
    const savedTab = State.get('tab');
    if (savedTab && (savedTab === 'diff' || savedTab === 'notes')) {
        currentDiffNotesTab = savedTab;
        switchDiffNotesTab(savedTab);
    }

    // Restore selected experiment
    const savedExp = State.get('selectedExp');
    if (savedExp && experiments && experiments.length > 0) {
        const match = experiments.find(e =>
            e.code_hash === savedExp || String(e.id) === savedExp
        );
        if (match) {
            await selectExperiment(match.code_hash || match.id, match.experiment_ids || [match.id]);
            const savedRun = State.get('selectedRun');
            if (savedRun) {
                const data = State.get('experimentData');
                if (data && data.find(d => d.id === savedRun)) {
                    showMetricsForRun(savedRun);
                }
            }
        }
    } else if (!State.get('selectedExp') && experiments && experiments.length > 0) {
        const first = experiments[0];
        selectExperiment(first.code_hash || first.id, first.experiment_ids || [first.id]);
    }

    refreshInterval = setInterval(() => {
        if (State.get('deleteInProgress')) return;
        if (State.get('view') === 'queue') {
            refreshQueue();
        } else {
            refreshExperiments();
            refreshQueueCount();
            const sel = State.get('selectedExp');
            const selIds = State.get('selectedExpIds');
            if (sel && selIds && selIds.length > 0) {
                selectExperiment(sel, selIds);
            }
        }
    }, 5000);
}

document.addEventListener('DOMContentLoaded', () => {
    // Apply persisted theme
    const theme = State.get('theme');
    const v = document.querySelector('meta[name="version"]')?.content || Date.now();
    document.getElementById('theme-link').href = `/static/themes/${theme}.css?v=${v}`;

    // Restore sidebar state
    if (State.get('sidebarCollapsed')) {
        document.getElementById('experiments-panel').classList.add('collapsed');
    }

    loadTracks();
    startAutoRefresh();
    refreshSessionChips();
    setInterval(refreshSessionChips, 5000);

    // Mobile init
    function initMobileState() {
        if (isMobile()) {
            const sidebar = document.getElementById('experiments-panel');
            const detail = document.querySelector('.detail-panel');
            if (!sidebar.classList.contains('mobile-active') && !detail.classList.contains('mobile-active')) {
                sidebar.classList.add('mobile-active');
            }
        } else {
            document.getElementById('experiments-panel').classList.remove('mobile-active');
            document.querySelector('.detail-panel').classList.remove('mobile-active');
        }
    }
    initMobileState();
    window.addEventListener('resize', initMobileState);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
        if (e.key === 'Escape') {
            const chartBlock = document.querySelector('.meta-chart-block');
            if (chartBlock && chartBlock.classList.contains('fullscreen')) toggleChartFullscreen();
        }
        if (e.key === 'b' || e.key === 'B') toggleSidebar();
        if (e.key === 'e' || e.key === 'E') switchView('experiments');
        if (e.key === 'q' || e.key === 'Q') switchView('queue');
        if (e.key === 'Meta' || e.key === 'Control') {
            document.querySelectorAll('.meta-run-id:hover').forEach(el => el.classList.add('cmd-hover'));
        }
    });
    document.addEventListener('keyup', (e) => {
        if (e.key === 'Meta' || e.key === 'Control') {
            document.querySelectorAll('.meta-run-id.cmd-hover').forEach(el => el.classList.remove('cmd-hover'));
        }
    });

    // Cmd-hover: toggle underline on run IDs when cmd held during hover
    document.addEventListener('mouseover', (e) => {
        const el = e.target.closest('.meta-run-id');
        if (el && (e.metaKey || e.ctrlKey)) el.classList.add('cmd-hover');
    });
    document.addEventListener('mouseout', (e) => {
        const el = e.target.closest('.meta-run-id');
        if (el) el.classList.remove('cmd-hover');
    });
});
