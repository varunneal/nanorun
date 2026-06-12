// Runs table and experiment management for nanorun dashboard

let currentMetricsData = null;
let runsIsSweep = false;
let _lastDigit = null;
let _lastDigitTime = 0;

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key >= '1' && e.key <= '9') {
        _lastDigit = parseInt(e.key);
        _lastDigitTime = Date.now();
    }
});
document.addEventListener('keyup', (e) => {
    if (e.key >= '1' && e.key <= '9' && parseInt(e.key) === _lastDigit) _lastDigit = null;
});

function getHeldDigit() {
    if (_lastDigit && Date.now() - _lastDigitTime < 2000) return _lastDigit;
    return null;
}

// --- Bucket helpers ---

function bucketKey(i) { return i === 0 ? 'bucket' : `bucket-${i + 1}`; }

function parseBucketKey(key) {
    if (key === 'bucket') return 0;
    if (typeof key === 'string' && key.startsWith('bucket-')) return parseInt(key.split('-')[1]) - 1;
    return -1;
}

function isBucketKey(key) { return parseBucketKey(key) >= 0; }

function getBucket(n = 0) { return State.get('buckets')[n] || []; }

function saveBucket(n, ids) {
    const buckets = State.get('buckets');
    buckets[n] = ids;
    State.set('buckets', buckets);
}

function getAllBucketExpIds() {
    const all = new Set();
    for (const b of State.get('buckets')) for (const id of b) all.add(id);
    return all;
}

function getBucketLabel(i) {
    const names = State.get('bucketNames');
    if (names[i]) return `★ ${names[i]}`;
    return i === 0 ? '★ Bucket' : `★ Bucket ${i + 1}`;
}

function clearBucketDetail() {
    State.update({ selectedExp: null, selectedExpIds: null });
    document.getElementById('experiment-details').innerHTML = '';
    document.getElementById('meta-row-container').innerHTML = '';
    document.getElementById('detail-title').textContent = '';
}

function toggleBucketItem(expId, bucketIndex = 0) {
    const bucket = getBucket(bucketIndex);
    const idx = bucket.indexOf(expId);
    if (idx >= 0) {
        bucket.splice(idx, 1);
    } else {
        // Remove from any other bucket first (run can only be in one)
        const buckets = State.get('buckets');
        for (let i = 0; i < buckets.length; i++) {
            if (i === bucketIndex) continue;
            const pos = buckets[i].indexOf(expId);
            if (pos >= 0) { buckets[i].splice(pos, 1); break; }
        }
        State.set('buckets', buckets);
        bucket.push(expId);
    }
    saveBucket(bucketIndex, bucket);
    renderBucketCards();
    const viewingIdx = parseBucketKey(State.get('selectedExp'));
    if (viewingIdx >= 0) {
        const viewingBucket = getBucket(viewingIdx);
        if (viewingBucket.length === 0) clearBucketDetail();
        else selectExperiment(bucketKey(viewingIdx), viewingBucket.slice());
    }
}

function removeFromBucket(expId) {
    const viewingIdx = parseBucketKey(State.get('selectedExp'));
    const bucket = getBucket(viewingIdx).filter(id => id !== expId);
    saveBucket(viewingIdx, bucket);
    renderBucketCards();
    if (bucket.length === 0) clearBucketDetail();
    else selectExperiment(bucketKey(viewingIdx), bucket.slice());
}

function renameBucket(i, el) {
    event.stopPropagation();
    const current = State.get('bucketNames')[i] || '';
    const input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    input.className = 'bucket-rename-input';
    input.placeholder = i === 0 ? 'Bucket' : `Bucket ${i + 1}`;

    const commit = () => {
        const val = input.value.trim();
        const names = State.get('bucketNames');
        names[i] = val;
        State.set('bucketNames', names);
        renderBucketCards();
        if (State.get('selectedExp') === bucketKey(i)) {
            document.getElementById('detail-title').textContent = getBucketLabel(i);
        }
    };

    input.addEventListener('keydown', e => {
        e.stopPropagation();
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { e.preventDefault(); renderBucketCards(); }
    });
    input.addEventListener('blur', commit);
    input.addEventListener('click', e => e.stopPropagation());

    el.replaceWith(input);
    input.focus();
    input.select();
}

function renderBucketCards() {
    document.querySelectorAll('.bucket-card-item').forEach(el => el.remove());
    const listEl = document.getElementById('experiments-list');
    if (!listEl) return;

    const buckets = State.get('buckets');
    const selectedExp = State.get('selectedExp');

    for (let i = buckets.length - 1; i >= 0; i--) {
        const bucket = buckets[i];
        if (bucket.length === 0) continue;
        const key = bucketKey(i);
        const card = document.createElement('div');
        card.className = `experiment-card bucket bucket-card-item${selectedExp === key ? ' selected' : ''}`;
        card.dataset.id = key;
        card.onclick = () => selectExperiment(key, bucket.slice());
        card.innerHTML = `
            <div class="exp-header">
                <div class="exp-name-group">
                    <span class="bucket-num">${i + 1}</span>
                    <span class="exp-name"><span class="exp-track bucket-label" ondblclick="renameBucket(${i}, this)">${getBucketLabel(i)}</span></span>
                </div>
                <div class="exp-badges">
                    <span class="bucket-count">${bucket.length} run${bucket.length !== 1 ? 's' : ''}</span>
                </div>
            </div>
        `;
        listEl.prepend(card);
    }
}

function renderBucketCard() { renderBucketCards(); }

function handleRunRowClick(event, expId) {
    const digit = getHeldDigit();
    if (!(event.metaKey || event.ctrlKey) && !digit) {
        showMetricsForRun(expId);
        return;
    }
    event.preventDefault();
    // Clear digit after use so it doesn't stick
    _lastDigit = null;

    const viewingIdx = parseBucketKey(State.get('selectedExp'));
    const inBucketView = viewingIdx >= 0;

    if (digit && inBucketView && digit - 1 !== viewingIdx) {
        // Move run from current bucket to target bucket
        const src = getBucket(viewingIdx).filter(id => id !== expId);
        saveBucket(viewingIdx, src);
        const dst = getBucket(digit - 1);
        if (!dst.includes(expId)) dst.push(expId);
        saveBucket(digit - 1, dst);
        renderBucketCards();
        if (src.length === 0) clearBucketDetail();
        else selectExperiment(bucketKey(viewingIdx), src.slice());
    } else {
        const idx = digit ? digit - 1 : (inBucketView ? viewingIdx : 0);
        toggleBucketItem(expId, idx);
        if (!inBucketView) renderRunsTable();
    }
}

function showMetricsForRun(expId) {
    const currentValidData = State.get('experimentData');
    if (!currentValidData) return;

    // Clear cell selection when clicking individual rows
    State.set('selectedCellExpIds', null);

    // Toggle: if same run clicked again, deselect it
    if (State.get('selectedRun') === expId) {
        State.set('selectedRun', null);
        // Show averaged metrics if multiple runs, or single run metrics
        if (currentValidData.length > 1) {
            showAveragedMetrics();
        } else {
            updateMetricsTable(currentValidData[0].loss_curve, false, null);
        }
        updateRunRowHighlights();
        updateHeatmapHighlights();
        return;
    }

    const exp = currentValidData.find(d => d.id === expId);
    if (exp) {
        State.set('selectedRun', expId);
        updateMetricsTable(exp.loss_curve, false, expId);
        updateRunRowHighlights();
        updateHeatmapHighlights();
    }
}

function showAveragedMetrics() {
    const currentValidData = State.get('experimentData');
    if (!currentValidData) return;
    const allCurves = currentValidData.map(d => d.loss_curve);
    const averaged = computeAveragedMetrics(allCurves);
    updateMetricsTable(averaged, true, null);
}

function showAveragedMetricsForCell(expIds) {
    const currentValidData = State.get('experimentData');
    if (!currentValidData) return;
    const cellData = currentValidData.filter(d => expIds.includes(d.id));
    if (cellData.length === 0) return;
    const allCurves = cellData.map(d => d.loss_curve);
    const averaged = computeAveragedMetrics(allCurves);
    updateMetricsTable(averaged, true, null);
}

function updateRunRowHighlights() {
    const selectedRunId = State.get('selectedRun');
    const selectedCellExpIds = State.get('selectedCellExpIds');
    document.querySelectorAll('.run-row').forEach(row => {
        const rowExpId = parseInt(row.dataset.expId);
        const isSelected = selectedCellExpIds
            ? selectedCellExpIds.includes(rowExpId)
            : rowExpId === selectedRunId;
        row.classList.toggle('selected', isSelected);
    });
}

function updateHeatmapHighlights() {
    const selectedRunId = State.get('selectedRun');
    const selectedCellExpIds = State.get('selectedCellExpIds');
    document.querySelectorAll('.heatmap-cell').forEach(cell => {
        const expIds = (cell.dataset.expIds || '').split(',').map(id => parseInt(id)).filter(id => !isNaN(id));
        const isSelected = selectedCellExpIds
            ? expIds.some(id => selectedCellExpIds.includes(id))
            : expIds.includes(selectedRunId);
        cell.classList.toggle('selected', isSelected);
    });
}

function toggleRunsSort(col) {
    const runsSort = State.get('runsSort');
    if (runsSort.col === col) {
        runsSort.dir = runsSort.dir === 'asc' ? 'desc' : 'asc';
    } else {
        runsSort.col = col;
        runsSort.dir = col === 'started_at' ? 'desc' : 'asc';
    }
    State.set('runsSort', runsSort);
    renderRunsTable();
}

function toggleEnvFilter(key, value) {
    const runsEnvFilters = State.get('runsEnvFilters');
    if (runsEnvFilters[key] === value) {
        delete runsEnvFilters[key];
    } else {
        runsEnvFilters[key] = value;
    }
    State.set('runsEnvFilters', runsEnvFilters);
    renderRunsTable();
}

function clearEnvFilters() {
    State.set('runsEnvFilters', {});
    renderRunsTable();
}

function setRunsLimit(val) {
    const n = parseInt(val);
    if (!isNaN(n) && n > 0) State.set('runsLimit', n);
    renderRunsTable();
}

function makeRunsLimitEditable(el) {
    const current = State.get('runsLimit');
    const input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    input.className = 'runs-limit-inline';
    input.size = String(current).length || 2;

    const commit = () => {
        const n = parseInt(input.value);
        if (!isNaN(n) && n > 0) {
            State.set('runsLimit', n);
        }
        renderRunsTable();
    };

    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); commit(); }
        if (e.key === 'Escape') { e.preventDefault(); renderRunsTable(); }
    });
    input.addEventListener('blur', commit);

    el.replaceWith(input);
    input.focus();
    input.select();
}

function getVisibleRuns() {
    const currentValidData = State.get('experimentData');
    if (!currentValidData || currentValidData.length === 0) return [];
    let data = currentValidData.slice();

    const runsEnvFilters = State.get('runsEnvFilters');
    const runsSort = State.get('runsSort');
    const runsLimit = State.get('runsLimit');

    // Apply env var filters
    const filterKeys = Object.keys(runsEnvFilters);
    if (filterKeys.length > 0) {
        data = data.filter(d => {
            const env = d.env_vars || {};
            return filterKeys.every(k => String(env[k] ?? '') === String(runsEnvFilters[k]));
        });
    }

    // Sort
    const col = runsSort.col;
    const dir = runsSort.dir === 'asc' ? 1 : -1;
    data.sort((a, b) => {
        let va, vb;
        if (col === 'started_at') {
            va = a.started_at || ''; vb = b.started_at || '';
        } else if (col === 'val_loss') {
            va = a.final_val_loss ?? Infinity; vb = b.final_val_loss ?? Infinity;
        } else if (col === 'time') {
            va = a.final_train_time_ms ?? Infinity; vb = b.final_train_time_ms ?? Infinity;
        } else if (col === 'status') {
            va = a.status || ''; vb = b.status || '';
        } else if (col === 'id') {
            va = a.id; vb = b.id;
        } else if (col === 'script') {
            va = a.script || ''; vb = b.script || '';
        } else {
            va = 0; vb = 0;
        }
        return va < vb ? -dir : va > vb ? dir : 0;
    });

    return data.slice(0, runsLimit);
}

function renderRunsTable() {
    const currentValidData = State.get('experimentData');
    if (!currentValidData) return;
    const container = document.getElementById('runs-table-container');
    const filtersEl = document.getElementById('runs-filters');
    if (!container) return;

    const runsEnvFilters = State.get('runsEnvFilters');
    const runsSort = State.get('runsSort');
    const runsLimit = State.get('runsLimit');
    const selectedRunId = State.get('selectedRun');
    const isBucketView = isBucketKey(State.get('selectedExp'));

    const bucketSet = getAllBucketExpIds();
    const limited = getVisibleRuns();
    const filterKeys = Object.keys(runsEnvFilters);
    // Total before limit (need to recount for "Showing X of Y")
    let totalFiltered = currentValidData.length;
    if (filterKeys.length > 0) {
        totalFiltered = currentValidData.filter(d => {
            const env = d.env_vars || {};
            return filterKeys.every(k => String(env[k] ?? '') === String(runsEnvFilters[k]));
        }).length;
    }
    const total = totalFiltered;

    // Sort indicator helper
    const sortCls = (c) => runsSort.col === c ? 'sortable sort-active' : 'sortable';
    const sortArrow = (c) => runsSort.col === c ? (runsSort.dir === 'asc' ? ' ▲' : ' ▼') : '';

    // Render active filters bar
    if (filtersEl) {
        if (filterKeys.length > 0) {
            filtersEl.innerHTML = `<div class="runs-active-filters">
                ${filterKeys.map(k => `<span class="runs-filter-chip active" onclick="toggleEnvFilter('${k}', '${runsEnvFilters[k]}')">${k}=${runsEnvFilters[k]} ✕</span>`).join('')}
                <span class="runs-filter-clear" onclick="clearEnvFilters()">Clear all</span>
            </div>`;
        } else {
            filtersEl.innerHTML = '';
        }
    }

    // Show session column if runs span multiple sessions
    const showSession = new Set(limited.map(d => d.session_name).filter(Boolean)).size > 1;

    // Render table
    container.innerHTML = `
        <div class="runs-table-controls">
            <span class="runs-count">Showing <span class="runs-limit-editable" onclick="makeRunsLimitEditable(this)" title="Click to edit">${Math.min(runsLimit, total)}</span> of ${total} runs</span>
        </div>
        <table>
        <thead>
            <tr>
                ${isBucketView ? `<th class="${sortCls('script')}" onclick="toggleRunsSort('script')">Script</th>` : `<th class="${sortCls('id')}" onclick="toggleRunsSort('id')">Exp</th>`}
                <th>Run ID</th>
                <th class="${sortCls('status')}" onclick="toggleRunsSort('status')">Status</th>
                ${showSession ? '<th>Session</th>' : ''}
                <th class="${sortCls('started_at')}" onclick="toggleRunsSort('started_at')">Started</th>
                ${runsIsSweep ? `<th>Env</th>` : ''}
                <th class="${sortCls('val_loss')}" onclick="toggleRunsSort('val_loss')">Val Loss</th>
                <th class="${sortCls('time')}" onclick="toggleRunsSort('time')">Time</th>
                <th></th>
            </tr>
        </thead>
        <tbody>
            ${limited.map(d => {
                const envEntries = Object.entries(d.env_vars || {});
                const envHtml = runsIsSweep ? `<td class="sweep-env">${envEntries.map(([k,v]) => {
                    const isFiltered = runsEnvFilters[k] === String(v);
                    return `<span class="env-chip${isFiltered ? ' active' : ''}" onclick="event.stopPropagation(); toggleEnvFilter('${k}', '${v}')">${k}=${v}</span>`;
                }).join(' ') || 'default'}</td>` : '';
                const sessionHtml = showSession ? `<td class="session-col">${d.session_name || '-'}</td>` : '';
                return `
                <tr class="run-row${selectedRunId === d.id ? ' selected' : ''}${bucketSet.has(d.id) ? ' in-bucket' : ''}" data-exp-id="${d.id}" onclick="handleRunRowClick(event, ${d.id})" title="Click to view metrics · ⌘+click bucket 1 · hold N+click bucket N">
                    <td>${isBucketView ? `<span class="exp-track">${d.track || 'untracked'}</span>/${d.script ? d.script.split('/').pop().replace('.py', '') : d.name}` : `#${d.id}`}</td>
                    <td>
                        ${d.remote_run_id
                            ? `<span class="meta-run-id copyable" title="Click to copy · ⌘+click to open log" onclick="event.stopPropagation(); if (event.metaKey || event.ctrlKey) { window.open('/api/logs/${d.remote_run_id}', '_blank'); } else { copyToClipboard('${d.remote_run_id}', this); }">${d.remote_run_id}</span>`
                            : '<span class="meta-hash-missing">none</span>'}${d.status === 'failed'
                            ? `<span class="log-sep">|</span><a href="/api/crash/${d.id}" target="_blank" class="crash-log-link" onclick="event.stopPropagation()">CRASH LOG</a>`
                            : ''}
                    </td>
                    <td><span class="status-badge-sm ${d.status}">${d.status}</span></td>
                    ${sessionHtml}
                    <td class="started-at">${formatStartedAt(d.started_at)}</td>
                    ${envHtml}
                    <td class="val-loss">${d.final_val_loss ? d.final_val_loss.toFixed(4) : 'n/a'}</td>
                    <td>${d.final_train_time_ms ? (d.final_train_time_ms/1000).toFixed(1) + 's' : 'n/a'}</td>
                    <td>${isBucketView
                        ? `<button class="delete-icon bucket-remove" onclick="event.stopPropagation(); removeFromBucket(${d.id})" title="Remove from bucket"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg></button>`
                        : `<button class="delete-icon" onclick="event.stopPropagation(); deleteExperiment(${d.id}, '${d.name.replace(/'/g, "\\'")}', this.closest('tr'))" title="Delete"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg></button>`}</td>
                </tr>`;
            }).join('')}
        </tbody>
        </table>`;

    // Update chart to reflect filtered data
    const cv = State.get('chartView');
    if (cv) switchChartView(cv, false);
}

function updateMetricsTable(lossData, isAveraged = false, selectedExpId = null) {
    const tableEl = document.getElementById('metrics-table');
    const currentValidData = State.get('experimentData');

    if (!lossData || lossData.length === 0) {
        tableEl.innerHTML = '<h3>Metrics History</h3><p class="placeholder">No metrics recorded yet</p>';
        currentMetricsData = null;
        return;
    }

    // Show most recent first
    const recentData = lossData.slice().reverse();
    const isMultiple = currentValidData && currentValidData.length > 1;

    // Store for clipboard copy (full data, not just recent)
    currentMetricsData = lossData.slice().reverse();

    tableEl.innerHTML = `
        <h3 class="metrics-header-clickable" onclick="copyMetricsAsMarkdown()" title="Click to copy as markdown">Metrics History</h3>
        ${isMultiple ? `<div class="metrics-header">
            <span class="metrics-label">${isAveraged ? 'Averaged across ' + currentValidData.length + ' runs' : 'Run #' + selectedExpId}</span>
        </div>` : ''}
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Val Loss</th>
                    <th>Time</th>
                    <th>Step Avg</th>
                </tr>
            </thead>
            <tbody>
                ${recentData.map(m => `
                    <tr>
                        <td>${m.step}</td>
                        <td class="val-loss">${m.val_loss ? m.val_loss.toFixed(4) : '-'}</td>
                        <td>${m.train_time_ms ? formatTime(m.train_time_ms) : '-'}</td>
                        <td>${m.step_avg_ms ? m.step_avg_ms.toFixed(1) + 'ms' : '-'}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        ${lossData.length > 20 ? `<p class="table-note">${lossData.length - 20} earlier checkpoints not shown</p>` : ''}
    `;
}

function copyMetricsAsMarkdown() {
    if (!currentMetricsData || currentMetricsData.length === 0) return;

    // Build markdown table
    const lines = [
        '| Step | Val Loss | Time | Step Avg |',
        '|------|----------|------|----------|'
    ];

    currentMetricsData.forEach(m => {
        const step = m.step;
        const valLoss = m.val_loss ? m.val_loss.toFixed(4) : '-';
        const time = m.train_time_ms ? formatTime(m.train_time_ms) : '-';
        const stepAvg = m.step_avg_ms ? m.step_avg_ms.toFixed(1) + 'ms' : '-';
        lines.push(`| ${step} | ${valLoss} | ${time} | ${stepAvg} |`);
    });

    const markdown = lines.join('\n');
    copyToClipboard(markdown, document.querySelector('.metrics-header-clickable'));
}

function copyRunsAsMarkdown() {
    const runs = getVisibleRuns();
    if (runs.length === 0) return;

    const isBucketView = isBucketKey(State.get('selectedExp'));
    const hasEnvVars = runs.some(d => d.env_vars && Object.keys(d.env_vars).length > 0);

    const statusLabel = (s) => s === 'running' ? 'in progress' : s === 'completed' ? 'success' : s;

    let header, separator;
    if (isBucketView && hasEnvVars) {
        header = '| Script | Run | Status | Env | Val Loss | Time |';
        separator = '|--------|-----|--------|-----|----------|------|';
    } else if (isBucketView) {
        header = '| Script | Run | Status | Val Loss | Time |';
        separator = '|--------|-----|--------|----------|------|';
    } else if (hasEnvVars) {
        header = '| Run | Status | Env | Val Loss | Time |';
        separator = '|-----|--------|-----|----------|------|';
    } else {
        header = '| Run | Status | Val Loss | Time |';
        separator = '|-----|--------|----------|------|';
    }
    const lines = [header, separator];

    runs.forEach(d => {
        const runId = d.remote_run_id || `#${d.id}`;
        const status = statusLabel(d.status || 'unknown');
        const valLoss = d.final_val_loss ? d.final_val_loss.toFixed(4) : '-';
        const time = d.final_train_time_ms ? (d.final_train_time_ms / 1000).toFixed(1) + 's' : '-';
        const script = d.script ? d.script.split('/').pop().replace('.py', '') : d.name;
        const env = Object.entries(d.env_vars || {}).map(([k, v]) => `${k}=${v}`).join(', ') || '-';

        if (isBucketView && hasEnvVars) {
            lines.push(`| ${script} | ${runId} | ${status} | ${env} | ${valLoss} | ${time} |`);
        } else if (isBucketView) {
            lines.push(`| ${script} | ${runId} | ${status} | ${valLoss} | ${time} |`);
        } else if (hasEnvVars) {
            lines.push(`| ${runId} | ${status} | ${env} | ${valLoss} | ${time} |`);
        } else {
            lines.push(`| ${runId} | ${status} | ${valLoss} | ${time} |`);
        }
    });

    const markdown = lines.join('\n');
    copyToClipboard(markdown, document.querySelector('.runs-header-clickable'));
}

async function deleteExperiment(expId, expName, rowElement) {
    // Prevent auto-refresh from racing with delete
    State.set('deleteInProgress', true);

    // Optimistic update: remove from state and re-render immediately
    let selectedExpIds = State.get('selectedExpIds');
    if (selectedExpIds) {
        selectedExpIds = selectedExpIds.filter(id => id !== expId);
        State.set('selectedExpIds', selectedExpIds);
    }

    let currentValidData = State.get('experimentData');
    if (currentValidData) {
        currentValidData = currentValidData.filter(d => d.id !== expId);
        State.set('experimentData', currentValidData);
    }

    if (State.get('selectedRun') === expId) {
        State.set('selectedRun', null);
    }

    if (!selectedExpIds || selectedExpIds.length === 0) {
        State.update({ selectedExp: null, selectedExpIds: null });
        State.set('experimentData', null);
    }
    renderRunsTable();

    // Fire delete request (don't block UI on it)
    try {
        const response = await fetch(`/api/experiment/${expId}`, { method: 'DELETE' });
        const data = await response.json();

        if (data.success) {
            refreshExperiments();
        } else {
            alert('Failed to delete: ' + (data.error || 'Unknown error'));
            // TODO: could roll back optimistic update here, but unlikely to fail
        }
    } finally {
        State.set('deleteInProgress', false);
    }
}
