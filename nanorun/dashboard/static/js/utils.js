// Utility functions for nanorun dashboard

function formatTime(ms) {
    if (!ms) return '';
    const seconds = ms / 1000;
    if (seconds >= 1000) {
        return Math.round(seconds) + 's';
    }
    return seconds.toPrecision(3) + 's';
}

function formatStartedAt(isoString) {
    if (!isoString) return '-';
    if (!/Z$/.test(isoString) && !/[+-]\d{2}:\d{2}$/.test(isoString)) {
        isoString += 'Z';
    }
    const date = new Date(isoString);
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    const hour24 = date.getHours();
    const hour12 = hour24 % 12 || 12;
    const ampm = hour24 < 12 ? 'am' : 'pm';
    const mins = date.getMinutes().toString().padStart(2, '0');
    return `${month}/${day} ${hour12}:${mins}${ampm}`;
}

// Short column/label text for a run's primary loss series. `loss_metric` is
// inferred server-side from the rows a run actually produced: 'val_loss' when
// it reports a validation loss, 'train_loss' when training loss is all it has.
function lossLabel(loss_metric, short = false) {
    if (loss_metric === 'train_loss') return short ? 'Train' : 'Train Loss';
    return short ? 'Val' : 'Val Loss';
}

// Column heading for a set of runs. Bucket views can mix scripts that report
// different series, in which case neither label is honest — fall back to 'Loss'.
function lossColumnLabel(items) {
    const metrics = new Set((items || []).map(d => d && d.loss_metric).filter(Boolean));
    if (metrics.size > 1) return 'Loss';
    return lossLabel(metrics.size === 1 ? [...metrics][0] : undefined);
}

function getAvailableLossMetrics(items) {
    const available = new Set();
    (items || []).forEach(item => {
        if (!item) return;
        (item.available_loss_metrics || []).forEach(metric => available.add(metric));
        if (item.loss_curves) {
            for (const metric of ['val_loss', 'train_loss']) {
                if ((item.loss_curves[metric] || []).length > 0) available.add(metric);
            }
        }
        // Backward compatibility with detail responses cached before loss_curves.
        if (item.loss_metric && (item.loss_curve || []).length > 0) {
            available.add(item.loss_metric);
        }
    });
    return ['val_loss', 'train_loss'].filter(metric => available.has(metric));
}

function getActiveLossMetric(items) {
    const available = getAvailableLossMetrics(items);
    const selected = State.get('selectedLossMetric');
    if (available.includes(selected)) return selected;

    const primary = (items || []).map(item => item && item.loss_metric)
        .find(metric => available.includes(metric));
    return primary || available[0] || null;
}

function getLossCurve(item, metric = null) {
    if (!item) return [];
    const selected = metric || State.get('selectedLossMetric') || getActiveLossMetric([item]);
    if (item.loss_curves && selected && Array.isArray(item.loss_curves[selected])) {
        return item.loss_curves[selected];
    }
    return selected === item.loss_metric ? (item.loss_curve || []) : [];
}

function formatLoss(loss, train_time_ms, loss_metric) {
    if (loss == null) return '<span class="val-loss-value">n/a</span>';
    // Train-loss runs get a marker so they're never mistaken for val numbers.
    const tag = loss_metric === 'train_loss' ? '<span class="loss-metric-tag">train</span>' : '';
    const value = `<span class="val-loss-value">${loss.toFixed(4)}</span>${tag}`;
    if (train_time_ms) {
        return `${value}<span class="val-loss-at">@</span><span class="val-loss-time">${formatTime(train_time_ms)}</span>`;
    }
    return value;
}

function renderDiff(diffText) {
    if (!diffText || diffText.trim() === '') {
        return '<p class="placeholder">Empty diff</p>';
    }
    const lines = diffText.split(/\r?\n/);
    const htmlLines = lines.map(line => {
        let className = 'diff-line';
        if (line.startsWith('+++') || line.startsWith('---')) {
            className += ' diff-file';
        } else if (line.startsWith('@@')) {
            className += ' diff-hunk';
        } else if (line.startsWith('+')) {
            className += ' diff-add';
        } else if (line.startsWith('-')) {
            className += ' diff-del';
        }
        const escaped = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        return `<div class="${className}">${escaped}</div>`;
    });
    return `<pre class="diff-pre">${htmlLines.join('')}</pre>`;
}

async function copyToClipboard(text, element) {
    try {
        await navigator.clipboard.writeText(text);
        if (element) {
            element.classList.add('copied');
            setTimeout(() => element.classList.remove('copied'), 400);
        }
    } catch (e) {
        console.error('Failed to copy:', e);
    }
}

function _themeColor(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

function isMobile() {
    return window.innerWidth < 768;
}

function detectStepInterval(runs) {
    for (const run of runs) {
        if (run.data && run.data.length >= 2) {
            // Loss curves arrive sorted by step, so count intervals in one pass
            // without cloning and sorting every point on each chart refresh.
            const counts = {};
            for (let i = 1; i < run.data.length; i++) {
                const interval = run.data[i].step - run.data[i - 1].step;
                if (interval > 0) counts[interval] = (counts[interval] || 0) + 1;
            }
            if (Object.keys(counts).length > 0) {
                const mode = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
                if (mode) return parseInt(mode[0]);
            }
        }
    }
    return 250;
}

function getDifferingKeys(runs) {
    const allKeys = new Set();
    runs.forEach(r => Object.keys(r.env_vars || {}).forEach(k => allKeys.add(k)));
    const differing = [];
    for (const key of allKeys) {
        const vals = new Set(runs.map(r => String((r.env_vars || {})[key] ?? '')));
        if (vals.size > 1) differing.push(key);
    }
    return differing;
}

function computeSmartLabels(runs) {
    if (runs.length <= 1) return runs.map(() => 'Validation Loss');
    const differingKeys = getDifferingKeys(runs);
    const inBucket = isBucketKey(State.get('selectedExp'));
    return runs.map((run, i) => {
        const env = run.env_vars || {};
        const scriptLabel = run.script ? run.script.split('/').pop().replace('.py', '') : run.name;
        if (differingKeys.length === 0) return inBucket ? scriptLabel : `Run ${i + 1}`;
        const parts = differingKeys.map(k => env[k] ?? '').join(', ');
        return inBucket ? `${scriptLabel} (${parts})` : parts;
    });
}

function computeAveragedMetrics(allLossCurves) {
    const byStep = {};
    allLossCurves.forEach(curve => {
        (curve || []).forEach(m => {
            if (!byStep[m.step]) {
                byStep[m.step] = { val_losses: [], train_times: [], step_avgs: [] };
            }
            // `loss` is the run's primary series; fall back to val_loss so this
            // still works on a cached response from before loss_metric existed.
            const loss = m.loss != null ? m.loss : m.val_loss;
            if (loss != null) byStep[m.step].val_losses.push(loss);
            if (m.train_time_ms != null) byStep[m.step].train_times.push(m.train_time_ms);
            if (m.step_avg_ms != null) byStep[m.step].step_avgs.push(m.step_avg_ms);
        });
    });
    const avgMetrics = Object.entries(byStep).map(([step, data]) => {
        const mean = data.val_losses.length ? data.val_losses.reduce((a,b) => a+b, 0) / data.val_losses.length : null;
        return {
            step: parseInt(step),
            loss: mean,
            val_loss: mean,
            train_time_ms: data.train_times.length ? data.train_times.reduce((a,b) => a+b, 0) / data.train_times.length : null,
            step_avg_ms: data.step_avgs.length ? data.step_avgs.reduce((a,b) => a+b, 0) / data.step_avgs.length : null,
            n: data.val_losses.length
        };
    });
    return avgMetrics.sort((a, b) => a.step - b.step);
}
