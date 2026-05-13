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

function formatValLoss(val_loss, train_time_ms) {
    if (!val_loss) return '<span class="val-loss-value">n/a</span>';
    const loss = val_loss.toFixed(4);
    if (train_time_ms) {
        return `<span class="val-loss-value">${loss}</span><span class="val-loss-at">@</span><span class="val-loss-time">${formatTime(train_time_ms)}</span>`;
    }
    return `<span class="val-loss-value">${loss}</span>`;
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
            const steps = run.data.map(d => d.step).sort((a, b) => a - b);
            const intervals = [];
            for (let i = 1; i < steps.length; i++) {
                const interval = steps[i] - steps[i - 1];
                if (interval > 0) intervals.push(interval);
            }
            if (intervals.length > 0) {
                const counts = {};
                intervals.forEach(i => counts[i] = (counts[i] || 0) + 1);
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
    return runs.map((run, i) => {
        const env = run.env_vars || {};
        const scriptLabel = run.script ? run.script.split('/').pop().replace('.py', '') : run.name;
        if (differingKeys.length === 0) {
            return State.get('selectedExp') === 'bucket' ? scriptLabel : `Run ${i + 1}`;
        }
        const parts = differingKeys.map(k => env[k] ?? '').join(', ');
        return State.get('selectedExp') === 'bucket' ? `${scriptLabel} (${parts})` : parts;
    });
}

function computeAveragedMetrics(allLossCurves) {
    const byStep = {};
    allLossCurves.forEach(curve => {
        (curve || []).forEach(m => {
            if (!byStep[m.step]) {
                byStep[m.step] = { val_losses: [], train_times: [], step_avgs: [] };
            }
            if (m.val_loss != null) byStep[m.step].val_losses.push(m.val_loss);
            if (m.train_time_ms != null) byStep[m.step].train_times.push(m.train_time_ms);
            if (m.step_avg_ms != null) byStep[m.step].step_avgs.push(m.step_avg_ms);
        });
    });
    const avgMetrics = Object.entries(byStep).map(([step, data]) => ({
        step: parseInt(step),
        val_loss: data.val_losses.length ? data.val_losses.reduce((a,b) => a+b, 0) / data.val_losses.length : null,
        train_time_ms: data.train_times.length ? data.train_times.reduce((a,b) => a+b, 0) / data.train_times.length : null,
        step_avg_ms: data.step_avgs.length ? data.step_avgs.reduce((a,b) => a+b, 0) / data.step_avgs.length : null,
        n: data.val_losses.length
    }));
    return avgMetrics.sort((a, b) => a.step - b.step);
}
