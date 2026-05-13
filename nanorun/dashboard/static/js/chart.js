// Chart rendering for nanorun dashboard

let lossChart = null;
let _chartOriginalColors = [];

const CHART_COLORS = ['#5a9a5e', '#5a8ab8', '#c9a054', '#b05a7a', '#7a5a9a', '#5a9a9a', '#c9b854', '#8a7058'];

function computeHeatmapData(validData, selectedVars = null) {
    const completedData = validData.filter(d => d.status === 'completed');
    if (completedData.length < 1) return null;

    const envVarValues = {};
    completedData.forEach(exp => {
        const envVars = exp.env_vars || {};
        Object.entries(envVars).forEach(([key, value]) => {
            if (!envVarValues[key]) envVarValues[key] = new Set();
            envVarValues[key].add(String(value));
        });
    });

    const sortValues = (values) => {
        const arr = Array.from(values);
        const allNumeric = arr.every(v => !isNaN(parseFloat(v)));
        if (allNumeric) {
            return arr.sort((a, b) => parseFloat(a) - parseFloat(b));
        }
        return arr.sort();
    };

    const sweptVars = Object.entries(envVarValues)
        .filter(([key, values]) => values.size > 1)
        .map(([key, values]) => ({ key, values: sortValues(values) }));

    if (sweptVars.length === 0) return null;

    let activeVars;
    if (selectedVars && selectedVars.length > 0) {
        activeVars = selectedVars
            .filter(name => sweptVars.some(v => v.key === name))
            .slice(0, 2);
    } else {
        activeVars = sweptVars.slice(0, 2).map(v => v.key);
    }

    if (activeVars.length === 0) return null;

    const sweptVarOrder = sweptVars.map(v => v.key);
    activeVars.sort((a, b) => sweptVarOrder.indexOf(a) - sweptVarOrder.indexOf(b));

    const getVarInfo = (name) => sweptVars.find(v => v.key === name);

    const allLosses = completedData.map(e => e.final_val_loss).filter(v => v != null);
    if (allLosses.length === 0) return null;
    const minLoss = Math.min(...allLosses);
    const maxLoss = Math.max(...allLosses);

    // 1D case
    if (activeVars.length === 1) {
        const varX = getVarInfo(activeVars[0]);
        const cellLosses = {};
        const cellExps = {};

        completedData.forEach(exp => {
            const envVars = exp.env_vars || {};
            const xVal = envVars[varX.key] !== undefined ? String(envVars[varX.key]) : undefined;
            if (xVal !== undefined) {
                if (!cellLosses[xVal]) cellLosses[xVal] = [];
                if (!cellExps[xVal]) cellExps[xVal] = [];
                if (exp.final_val_loss != null) {
                    cellLosses[xVal].push(exp.final_val_loss);
                }
                cellExps[xVal].push(exp);
            }
        });

        const values = {};
        const cellData = {};
        varX.values.forEach(xVal => {
            const losses = cellLosses[xVal] || [];
            values[xVal] = losses.length > 0
                ? losses.reduce((a, b) => a + b, 0) / losses.length
                : null;
            cellData[xVal] = {
                exp: cellExps[xVal]?.[0],
                count: cellExps[xVal]?.length || 0,
                allExps: cellExps[xVal] || []
            };
        });

        return {
            dim: 1,
            varX: varX.key,
            xValues: varX.values,
            values,
            cellData,
            minLoss,
            maxLoss,
            sweptVars
        };
    }

    // 2D case
    const varX = getVarInfo(activeVars[0]);
    const varY = getVarInfo(activeVars[1]);

    const cellLosses = {};
    const cellExps = {};

    completedData.forEach(exp => {
        const envVars = exp.env_vars || {};
        const xVal = envVars[varX.key] !== undefined ? String(envVars[varX.key]) : undefined;
        const yVal = envVars[varY.key] !== undefined ? String(envVars[varY.key]) : undefined;
        if (xVal !== undefined && yVal !== undefined) {
            if (!cellLosses[yVal]) cellLosses[yVal] = {};
            if (!cellExps[yVal]) cellExps[yVal] = {};
            if (!cellLosses[yVal][xVal]) cellLosses[yVal][xVal] = [];
            if (!cellExps[yVal][xVal]) cellExps[yVal][xVal] = [];
            if (exp.final_val_loss != null) {
                cellLosses[yVal][xVal].push(exp.final_val_loss);
            }
            cellExps[yVal][xVal].push(exp);
        }
    });

    const matrix = {};
    const cellData = {};
    varY.values.forEach(yVal => {
        matrix[yVal] = {};
        cellData[yVal] = {};
        varX.values.forEach(xVal => {
            const losses = cellLosses[yVal]?.[xVal] || [];
            matrix[yVal][xVal] = losses.length > 0
                ? losses.reduce((a, b) => a + b, 0) / losses.length
                : null;
            cellData[yVal][xVal] = {
                exp: cellExps[yVal]?.[xVal]?.[0],
                count: cellExps[yVal]?.[xVal]?.length || 0,
                allExps: cellExps[yVal]?.[xVal] || []
            };
        });
    });

    return {
        dim: 2,
        varX: varX.key,
        varY: varY.key,
        xValues: varX.values,
        yValues: varY.values,
        matrix,
        cellData,
        minLoss,
        maxLoss,
        sweptVars
    };
}

function getHeatmapColor(valLoss, minLoss, maxLoss) {
    if (valLoss == null) return 'var(--bg-dark)';

    const range = maxLoss - minLoss;
    const normalized = range > 0 ? (valLoss - minLoss) / range : 0.5;

    const style = getComputedStyle(document.documentElement);
    const good = style.getPropertyValue('--heatmap-good').trim().split(',').map(Number);
    const bad = style.getPropertyValue('--heatmap-bad').trim().split(',').map(Number);

    const r = Math.round(good[0] + normalized * (bad[0] - good[0]));
    const g = Math.round(good[1] + normalized * (bad[1] - good[1]));
    const b = Math.round(good[2] + normalized * (bad[2] - good[2]));

    return `rgb(${r}, ${g}, ${b})`;
}

function renderHeatmapControls(sweptVars, selectedVars) {
    if (!sweptVars || sweptVars.length < 2) return '';

    return `
        <div class="heatmap-var-toggles">
            ${sweptVars.map(v => {
                const isActive = selectedVars.includes(v.key);
                return `<button class="heatmap-var-toggle${isActive ? ' active' : ''}"
                    onclick="toggleHeatmapVar('${v.key}')">${v.key}</button>`;
            }).join('')}
        </div>
    `;
}

function toggleHeatmapVar(varName) {
    const heatmapSelectedVars = State.get('heatmapSelectedVars');
    const idx = heatmapSelectedVars.indexOf(varName);
    if (idx >= 0) {
        if (heatmapSelectedVars.length > 1) {
            heatmapSelectedVars.splice(idx, 1);
        }
    } else {
        if (heatmapSelectedVars.length >= 2) {
            heatmapSelectedVars.shift();
        }
        heatmapSelectedVars.push(varName);
    }
    State.set('heatmapSelectedVars', heatmapSelectedVars);
    switchChartView('heatmap', false);
}

function renderHeatmap(heatmapData) {
    if (!heatmapData) return '';

    const { dim, sweptVars } = heatmapData;

    const selectedVarNames = dim === 1
        ? [heatmapData.varX]
        : [heatmapData.varX, heatmapData.varY];

    const controls = renderHeatmapControls(sweptVars, selectedVarNames);

    if (dim === 1) {
        return renderHeatmap1D(heatmapData, controls);
    } else {
        return renderHeatmap2D(heatmapData, controls);
    }
}

function renderHeatmap1D(heatmapData, controls) {
    const { varX, xValues, values, cellData, minLoss, maxLoss } = heatmapData;

    let bestX = null, bestLoss = Infinity;
    xValues.forEach(x => {
        const loss = values[x];
        if (loss != null && loss < bestLoss) {
            bestLoss = loss;
            bestX = x;
        }
    });

    const compact = xValues.length > 6 ? ' heatmap-compact' : '';
    const selectedCellExpIds = State.get('selectedCellExpIds');
    const selectedRunId = State.get('selectedRun');

    return `
        <div class="heatmap-panel${compact}">
            <div class="heatmap-content">
                ${controls}
                <div class="heatmap-1d">
                    <div class="heatmap-1d-label">${varX}</div>
                    <div class="heatmap-1d-cells">
                        ${xValues.map(x => {
                            const val = values[x];
                            const color = getHeatmapColor(val, minLoss, maxLoss);
                            const isBest = (x === bestX);
                            const cell = cellData[x];
                            const count = cell?.count || 0;
                            const allExpIds = cell?.allExps?.map(e => e.id) || [];
                            const isSelected = selectedCellExpIds
                                ? allExpIds.some(id => selectedCellExpIds.includes(id))
                                : allExpIds.includes(selectedRunId);
                            const countLabel = count > 1 ? ` (n=${count})` : '';
                            const title = val != null
                                ? `${varX}=${x}\nMean Val Loss: ${val.toFixed(4)}${countLabel}`
                                : `${varX}=${x}\nNo data`;
                            const countDisplay = count > 1 ? `<span class="cell-count">${count}</span>` : '';
                            return `
                                <div class="heatmap-1d-cell-wrapper">
                                    <div class="heatmap-cell${isBest ? ' best' : ''}${isSelected ? ' selected' : ''}"
                                        style="background-color: ${color}"
                                        title="${title}"
                                        data-exp-ids="${allExpIds.join(',')}"
                                        ${allExpIds.length > 0 ? `onclick="selectHeatmapCell([${allExpIds.join(',')}])"` : ''}>
                                        ${val != null ? val.toFixed(4) : '-'}${countDisplay}
                                    </div>
                                    <div class="heatmap-1d-col-label">${x}</div>
                                </div>`;
                        }).join('')}
                    </div>
                    <div class="heatmap-legend">
                        <span class="legend-label">${maxLoss.toFixed(4)}</span>
                        <div class="legend-gradient"></div>
                        <span class="legend-label">${minLoss.toFixed(4)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderHeatmap2D(heatmapData, controls) {
    const { varX, varY, xValues, yValues, matrix, cellData, minLoss, maxLoss } = heatmapData;

    let bestX = null, bestY = null, bestLoss = Infinity;
    yValues.forEach(y => {
        xValues.forEach(x => {
            const loss = matrix[y]?.[x];
            if (loss != null && loss < bestLoss) {
                bestLoss = loss;
                bestX = x;
                bestY = y;
            }
        });
    });

    const numCols = xValues.length;
    const compact = numCols > 6 ? ' heatmap-compact' : '';
    const yValuesReversed = [...yValues].reverse();
    const selectedCellExpIds = State.get('selectedCellExpIds');
    const selectedRunId = State.get('selectedRun');

    return `
        <div class="heatmap-panel${compact}">
            <div class="heatmap-content">
                ${controls}
                <div class="heatmap-container">
                    <div class="heatmap-main">
                        <div class="heatmap-grid-row">
                            <div class="heatmap-y-label">${varY}</div>
                            <div class="heatmap-grid" style="grid-template-columns: auto repeat(${numCols}, minmax(0, 1fr));">
                                ${yValuesReversed.map(y => `
                                    <div class="heatmap-row-label">${y}</div>
                                    ${xValues.map(x => {
                                        const val = matrix[y]?.[x];
                                        const color = getHeatmapColor(val, minLoss, maxLoss);
                                        const isBest = (x === bestX && y === bestY);
                                        const cell = cellData[y]?.[x];
                                        const count = cell?.count || 0;
                                        const allExpIds = cell?.allExps?.map(e => e.id) || [];
                                        const isSelected = selectedCellExpIds
                                            ? allExpIds.some(id => selectedCellExpIds.includes(id))
                                            : allExpIds.includes(selectedRunId);
                                        const countLabel = count > 1 ? ` (n=${count})` : '';
                                        const title = val != null
                                            ? `${varX}=${x}, ${varY}=${y}\nMean Val Loss: ${val.toFixed(4)}${countLabel}`
                                            : `${varX}=${x}, ${varY}=${y}\nNo data`;
                                        const countDisplay = count > 1 ? `<span class="cell-count">${count}</span>` : '';
                                        return `<div class="heatmap-cell${isBest ? ' best' : ''}${isSelected ? ' selected' : ''}"
                                            style="background-color: ${color}"
                                            title="${title}"
                                            data-exp-ids="${allExpIds.join(',')}"
                                            ${allExpIds.length > 0 ? `onclick="selectHeatmapCell([${allExpIds.join(',')}])"` : ''}>
                                            ${val != null ? val.toFixed(4) : '-'}${countDisplay}
                                        </div>`;
                                    }).join('')}
                                `).join('')}
                                <div class="heatmap-row-label"></div>
                                ${xValues.map(x => `<div class="heatmap-col-label">${x}</div>`).join('')}
                            </div>
                            <div class="heatmap-legend">
                                <span class="legend-label">${maxLoss.toFixed(4)}</span>
                                <div class="legend-gradient"></div>
                                <span class="legend-label">${minLoss.toFixed(4)}</span>
                            </div>
                        </div>
                        <div class="heatmap-x-label">${varX}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function selectHeatmapCell(expIds) {
    if (!expIds || expIds.length === 0) return;

    if (expIds.length === 1) {
        showMetricsForRun(expIds[0]);
    } else {
        const selectedCellExpIds = State.get('selectedCellExpIds');
        const sameCell = selectedCellExpIds &&
            selectedCellExpIds.length === expIds.length &&
            expIds.every(id => selectedCellExpIds.includes(id));

        if (sameCell) {
            State.set('selectedCellExpIds', null);
            State.set('selectedRun', null);
            showAveragedMetrics();
        } else {
            State.set('selectedCellExpIds', expIds);
            State.set('selectedRun', null);
            showAveragedMetricsForCell(expIds);
        }
        updateRunRowHighlights();
        updateHeatmapHighlights();
    }
}

function computeResidualData(validData) {
    const completedData = validData.filter(d => d.status === 'completed');
    if (completedData.length < 2) return null;

    const curves = completedData.map(d => ({
        id: d.id,
        name: d.name,
        script: d.script,
        env_vars: d.env_vars,
        data: d.loss_curve || []
    })).filter(c => c.data.length > 0);

    if (curves.length < 2) return null;

    const lossesByStep = {};
    curves.forEach(curve => {
        curve.data.forEach(point => {
            if (!lossesByStep[point.step]) lossesByStep[point.step] = [];
            lossesByStep[point.step].push({
                curveId: curve.id,
                loss: point.val_loss
            });
        });
    });

    const validSteps = Object.keys(lossesByStep)
        .map(s => parseInt(s))
        .filter(step => lossesByStep[step].length >= 2)
        .sort((a, b) => a - b);

    if (validSteps.length === 0) return null;

    const medianByStep = {};
    validSteps.forEach(step => {
        const losses = lossesByStep[step].map(d => d.loss).filter(l => l != null).sort((a, b) => a - b);
        if (losses.length > 0) {
            const mid = Math.floor(losses.length / 2);
            medianByStep[step] = losses.length % 2 === 0
                ? (losses[mid - 1] + losses[mid]) / 2
                : losses[mid];
        }
    });

    const residualCurves = curves.map(curve => {
        const residuals = curve.data
            .filter(point => medianByStep[point.step] !== undefined && point.val_loss != null)
            .map(point => ({
                step: point.step,
                residual: point.val_loss - medianByStep[point.step]
            }));
        return {
            id: curve.id,
            name: curve.name,
            script: curve.script,
            env_vars: curve.env_vars,
            residuals
        };
    }).filter(c => c.residuals.length > 0);

    return residualCurves;
}

function getEligibleViews(validData) {
    const views = [];
    const completedData = validData.filter(d => d.status === 'completed');
    const runningOrCompleted = validData.filter(d =>
        d.status === 'running' || d.status === 'completed'
    );

    // Heatmap: disabled in bucket view
    if (State.get('selectedExp') !== 'bucket') {
        const { processed } = getHeatmapData(validData);
        if (computeHeatmapData(processed)) views.push('heatmap');
    }

    // Residual: >=2 completed
    if (completedData.length >= 2) views.push('residual');

    // Line: >=1 running/completed
    if (runningOrCompleted.length >= 1) views.push('line');

    return views;
}

function detectPartialVars(validData) {
    const completedData = validData.filter(d => d.status === 'completed');
    if (completedData.length < 2) return [];
    const keyCounts = {};
    completedData.forEach(exp => {
        Object.keys(exp.env_vars || {}).forEach(key => {
            keyCounts[key] = (keyCounts[key] || 0) + 1;
        });
    });
    return Object.entries(keyCounts)
        .filter(([key, count]) => count > 0 && count < completedData.length)
        .map(([key]) => key);
}

function applyHeatmapDefaults(validData, defaults, partialVars) {
    if (!partialVars || partialVars.length === 0) return validData;
    return validData.map(d => {
        const envVars = { ...(d.env_vars || {}) };
        let changed = false;
        for (const key of partialVars) {
            if (envVars[key] === undefined) {
                envVars[key] = (defaults[key] !== undefined && defaults[key] !== '')
                    ? defaults[key]
                    : '(default)';
                changed = true;
            }
        }
        return changed ? { ...d, env_vars: envVars } : d;
    });
}

function getHeatmapData(validData) {
    const partialVars = detectPartialVars(validData);
    const codeHash = validData.length > 0 ? validData[0].code_hash : null;
    const defaults = codeHash ? (State.get('heatmapDefaults', codeHash) || {}) : {};
    const processed = applyHeatmapDefaults(validData, defaults, partialVars);
    return { processed, partialVars };
}

function updateViewSwitcher(eligibleViews) {
    const switcher = document.getElementById('chart-view-switcher');
    const buttons = switcher.querySelectorAll('.chart-view-btn');

    buttons.forEach(btn => {
        const view = btn.dataset.view;
        if (eligibleViews.includes(view)) {
            btn.classList.remove('hidden');
        } else {
            btn.classList.add('hidden');
        }
    });

    if (eligibleViews.length <= 1) {
        switcher.style.display = 'none';
    } else {
        switcher.style.display = 'flex';
    }
}

function switchChartView(viewName, updateState = true) {
    const currentValidData = State.get('experimentData');
    if (!currentValidData) return;

    if (updateState) State.set('chartView', viewName);

    // Update button highlights
    const buttons = document.querySelectorAll('.chart-view-btn');
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    const chartContainer = document.querySelector('.chart-container');
    const canvas = document.getElementById('loss-chart');
    const heatmapContainer = document.getElementById('chart-heatmap-container');

    chartContainer.classList.toggle('heatmap-view', viewName === 'heatmap');
    chartContainer.classList.toggle('residual-view', viewName === 'residual');

    const legendEl = document.getElementById('chart-legend');
    const rangeEl = document.getElementById('chart-range');
    if (viewName !== 'line' && viewName !== 'residual') {
        if (legendEl) legendEl.innerHTML = '';
    }
    if (viewName !== 'line' && viewName !== 'residual') {
        if (rangeEl) rangeEl.innerHTML = '';
    }

    switch (viewName) {
        case 'heatmap':
            canvas.style.display = 'none';
            heatmapContainer.style.display = 'block';
            const { processed: heatmapProcessed } = getHeatmapData(getVisibleRuns());
            const initialData = computeHeatmapData(heatmapProcessed, null);
            if (initialData && initialData.sweptVars) {
                const validVarNames = initialData.sweptVars.map(v => v.key);
                const heatmapSelectedVars = State.get('heatmapSelectedVars');
                const validSelected = heatmapSelectedVars.filter(v => validVarNames.includes(v));
                if (validSelected.length === 0) {
                    State.set('heatmapSelectedVars', validVarNames.slice(0, 2));
                } else {
                    State.set('heatmapSelectedVars', validSelected);
                }
            }
            const heatmapData = computeHeatmapData(heatmapProcessed, State.get('heatmapSelectedVars'));
            heatmapContainer.innerHTML = renderHeatmap(heatmapData);
            break;
        case 'residual':
            canvas.style.display = 'block';
            heatmapContainer.style.display = 'none';
            renderResidualChart(computeResidualData(getVisibleRuns()));
            break;
        case 'line':
            canvas.style.display = 'block';
            heatmapContainer.style.display = 'none';
            const runs = getVisibleRuns().map(d => ({
                name: d.name,
                script: d.script,
                data: d.loss_curve,
                env_vars: d.env_vars
            }));
            updateChartMultiple(runs, State.get('currentTotalSteps'));
            break;
    }
}

function renderResidualChart(residualData) {
    const ctx = document.getElementById('loss-chart').getContext('2d');

    if (lossChart) {
        lossChart.destroy();
    }

    if (!residualData || residualData.length === 0) {
        return;
    }

    const multiRun = residualData.length > 1;

    const diffKeys = getDifferingKeys(residualData);
    if (multiRun && diffKeys.length > 0) {
        residualData.sort((a, b) => {
            for (const k of diffKeys) {
                const va = (a.env_vars || {})[k] ?? '';
                const vb = (b.env_vars || {})[k] ?? '';
                const na = parseFloat(va), nb = parseFloat(vb);
                if (!isNaN(na) && !isNaN(nb)) {
                    if (na !== nb) return na - nb;
                } else {
                    if (va < vb) return -1;
                    if (va > vb) return 1;
                }
            }
            return 0;
        });
    }

    renderChartLegend(residualData);

    const labels = computeSmartLabels(residualData);

    const allSteps = residualData.flatMap(c => c.residuals.map(r => r.step));
    const dataXMin = Math.min(...allSteps);
    const maxStep = Math.max(...allSteps);

    let chartXRange = State.get('chartXRange');
    if (!chartXRange) {
        chartXRange = { min: dataXMin, max: maxStep };
        State.set('chartXRange', chartXRange);
    }

    const rangeEl = document.getElementById('chart-range');
    if (rangeEl && multiRun) {
        rangeEl.innerHTML = `<div class="chart-range-controls">
            <span class="chart-range-text">Step
                <span class="runs-limit-editable" onclick="makeChartRangeEditable(this, 'min')" title="Click to edit">${chartXRange.min}</span>
                –
                <span class="runs-limit-editable" onclick="makeChartRangeEditable(this, 'max')" title="Click to edit">${chartXRange.max}</span>
            </span>
        </div>`;
    }

    const datasets = residualData.map((curve, i) => {
        const color = CHART_COLORS[i % CHART_COLORS.length];

        return {
            label: multiRun ? labels[i] : 'Residual',
            data: curve.residuals.map(d => ({ x: d.step, y: d.residual })),
            borderColor: color,
            backgroundColor: color,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.1
        };
    });

    datasets.push({
        label: 'Median',
        data: [{ x: 0, y: 0 }, { x: maxStep, y: 0 }],
        borderColor: _themeColor('--text-placeholder'),
        borderWidth: 1,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false
    });

    const allResiduals = residualData.flatMap(c => c.residuals.map(r => r.residual));
    const minResidual = Math.min(...allResiduals);
    const maxResidual = Math.max(...allResiduals);
    const absMax = Math.max(Math.abs(minResidual), Math.abs(maxResidual));
    const yRange = Math.max(absMax * 1.05, 0.02);

    const stepInterval = detectStepInterval(residualData.map(c => ({ data: c.residuals.map(r => ({ step: r.step })) })));

    lossChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: false },
                    grid: { color: _themeColor('--border-subtle') },
                    ticks: {
                        color: _themeColor('--text-tertiary'),
                        stepSize: stepInterval,
                        callback: (v) => Number.isInteger(v) ? v.toLocaleString() : ''
                    },
                    min: chartXRange.min,
                    max: chartXRange.max
                },
                y: {
                    title: { display: true, text: 'Residual (Loss - Median)', color: _themeColor('--text-tertiary') },
                    grid: { color: _themeColor('--border-subtle') },
                    ticks: { color: _themeColor('--text-tertiary') },
                    min: -yRange,
                    max: yRange
                }
            },
            plugins: {
                legend: {
                    display: false,
                    labels: {
                        color: '#aaa',
                        usePointStyle: true,
                        pointStyle: 'line',
                        filter: (item) => item.text !== 'Median'
                    },
                    onHover: (e) => { e.native.target.style.cursor = 'pointer'; },
                    onLeave: (e) => { e.native.target.style.cursor = 'default'; }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            if (ctx.dataset.label === 'Median') return null;
                            return `${ctx.dataset.label}: ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(4)}`;
                        }
                    }
                }
            }
        }
    });
}

function renderChartLegend(runs) {
    const legendEl = document.getElementById('chart-legend');
    if (!legendEl || runs.length <= 1) {
        if (legendEl) legendEl.innerHTML = '';
        return;
    }
    const diffKeys = getDifferingKeys(runs);
    const isBucketView = State.get('selectedExp') === 'bucket';
    const showScript = isBucketView || new Set(runs.map(r => r.script || r.name)).size > 1;
    let headerCols = showScript ? '<th>Script</th>' : '';
    headerCols += diffKeys.length > 0
        ? diffKeys.map(k => `<th>${k}</th>`).join('')
        : (!showScript ? '<th>Run</th>' : '');
    const rows = runs.map((run, i) => {
        const color = CHART_COLORS[i % CHART_COLORS.length];
        const env = run.env_vars || {};
        const scriptCell = showScript ? `<td>${run.script ? run.script.split('/').pop().replace('.py', '') : run.name}</td>` : '';
        const envCells = diffKeys.length > 0
            ? diffKeys.map(k => `<td>${env[k] ?? ''}</td>`).join('')
            : (!showScript ? `<td>${run.script ? run.script.split('/').pop().replace('.py', '') : run.name}</td>` : '');
        return `<tr class="chart-legend-row" data-idx="${i}" onclick="toggleChartDataset(${i})" onmouseenter="highlightChartDataset(${i})" onmouseleave="unhighlightChartDataset()">
            <td class="chart-legend-swatch-cell"><span class="chart-legend-swatch" style="background:${color}"></span></td>
            ${scriptCell}${envCells}
        </tr>`;
    }).join('');
    legendEl.innerHTML = `<table class="chart-legend-table">
        <thead><tr><th></th>${headerCols}</tr></thead>
        <tbody>${rows}</tbody>
    </table>`;
}

function updateChartMultiple(runs, totalSteps = 0) {
    const ctx = document.getElementById('loss-chart').getContext('2d');

    if (lossChart) {
        lossChart.destroy();
    }

    const legendEl = document.getElementById('chart-legend');
    const rangeEl = document.getElementById('chart-range');

    const validRuns = runs.filter(r => r.data && r.data.length > 0);
    if (validRuns.length === 0) {
        legendEl.innerHTML = '';
        rangeEl.innerHTML = '';
        return;
    }

    const isMulti = validRuns.length > 1;

    const diffKeys = getDifferingKeys(validRuns);
    if (isMulti && diffKeys.length > 0) {
        validRuns.sort((a, b) => {
            for (const k of diffKeys) {
                const va = (a.env_vars || {})[k] ?? '';
                const vb = (b.env_vars || {})[k] ?? '';
                const na = parseFloat(va), nb = parseFloat(vb);
                if (!isNaN(na) && !isNaN(nb)) {
                    if (na !== nb) return na - nb;
                } else {
                    if (va < vb) return -1;
                    if (va > vb) return 1;
                }
            }
            return 0;
        });
    }

    const labels = computeSmartLabels(validRuns);

    const datasets = validRuns.map((run, i) => {
        const color = CHART_COLORS[i % CHART_COLORS.length];
        return {
            label: labels[i],
            data: run.data.map(d => ({ x: d.step, y: d.val_loss })),
            borderColor: color,
            backgroundColor: color + '20',
            fill: !isMulti,
            tension: 0.1,
            pointRadius: isMulti ? 0 : 3,
            borderWidth: isMulti ? 1.5 : 2,
        };
    });

    const allSteps = validRuns.flatMap(r => r.data.map(d => d.step));
    const dataXMin = Math.min(...allSteps);
    const dataXMax = Math.max(...allSteps);
    const xMax = totalSteps > 0 ? Math.max(totalSteps, dataXMax) : dataXMax;

    let chartXRange = State.get('chartXRange');
    if (!chartXRange) {
        chartXRange = { min: dataXMin, max: xMax };
        State.set('chartXRange', chartXRange);
    }

    const stepInterval = detectStepInterval(validRuns);

    renderChartLegend(validRuns);

    rangeEl.innerHTML = `
        <div class="chart-range-controls">
            <span class="chart-range-text">Step
                <span class="runs-limit-editable" onclick="makeChartRangeEditable(this, 'min')" title="Click to edit">${chartXRange.min}</span>
                –
                <span class="runs-limit-editable" onclick="makeChartRangeEditable(this, 'max')" title="Click to edit">${chartXRange.max}</span>
            </span>
        </div>
    `;

    lossChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                mode: 'x',
                intersect: false
            },
            hover: {
                mode: 'x',
                intersect: false
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: false },
                    min: chartXRange.min,
                    max: chartXRange.max,
                    ticks: { stepSize: stepInterval, callback: (v) => Number.isInteger(v) ? v.toLocaleString() : '' },
                    grid: { color: 'rgba(255, 255, 255, 0.15)', lineWidth: 1 }
                },
                y: {
                    title: { display: false },
                    grid: { color: 'rgba(255, 255, 255, 0.15)', lineWidth: 1 }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'x',
                    intersect: false,
                    callbacks: {
                        title: function(items) {
                            return items.length > 0 ? 'Step ' + items[0].parsed.x : '';
                        },
                        label: function(context) {
                            const value = context.parsed.y;
                            return value !== null ? context.dataset.label + ': ' + value.toFixed(4) : null;
                        },
                        labelColor: function(context) {
                            return { borderColor: context.dataset.borderColor, backgroundColor: context.dataset.borderColor };
                        }
                    }
                }
            }
        }
    });
}

function updateChart(lossData, totalSteps) {
    updateChartMultiple([{ name: 'Run', data: lossData, env_vars: {} }], totalSteps);
}

function toggleChartDataset(idx) {
    if (!lossChart) return;
    const meta = lossChart.getDatasetMeta(idx);
    meta.hidden = !meta.hidden;
    lossChart.update();
    const rows = document.querySelectorAll('.chart-legend-row');
    if (rows[idx]) rows[idx].classList.toggle('hidden', meta.hidden);
}

function highlightChartDataset(idx) {
    if (!lossChart) return;
    if (_chartOriginalColors.length === 0) {
        _chartOriginalColors = lossChart.data.datasets.map(ds => ds.borderColor);
    }
    lossChart.data.datasets.forEach((ds, i) => {
        if (i === idx) {
            ds.borderWidth = 3;
            ds.borderColor = _chartOriginalColors[i];
        } else {
            ds.borderWidth = 0.5;
            ds.borderColor = _chartOriginalColors[i] + '33';
        }
    });
    lossChart.update('none');
}

function unhighlightChartDataset() {
    if (!lossChart) return;
    const isMulti = lossChart.data.datasets.length > 1;
    lossChart.data.datasets.forEach((ds, i) => {
        ds.borderWidth = isMulti ? 1.5 : 2;
        if (_chartOriginalColors[i]) ds.borderColor = _chartOriginalColors[i];
    });
    _chartOriginalColors = [];
    lossChart.update('none');
}

function makeChartRangeEditable(el, which) {
    const chartXRange = State.get('chartXRange');
    const current = chartXRange ? chartXRange[which] : 0;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    input.className = 'runs-limit-inline';
    input.size = String(current).length || 4;

    const commit = () => {
        const n = parseInt(input.value);
        const range = State.get('chartXRange');
        if (!isNaN(n) && n >= 0 && range) {
            range[which] = n;
            if (range.min > range.max) {
                const tmp = range.min;
                range.min = range.max;
                range.max = tmp;
            }
            State.set('chartXRange', range);
            if (lossChart) {
                lossChart.options.scales.x.min = range.min;
                lossChart.options.scales.x.max = range.max;
                lossChart.update('none');
            }
        }
        const chartView = State.get('chartView');
        if (chartView) switchChartView(chartView, false);
    };

    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); commit(); }
        if (e.key === 'Escape') { e.preventDefault(); const cv = State.get('chartView'); if (cv) switchChartView(cv, false); }
    });
    input.addEventListener('blur', commit);

    el.replaceWith(input);
    input.focus();
    input.select();
}

function toggleChartFullscreen() {
    const chartBlock = document.querySelector('.meta-chart-block');
    chartBlock.classList.toggle('fullscreen');
    if (lossChart) {
        setTimeout(() => lossChart.resize(), 50);
    }
}

async function copyChartAsImage() {
    const btn = document.querySelector('.chart-copy-btn');
    const titleInput = document.getElementById('chart-copy-title');
    const title = titleInput ? titleInput.value.trim() : '';

    let blob;
    if (State.get('chartView') === 'heatmap') {
        blob = await copyHeatmapAsImage(title);
    } else {
        blob = await copyLineChartAsImage(title);
    }

    if (blob) {
        try {
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
            if (btn) {
                btn.classList.add('copied');
                setTimeout(() => btn.classList.remove('copied'), 400);
            }
        } catch (e) {
            console.error('Failed to copy:', e);
        }
    }
}

async function copyLineChartAsImage(title) {
    if (!lossChart) return null;
    const chartCanvas = lossChart.canvas;
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    const hasTitle = title.length > 0;

    const pad = { top: hasTitle ? 44 : 10, right: 20, bottom: 44, left: 28 };
    const offscreen = document.createElement('canvas');
    offscreen.width = w + pad.left + pad.right;
    offscreen.height = h + pad.top + pad.bottom;
    const ctx = offscreen.getContext('2d');

    ctx.fillStyle = _themeColor('--bg-dark');
    ctx.fillRect(0, 0, offscreen.width, offscreen.height);

    if (hasTitle) {
        ctx.fillStyle = _themeColor('--text-primary');
        ctx.font = 'bold 24px SF Mono, Menlo, Monaco, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(title, offscreen.width / 2, 30);
    }

    ctx.drawImage(chartCanvas, pad.left, pad.top);

    ctx.fillStyle = _themeColor('--text-secondary');
    ctx.font = '18px SF Mono, Menlo, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Step', pad.left + w / 2, pad.top + h + 32);

    ctx.save();
    ctx.font = '18px SF Mono, Menlo, Monaco, monospace';
    ctx.translate(16, pad.top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Val Loss', 0, 0);
    ctx.restore();

    return new Promise(resolve => offscreen.toBlob(resolve, 'image/png'));
}

async function copyHeatmapAsImage(title) {
    const { processed } = getHeatmapData(getVisibleRuns());
    const data = computeHeatmapData(processed, State.get('heatmapSelectedVars'));
    if (!data) return null;

    const font = '16px SF Mono, Menlo, Monaco, monospace';
    const titleFont = 'bold 22px SF Mono, Menlo, Monaco, monospace';
    const labelFont = '16px SF Mono, Menlo, Monaco, monospace';
    const cellW = 100;
    const cellH = 50;
    const cellPad = 1;
    const hasTitle = title.length > 0;
    const pad = { top: hasTitle ? 60 : 20, right: 30, bottom: 50, left: 110 };

    if (data.dim === 1) {
        const nCols = data.xValues.length;
        const w = pad.left + nCols * (cellW + cellPad) + pad.right;
        const h = pad.top + cellH + pad.bottom;
        const offscreen = document.createElement('canvas');
        offscreen.width = w;
        offscreen.height = h;
        const ctx = offscreen.getContext('2d');

        ctx.fillStyle = _themeColor('--bg-dark');
        ctx.fillRect(0, 0, w, h);

        if (hasTitle) {
            ctx.fillStyle = _themeColor('--text-primary');
            ctx.font = titleFont;
            ctx.textAlign = 'center';
            ctx.fillText(title, w / 2, 34);
        }

        ctx.fillStyle = _themeColor('--text-secondary');
        ctx.font = labelFont;
        ctx.textAlign = 'right';
        ctx.fillText(data.varX, pad.left - 10, pad.top + cellH / 2 + 6);

        data.xValues.forEach((xVal, xi) => {
            const x = pad.left + xi * (cellW + cellPad);
            const y = pad.top;
            const val = data.values[xVal];
            ctx.fillStyle = getHeatmapColor(val, data.minLoss, data.maxLoss);
            ctx.fillRect(x, y, cellW, cellH);
            ctx.fillStyle = _themeColor('--text-primary');
            ctx.font = font;
            ctx.textAlign = 'center';
            ctx.fillText(val != null ? val.toFixed(4) : '–', x + cellW / 2, y + cellH / 2 + 6);
            ctx.fillStyle = _themeColor('--text-secondary');
            ctx.fillText(xVal, x + cellW / 2, y + cellH + 20);
        });

        return new Promise(resolve => offscreen.toBlob(resolve, 'image/png'));
    }

    // 2D case
    const nCols = data.xValues.length;
    const nRows = data.yValues.length;
    const gridW = nCols * (cellW + cellPad);
    const gridH = nRows * (cellH + cellPad);
    const w = pad.left + gridW + pad.right;
    const h = pad.top + gridH + pad.bottom;
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d');

    ctx.fillStyle = _themeColor('--bg-dark');
    ctx.fillRect(0, 0, w, h);

    if (hasTitle) {
        ctx.fillStyle = _themeColor('--text-primary');
        ctx.font = titleFont;
        ctx.textAlign = 'center';
        ctx.fillText(title, w / 2, 34);
    }

    // X-axis label
    ctx.fillStyle = _themeColor('--text-secondary');
    ctx.font = labelFont;
    ctx.textAlign = 'center';
    ctx.fillText(data.varX, pad.left + gridW / 2, h - 10);

    // Y-axis label
    ctx.save();
    ctx.font = labelFont;
    ctx.translate(16, pad.top + gridH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(data.varY, 0, 0);
    ctx.restore();

    // Draw cells
    data.yValues.forEach((yVal, yi) => {
        const y = pad.top + yi * (cellH + cellPad);
        ctx.fillStyle = _themeColor('--text-secondary');
        ctx.font = font;
        ctx.textAlign = 'right';
        ctx.fillText(yVal, pad.left - 8, y + cellH / 2 + 6);

        data.xValues.forEach((xVal, xi) => {
            const x = pad.left + xi * (cellW + cellPad);
            const val = data.matrix[yVal]?.[xVal];
            ctx.fillStyle = getHeatmapColor(val, data.minLoss, data.maxLoss);
            ctx.fillRect(x, y, cellW, cellH);
            ctx.fillStyle = _themeColor('--text-primary');
            ctx.font = font;
            ctx.textAlign = 'center';
            ctx.fillText(val != null ? val.toFixed(4) : '–', x + cellW / 2, y + cellH / 2 + 6);
        });
    });

    // X labels at bottom
    data.xValues.forEach((xVal, xi) => {
        const x = pad.left + xi * (cellW + cellPad);
        ctx.fillStyle = _themeColor('--text-secondary');
        ctx.font = font;
        ctx.textAlign = 'center';
        ctx.fillText(xVal, x + cellW / 2, pad.top + gridH + 20);
    });

    return new Promise(resolve => offscreen.toBlob(resolve, 'image/png'));
}
