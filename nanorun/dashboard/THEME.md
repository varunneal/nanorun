# Dashboard Theme

Design tokens for the nanorun dashboard. All colors and typography should reference these values.

## Fonts

- **Brand** — `'Space Grotesk', sans-serif` (weights: 500, 700). Used for: page title, chip labels.
- **Mono** — `'SF Mono', 'Menlo', 'Monaco', monospace`. Used for: body text, tables, code, popovers, everything else.

## Font Sizes

- `--font-xs` — `0.65rem` — badges, tags, compact heatmap cells, env chips, queue item metadata
- `--font-sm` — `0.72rem` — chips, buttons, progress text, chart legends, view switcher tabs
- `--font-base` — `0.8rem` — body text, table data, filter inputs, status badges, metric values, popovers
- `--font-lg` — `1rem` — section headers (panel titles, notes h2)
- `--font-xl` — `1.1rem` — page title

## Colors

### Backgrounds

- `--bg-dark` — `#0f1218` — page background
- `--bg-card` — `#161a24` — card/panel/popover background
- `--bg-hover` — `#1e2433` — hover state on cards/rows
- `--bg-selected` — `#0a0d12` — selected card background

### Text

- `--text-primary` — `#eef` — primary text
- `--text-secondary` — `#aaa` — secondary/muted text
- `--text-tertiary` — `#888` — axis labels, GPU type, least-important text
- `--text-placeholder` — `#666` — input placeholders, missing/disabled

### Borders

- `--border-subtle` — `#333` — default separators, dividers, scrollbar
- `--border` — `#444` — card borders, popover borders, table headers
- `--border-hover` — `#555` — interactive element borders on hover
- `--border-active` — `#777` — focused/pressed borders

### Semantic

- `--accent` — `#6aab6e` — primary action, success, good metric, connected status, healthy state
- `--accent-dim` — `#4a7a4e` — dimmed accent (flash states)
- `--warning` — `#e0b868` — running state, paused queue, connecting/reconnecting status
- `--error` — `#e07570` — failed experiments, disconnected status, delete actions
- `--info` — `#3498db` — informational highlights (experiment IDs, queue counts)

### Experiment UI

- `--track-color` — `#8fb4d8` — track name text
- `--hash-color` — `#9aa0a6` — code hash text
- `--env-color` — `#8ab4f8` — environment variable chips
- `--run-id-color` — `#f0b27a` — run ID links
- `--sweep-color` — `#b388ff` — sweep badge text
- `--session-color` — `#b8a0e8` — session label text

### Chart

- Series palette: `#5a9a5e`, `#5a8ab8`, `#c9a054`, `#b05a7a`, `#7a5a9a`, `#5a9a9a`, `#c9b854`, `#8a7058`
- Heatmap gradient: `rgb(90, 154, 94)` (good) → `rgb(157, 136, 103)` (mid) → `rgb(224, 117, 112)` (bad)
- Grid lines: `rgba(255, 255, 255, 0.15)`
