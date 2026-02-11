/**
 * BitNet-Mamba 204M - Training Dashboard Frontend
 * ================================================
 * Pure JS application using Chart.js for visualization.
 * Polls FastAPI backend for training metrics, decisions, and checkpoints.
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------------
    // Config
    // -----------------------------------------------------------------------
    const API_BASE = "";  // same origin
    const MAX_STEPS = 61035;
    const MAX_TOKENS = 8e9;

    let statusInterval = null;
    let metricsInterval = null;
    let checkpointsInterval = null;
    let hardwareInterval = null;
    let refreshRate = 10;  // seconds for status
    let lastUpdateTs = Date.now();
    let xAxisMode = "step";  // "step" or "time"

    // Chart instances
    let chartLoss = null;
    let chartLR = null;
    let chartGrad = null;
    let chartTokens = null;

    // Cached data
    let cachedMetrics = null;
    let cachedDecisions = null;
    let cachedGradNorms = null;
    let currentPolicyFilter = "all";

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------
    function formatNumber(n) {
        if (n === null || n === undefined) return "--";
        if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
        if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
        if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "K";
        return n.toLocaleString();
    }

    function formatSci(n) {
        if (n === null || n === undefined) return "--";
        if (n === 0) return "0";
        return n.toExponential(2);
    }

    function formatLoss(n) {
        if (n === null || n === undefined) return "--";
        return n.toFixed(4);
    }

    function escapeHtml(s) {
        const div = document.createElement("div");
        div.textContent = s;
        return div.innerHTML;
    }

    function formatTimestamp(ts) {
        if (!ts) return "--";
        try {
            const d = new Date(ts);
            return d.toLocaleString();
        } catch (e) {
            return ts;
        }
    }

    function formatTimestampShort(ts) {
        if (!ts) return "--";
        try {
            const d = new Date(ts);
            return d.toLocaleTimeString();
        } catch (e) {
            return ts;
        }
    }

    // -----------------------------------------------------------------------
    // API Fetch wrapper
    // -----------------------------------------------------------------------
    async function apiFetch(endpoint) {
        try {
            const resp = await fetch(API_BASE + endpoint, {
                cache: "no-store",
                headers: {
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                },
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            setConnected(true);
            return await resp.json();
        } catch (err) {
            console.error(`[Dashboard] API error ${endpoint}:`, err);
            setConnected(false);
            return null;
        }
    }

    function setConnected(ok) {
        const el = document.getElementById("connection-status");
        const banner = document.getElementById("disconnect-banner");
        if (ok) {
            el.className = "conn-status conn-ok";
            el.querySelector(".conn-text").textContent = "Connected";
            banner.style.display = "none";
            lastUpdateTs = Date.now();
        } else {
            el.className = "conn-status conn-err";
            el.querySelector(".conn-text").textContent = "Disconnected";
            banner.style.display = "block";
        }
    }

    // -----------------------------------------------------------------------
    // Status panel
    // -----------------------------------------------------------------------
    async function fetchStatus() {
        const data = await apiFetch("/api/status");
        if (!data) return;

        document.getElementById("stat-step").textContent = formatNumber(data.step);
        document.getElementById("stat-max-steps").textContent = formatNumber(data.max_steps);
        document.getElementById("stat-tokens").textContent = formatNumber(data.tokens);
        document.getElementById("stat-max-tokens").textContent = formatNumber(data.max_tokens);
        document.getElementById("stat-tps").textContent = formatNumber(data.tokens_per_sec);
        document.getElementById("stat-elapsed").textContent = data.elapsed_time || "--";
        document.getElementById("stat-progress").textContent = data.progress_pct != null ? data.progress_pct.toFixed(1) : "--";
        document.getElementById("stat-lr").textContent = formatSci(data.lr_current);
        document.getElementById("stat-lr-bit").textContent = formatSci(data.lr_bitlinear);
        document.getElementById("stat-val-loss").textContent = formatLoss(data.last_val_loss);
        document.getElementById("stat-prev-val-loss").textContent = formatLoss(data.previous_val_loss);
        document.getElementById("stat-last-step-time").textContent = formatTimestamp(data.last_step_time);
        document.getElementById("stat-last-tps").textContent = formatNumber(data.last_tokens_per_sec);
        document.getElementById("stat-last-checkpoint").textContent = data.latest_checkpoint_name || "--";
        document.getElementById("stat-best-val").textContent = formatLoss(data.best_val_loss);

        // Progress bar
        const pbar = document.getElementById("stat-progress-bar");
        pbar.style.width = Math.min(data.progress_pct || 0, 100) + "%";

        // State badge
        const stateEl = document.getElementById("stat-state");
        stateEl.textContent = data.state;
        stateEl.className = "badge badge-state state-" + (data.state || "unknown");

        // Regime badge
        const regimeEl = document.getElementById("stat-regime");
        regimeEl.textContent = data.regime;
        regimeEl.className = "badge badge-regime regime-" + (data.regime || "UNKNOWN");
    }

    // -----------------------------------------------------------------------
    // Hardware monitor
    // -----------------------------------------------------------------------
    function tempBarClass(tempC) {
        if (tempC === null || tempC === undefined) return "";
        if (tempC >= 80) return "hw-crit";
        if (tempC >= 65) return "hw-warn";
        return "";
    }

    function setBar(barId, pct, extraClass) {
        const bar = document.getElementById(barId);
        if (!bar) return;
        bar.style.width = Math.min(Math.max(pct || 0, 0), 100) + "%";
        if (extraClass !== undefined) {
            bar.classList.remove("hw-warn", "hw-crit");
            if (extraClass) bar.classList.add(extraClass);
        }
    }

    async function fetchHardware() {
        const data = await apiFetch("/api/hardware");
        if (!data) return;

        // --- GPU ---
        const gpuCard = document.getElementById("hw-gpu-card");
        if (data.gpus && data.gpus.length > 0) {
            const g = data.gpus[0];  // primary GPU
            gpuCard.style.display = "";

            document.getElementById("hw-gpu-name").textContent = g.name || "--";

            // Temperature
            if (g.temperature_c !== null) {
                document.getElementById("hw-gpu-temp").textContent = g.temperature_c + "\u00B0C";
                setBar("hw-gpu-temp-bar", (g.temperature_c / 100) * 100, tempBarClass(g.temperature_c));
            } else {
                document.getElementById("hw-gpu-temp").textContent = "--";
                setBar("hw-gpu-temp-bar", 0);
            }

            // VRAM
            if (g.memory_used_mb !== null && g.memory_total_mb !== null) {
                const usedGB = (g.memory_used_mb / 1024).toFixed(1);
                const totalGB = (g.memory_total_mb / 1024).toFixed(1);
                const pct = (g.memory_used_mb / g.memory_total_mb) * 100;
                document.getElementById("hw-gpu-mem").textContent = usedGB + " / " + totalGB + " GB";
                setBar("hw-gpu-mem-bar", pct);
            } else {
                document.getElementById("hw-gpu-mem").textContent = "--";
                setBar("hw-gpu-mem-bar", 0);
            }

            // Utilization
            if (g.utilization_pct !== null) {
                document.getElementById("hw-gpu-util").textContent = g.utilization_pct.toFixed(0) + "%";
                setBar("hw-gpu-util-bar", g.utilization_pct);
            } else {
                document.getElementById("hw-gpu-util").textContent = "--";
                setBar("hw-gpu-util-bar", 0);
            }

            // Power
            if (g.power_draw_w !== null && g.power_limit_w !== null) {
                document.getElementById("hw-gpu-power").textContent =
                    g.power_draw_w.toFixed(0) + " / " + g.power_limit_w.toFixed(0) + " W";
                setBar("hw-gpu-power-bar", (g.power_draw_w / g.power_limit_w) * 100);
            } else {
                document.getElementById("hw-gpu-power").textContent = "--";
                setBar("hw-gpu-power-bar", 0);
            }

            // Fan
            const fanMetric = document.getElementById("hw-gpu-fan-metric");
            if (g.fan_speed_pct !== null) {
                fanMetric.style.display = "";
                document.getElementById("hw-gpu-fan").textContent = g.fan_speed_pct.toFixed(0) + "%";
                setBar("hw-gpu-fan-bar", g.fan_speed_pct);
            } else {
                fanMetric.style.display = "none";
            }
        } else {
            gpuCard.innerHTML = '<div class="hw-card-header"><span class="hw-icon hw-icon-gpu">GPU</span><span class="hw-name">--</span></div><div class="hw-unavailable">No GPU detected</div>';
        }

        // --- CPU ---
        const cpu = data.cpu;
        if (cpu && cpu.available) {
            var cpuLabel = cpu.cpu_count ? cpu.cpu_count + " cores" : "--";
            document.getElementById("hw-cpu-name").textContent = cpuLabel;

            // Usage
            document.getElementById("hw-cpu-usage").textContent =
                cpu.cpu_percent !== null ? cpu.cpu_percent.toFixed(1) + "%" : "--";
            setBar("hw-cpu-usage-bar", cpu.cpu_percent);

            // RAM
            document.getElementById("hw-cpu-ram").textContent =
                cpu.ram_used_gb + " / " + cpu.ram_total_gb + " GB (" + cpu.ram_percent + "%)";
            setBar("hw-cpu-ram-bar", cpu.ram_percent);

            // Temperature
            var tempMetric = document.getElementById("hw-cpu-temp-metric");
            if (cpu.cpu_temp_c !== undefined && cpu.cpu_temp_c !== null) {
                tempMetric.style.display = "";
                document.getElementById("hw-cpu-temp").textContent = cpu.cpu_temp_c + "\u00B0C";
                setBar("hw-cpu-temp-bar", (cpu.cpu_temp_c / 100) * 100, tempBarClass(cpu.cpu_temp_c));
            } else {
                tempMetric.style.display = "none";
            }
        } else {
            document.getElementById("hw-cpu-name").textContent = "--";
            document.getElementById("hw-cpu-usage").textContent = "N/A";
            document.getElementById("hw-cpu-ram").textContent = "psutil not installed";
        }
    }

    // -----------------------------------------------------------------------
    // Chart.js defaults
    // -----------------------------------------------------------------------
    function setupChartDefaults() {
        Chart.defaults.color = "#8b949e";
        Chart.defaults.borderColor = "#30363d";
        Chart.defaults.font.family = "'JetBrains Mono', 'Inter', sans-serif";
        Chart.defaults.font.size = 11;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.pointStyleWidth = 12;
        Chart.defaults.plugins.tooltip.backgroundColor = "#1c2333";
        Chart.defaults.plugins.tooltip.borderColor = "#30363d";
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleFont = { family: "'JetBrains Mono'", size: 11 };
        Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono'", size: 11 };
        Chart.defaults.plugins.tooltip.padding = 10;
        Chart.defaults.elements.point.radius = 0;
        Chart.defaults.elements.point.hoverRadius = 4;
        Chart.defaults.elements.line.borderWidth = 1.5;
    }

    function getZoomPlugin() {
        return {
            zoom: {
                pan: {
                    enabled: true,
                    mode: "x",
                    modifierKey: "shift",
                },
                zoom: {
                    wheel: { enabled: false },
                    pinch: { enabled: false },
                    drag: {
                        enabled: true,
                        backgroundColor: "rgba(88,166,255,0.15)",
                        borderColor: "rgba(88,166,255,0.45)",
                        borderWidth: 1,
                    },
                    modifierKey: "shift",
                    mode: "x",
                },
            },
        };
    }

    // -----------------------------------------------------------------------
    // Loss chart
    // -----------------------------------------------------------------------
    function buildLossChart(metrics, decisions) {
        const ctx = document.getElementById("chart-loss").getContext("2d");

        // Prepare data points
        const trainData = [];
        const valData = [];
        for (let i = 0; i < metrics.steps.length; i++) {
            const x = xAxisMode === "time" ? new Date(metrics.timestamps[i]) : metrics.steps[i];
            if (metrics.loss[i] !== null) {
                trainData.push({ x: x, y: metrics.loss[i] });
            }
            if (metrics.val_loss[i] !== null) {
                valData.push({ x: x, y: metrics.val_loss[i] });
            }
        }

        // Decision vertical lines as annotations
        const annotations = {};
        if (decisions && decisions.decisions) {
            decisions.decisions.forEach((d, idx) => {
                const x = xAxisMode === "time" ? new Date(d.timestamp) : d.step;
                annotations["dec_" + idx] = {
                    type: "line",
                    xMin: x,
                    xMax: x,
                    borderColor: policyColor(d.policy),
                    borderWidth: 1,
                    borderDash: [4, 4],
                    label: {
                        display: false,
                        content: d.policy || "decision",
                    },
                };
            });
        }

        const config = {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Train Loss",
                        data: trainData,
                        borderColor: "#58a6ff",
                        backgroundColor: "rgba(88, 166, 255, 0.05)",
                        fill: false,
                        tension: 0.1,
                    },
                    {
                        label: "Val Loss",
                        data: valData,
                        borderColor: "#f778ba",
                        backgroundColor: "rgba(247, 120, 186, 0.05)",
                        fill: false,
                        tension: 0.1,
                        borderWidth: 2,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false,
                },
                scales: {
                    x: xAxisConfig(),
                    y: {
                        title: { display: true, text: "Loss" },
                        grid: { color: "rgba(48,54,61,0.5)" },
                    },
                },
                plugins: {
                    ...getZoomPlugin(),
                    annotation: {
                        annotations: annotations,
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return ctx.dataset.label + ": " + (ctx.parsed.y != null ? ctx.parsed.y.toFixed(4) : "--");
                            },
                        },
                    },
                },
            },
        };

        if (chartLoss) {
            chartLoss.data = config.data;
            chartLoss.options = config.options;
            chartLoss.update("none");
        } else {
            chartLoss = new Chart(ctx, config);
        }
    }

    // -----------------------------------------------------------------------
    // Learning Rate chart
    // -----------------------------------------------------------------------
    function buildLRChart(metrics, decisions) {
        const ctx = document.getElementById("chart-lr").getContext("2d");

        const lrData = [];
        for (let i = 0; i < metrics.steps.length; i++) {
            const x = xAxisMode === "time" ? new Date(metrics.timestamps[i]) : metrics.steps[i];
            if (metrics.lr[i] !== null) {
                lrData.push({ x: x, y: metrics.lr[i] });
            }
        }

        // Highlight LR adjustment regions
        const annotations = {};
        if (decisions && decisions.decisions) {
            decisions.decisions.forEach((d, idx) => {
                if (d.action && d.action.param && d.action.param.includes("lr")) {
                    const x = xAxisMode === "time" ? new Date(d.timestamp) : d.step;
                    annotations["lr_" + idx] = {
                        type: "line",
                        xMin: x,
                        xMax: x,
                        borderColor: "#db6d28",
                        borderWidth: 2,
                        borderDash: [6, 3],
                        label: {
                            display: true,
                            content: (d.action.change_pct || 0).toFixed(0) + "%",
                            position: "start",
                            font: { size: 9, family: "'JetBrains Mono'" },
                            backgroundColor: "rgba(219,109,40,0.8)",
                            color: "#fff",
                            padding: 3,
                        },
                    };
                }
            });
        }

        const config = {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Learning Rate",
                        data: lrData,
                        borderColor: "#d29922",
                        backgroundColor: "rgba(210, 153, 34, 0.05)",
                        fill: true,
                        tension: 0.1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: xAxisConfig(),
                    y: {
                        title: { display: true, text: "LR" },
                        grid: { color: "rgba(48,54,61,0.5)" },
                        ticks: {
                            callback: function (v) {
                                return v.toExponential(1);
                            },
                        },
                    },
                },
                plugins: {
                    ...getZoomPlugin(),
                    annotation: { annotations },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                return "LR: " + (ctx.parsed.y != null ? ctx.parsed.y.toExponential(3) : "--");
                            },
                        },
                    },
                },
            },
        };

        if (chartLR) {
            chartLR.data = config.data;
            chartLR.options = config.options;
            chartLR.update("none");
        } else {
            chartLR = new Chart(ctx, config);
        }
    }

    // -----------------------------------------------------------------------
    // Gradient Norm chart
    // -----------------------------------------------------------------------
    function buildGradChart(gradData) {
        const ctx = document.getElementById("chart-grad").getContext("2d");

        if (!gradData || gradData.steps.length === 0) {
            // Show empty state
            const config = {
                type: "line",
                data: { datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: "No gradient norm data (training manager not active yet)",
                            color: "#6e7681",
                            font: { size: 12 },
                        },
                    },
                },
            };
            if (chartGrad) {
                chartGrad.data = config.data;
                chartGrad.options = config.options;
                chartGrad.update("none");
            } else {
                chartGrad = new Chart(ctx, config);
            }
            return;
        }

        const gnData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.grad_norm[i],
        }));
        const gnDeltaData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.grad_norm_delta ? gradData.grad_norm_delta[i] : null,
        }));
        const gnBitlinearData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.grad_norm_bitlinear ? gradData.grad_norm_bitlinear[i] : null,
        }));
        const gnSsmData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.grad_norm_ssm ? gradData.grad_norm_ssm[i] : null,
        }));
        const gnEmbeddingData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.grad_norm_embedding ? gradData.grad_norm_embedding[i] : null,
        }));
        const cfPctData = gradData.steps.map((s, i) => ({
            x: xAxisMode === "time" && gradData.timestamps ? new Date(gradData.timestamps[i]) : s,
            y: gradData.clipping_freq_pct
                ? gradData.clipping_freq_pct[i]
                : ((gradData.clipping_freq[i] || 0) <= 1.0
                    ? (gradData.clipping_freq[i] || 0) * 100.0
                    : (gradData.clipping_freq[i] || 0)),
        }));
        const gradClipThreshold = Number.isFinite(gradData.grad_clip_threshold)
            ? gradData.grad_clip_threshold
            : 1.0;
        const gradNormYMin = Array.isArray(gradData.grad_norm_ylim) ? gradData.grad_norm_ylim[0] : 0.99;
        const gradNormYMax = Array.isArray(gradData.grad_norm_ylim) ? gradData.grad_norm_ylim[1] : 1.001;
        const clipFreqYMin = Array.isArray(gradData.clip_freq_ylim) ? gradData.clip_freq_ylim[0] : 0.0;
        const clipFreqYMax = Array.isArray(gradData.clip_freq_ylim) ? gradData.clip_freq_ylim[1] : 100.0;

        const config = {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Grad Norm",
                        data: gnData,
                        borderColor: "#3fb950",
                        backgroundColor: "rgba(63, 185, 80, 0.05)",
                        fill: false,
                        tension: 0,
                        yAxisID: "y",
                    },
                    {
                        label: "Grad Delta",
                        data: gnDeltaData,
                        borderColor: "rgba(139, 148, 158, 0.9)",
                        backgroundColor: "rgba(139, 148, 158, 0.1)",
                        borderDash: [1, 3],
                        fill: false,
                        tension: 0,
                        yAxisID: "y",
                        hidden: true,
                    },
                    {
                        label: "BitLinear",
                        data: gnBitlinearData,
                        borderColor: "rgba(88, 166, 255, 0.9)",
                        backgroundColor: "rgba(88, 166, 255, 0.1)",
                        borderDash: [6, 3],
                        fill: false,
                        tension: 0,
                        yAxisID: "y",
                    },
                    {
                        label: "SSM",
                        data: gnSsmData,
                        borderColor: "rgba(210, 153, 34, 0.95)",
                        backgroundColor: "rgba(210, 153, 34, 0.1)",
                        borderDash: [4, 3],
                        fill: false,
                        tension: 0,
                        yAxisID: "y",
                    },
                    {
                        label: "Embedding",
                        data: gnEmbeddingData,
                        borderColor: "rgba(247, 120, 186, 0.9)",
                        backgroundColor: "rgba(247, 120, 186, 0.1)",
                        borderDash: [2, 2],
                        fill: false,
                        tension: 0,
                        yAxisID: "y",
                    },
                    {
                        label: "Clipping Freq (%)",
                        data: cfPctData,
                        borderColor: "rgba(248, 81, 73, 0.6)",
                        backgroundColor: "rgba(248, 81, 73, 0.1)",
                        fill: true,
                        tension: 0,
                        yAxisID: "y1",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                scales: {
                    x: {
                        ...xAxisConfig(),
                    },
                    y: {
                        type: "linear",
                        position: "left",
                        title: { display: true, text: "Grad Norm" },
                        grid: { color: "rgba(48,54,61,0.5)" },
                        min: gradNormYMin,
                        max: gradNormYMax,
                    },
                    y1: {
                        type: "linear",
                        position: "right",
                        title: { display: true, text: "Clip Freq (%)" },
                        grid: { drawOnChartArea: false },
                        min: clipFreqYMin,
                        max: clipFreqYMax,
                        ticks: {
                            callback: function (v) {
                                return v.toFixed(0) + "%";
                            },
                        },
                    },
                },
                plugins: {
                    ...getZoomPlugin(),
                    annotation: {
                        annotations: {
                            clipLine: {
                                type: "line",
                                yMin: gradClipThreshold,
                                yMax: gradClipThreshold,
                                yScaleID: "y",
                                borderColor: "rgba(248, 81, 73, 0.5)",
                                borderWidth: 1,
                                borderDash: [8, 4],
                                label: {
                                    display: true,
                                    content: "grad_clip=" + gradClipThreshold.toFixed(3),
                                    position: "end",
                                    font: { size: 9, family: "'JetBrains Mono'" },
                                    backgroundColor: "rgba(248,81,73,0.7)",
                                    color: "#fff",
                                    padding: 3,
                                },
                            },
                        },
                    },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                if (ctx.dataset.yAxisID === "y1") return "Clip Freq: " + ctx.parsed.y.toFixed(1) + "%";
                                return ctx.dataset.label + ": " + ctx.parsed.y.toFixed(4);
                            },
                        },
                    },
                },
            },
        };

        if (chartGrad) {
            chartGrad.data = config.data;
            chartGrad.options = config.options;
            chartGrad.update("none");
        } else {
            chartGrad = new Chart(ctx, config);
        }
    }

    // -----------------------------------------------------------------------
    // Tokens Progress chart
    // -----------------------------------------------------------------------
    function buildTokensChart(metrics) {
        const ctx = document.getElementById("chart-tokens").getContext("2d");

        const tokData = [];
        for (let i = 0; i < metrics.steps.length; i++) {
            const x = xAxisMode === "time" ? new Date(metrics.timestamps[i]) : metrics.steps[i];
            if (metrics.tokens[i] !== null) {
                tokData.push({ x: x, y: metrics.tokens[i] });
            }
        }

        // Target line
        const targetData = [];
        if (tokData.length > 0) {
            targetData.push({ x: tokData[0].x, y: MAX_TOKENS });
            targetData.push({ x: tokData[tokData.length - 1].x, y: MAX_TOKENS });
        }

        const config = {
            type: "line",
            data: {
                datasets: [
                    {
                        label: "Tokens Processed",
                        data: tokData,
                        borderColor: "#39d2c0",
                        backgroundColor: "rgba(57, 210, 192, 0.08)",
                        fill: true,
                        tension: 0.1,
                    },
                    {
                        label: "Target (8B)",
                        data: targetData,
                        borderColor: "rgba(88, 166, 255, 0.4)",
                        backgroundColor: "rgba(88, 166, 255, 0.05)",
                        borderDash: [8, 4],
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: xAxisConfig(),
                    y: {
                        title: { display: true, text: "Tokens" },
                        grid: { color: "rgba(48,54,61,0.5)" },
                        ticks: {
                            callback: function (v) {
                                if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
                                if (v >= 1e6) return (v / 1e6).toFixed(0) + "M";
                                return v;
                            },
                        },
                    },
                },
                plugins: {
                    ...getZoomPlugin(),
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                if (ctx.parsed.y >= 1e9) return ctx.dataset.label + ": " + (ctx.parsed.y / 1e9).toFixed(3) + "B";
                                if (ctx.parsed.y >= 1e6) return ctx.dataset.label + ": " + (ctx.parsed.y / 1e6).toFixed(1) + "M";
                                return ctx.dataset.label + ": " + ctx.parsed.y.toLocaleString();
                            },
                        },
                    },
                },
            },
        };

        if (chartTokens) {
            chartTokens.data = config.data;
            chartTokens.options = config.options;
            chartTokens.update("none");
        } else {
            chartTokens = new Chart(ctx, config);
        }
    }

    // -----------------------------------------------------------------------
    // Helpers for charts
    // -----------------------------------------------------------------------
    function xAxisConfig() {
        if (xAxisMode === "time") {
            return {
                type: "time",
                title: { display: true, text: "Time" },
                grid: { color: "rgba(48,54,61,0.5)" },
                time: {
                    tooltipFormat: "HH:mm:ss",
                    displayFormats: {
                        minute: "HH:mm",
                        hour: "HH:mm",
                        day: "MMM d",
                    },
                },
            };
        }
        return {
            type: "linear",
            title: { display: true, text: "Step" },
            grid: { color: "rgba(48,54,61,0.5)" },
            ticks: {
                callback: function (v) {
                    if (v >= 1e3) return (v / 1e3).toFixed(0) + "K";
                    return v;
                },
            },
        };
    }

    function policyColor(policy) {
        if (!policy) return "rgba(139,148,158,0.5)";
        const p = policy.toLowerCase();
        if (p.includes("reducelr") || p.includes("plateau")) return "#db6d28";
        if (p.includes("sgdr") || p.includes("restart")) return "#58a6ff";
        if (p.includes("clip")) return "#f85149";
        if (p.includes("overfit")) return "#f778ba";
        return "#bc8cff";
    }

    // -----------------------------------------------------------------------
    // Metrics fetch and chart update
    // -----------------------------------------------------------------------
    async function fetchMetrics() {
        const [metrics, decisions, gradNorms] = await Promise.all([
            apiFetch("/api/metrics"),
            apiFetch("/api/decisions?last_n=200&policy=all"),
            apiFetch("/api/grad_norms"),
        ]);

        if (metrics) {
            cachedMetrics = metrics;
            buildLossChart(metrics, decisions);
            buildLRChart(metrics, decisions);
            buildTokensChart(metrics);
        }

        if (decisions) {
            cachedDecisions = decisions;
            renderRegimeTimeline(decisions.regimes_timeline);
            renderDecisions(decisions.events || decisions.decisions);
        }

        if (gradNorms) {
            cachedGradNorms = gradNorms;
            buildGradChart(gradNorms);
        }
    }

    // -----------------------------------------------------------------------
    // Regime timeline bar
    // -----------------------------------------------------------------------
    const REGIME_COLORS = {
        UNSTABLE_EXPLORATION: "#f85149",
        HEALTHY_LEARNING: "#3fb950",
        NOISY_PLATEAU: "#d29922",
        REAL_PLATEAU: "#db6d28",
        CONVERGENCE: "#58a6ff",
        LATE_OVERFITTING: "#f778ba",
    };

    function renderRegimeTimeline(timeline) {
        const bar = document.getElementById("regime-bar");
        const legend = document.getElementById("regime-legend");

        if (!timeline || timeline.length === 0) {
            bar.innerHTML = '<div class="regime-empty">No regime data yet</div>';
            legend.innerHTML = "";
            return;
        }

        // Compute widths based on step ranges
        const maxStep = cachedMetrics ? Math.max(...cachedMetrics.steps) : MAX_STEPS;
        let html = "";
        const seenRegimes = new Set();

        for (let i = 0; i < timeline.length; i++) {
            const t = timeline[i];
            const startStep = t.step;
            const endStep = i < timeline.length - 1 ? timeline[i + 1].step : maxStep;
            const widthPct = Math.max(((endStep - startStep) / maxStep) * 100, 0.5);
            const label = t.regime.replace(/_/g, " ");
            seenRegimes.add(t.regime);

            html += `<div class="regime-segment" data-regime="${escapeHtml(t.regime)}"
                          style="width:${widthPct}%"
                          title="${label} (step ${startStep} - ${endStep})">
                        ${widthPct > 8 ? label : ""}
                     </div>`;
        }
        bar.innerHTML = html;

        // Legend
        let legendHtml = "";
        for (const [regime, color] of Object.entries(REGIME_COLORS)) {
            legendHtml += `<div class="regime-legend-item">
                <span class="regime-legend-dot" style="background:${color}"></span>
                <span>${regime.replace(/_/g, " ")}</span>
            </div>`;
        }
        legend.innerHTML = legendHtml;
    }

    // -----------------------------------------------------------------------
    // Decisions timeline
    // -----------------------------------------------------------------------
    function renderDecisions(decisions) {
        const container = document.getElementById("decisions-timeline");
        const countEl = document.getElementById("decisions-count");

        if (!decisions || decisions.length === 0) {
            container.innerHTML = '<div class="empty-state">No training manager events recorded yet.</div>';
            countEl.textContent = "";
            return;
        }

        // Apply local filters
        let filtered = decisions;

        // Policy filter
        if (currentPolicyFilter !== "all") {
            filtered = filtered.filter(
                (d) =>
                    (d.policy && d.policy.toLowerCase().includes(currentPolicyFilter.toLowerCase())) ||
                    (d.action && d.action.type && d.action.type.toLowerCase().includes(currentPolicyFilter.toLowerCase()))
            );
        }

        // Step range filter
        const minStep = parseInt(document.getElementById("filter-step-min").value) || 0;
        const maxStep = parseInt(document.getElementById("filter-step-max").value) || Infinity;
        if (minStep > 0 || maxStep < Infinity) {
            filtered = filtered.filter((d) => d.step >= minStep && d.step <= maxStep);
        }

        // Sort most recent first
        filtered = filtered.slice().reverse();

        if (filtered.length === 0) {
            container.innerHTML = '<div class="empty-state">No events match the current filters.</div>';
            countEl.textContent = `0 of ${decisions.length} events shown`;
            return;
        }

        let html = "";
        for (const d of filtered) {
            const hasAction = d.action !== null && d.action !== undefined;
            const regimeClass = "regime-" + (d.regime || "UNKNOWN");

            html += `<div class="decision-card ${hasAction ? "has-action" : ""}">`;
            html += `<div class="decision-card-header">`;
            html += `<span class="decision-step">Step ${formatNumber(d.step)}</span>`;
            html += `<span class="decision-tokens">${formatNumber(d.tokens)} tokens</span>`;
            html += `<span class="badge badge-regime ${regimeClass}">${escapeHtml(d.regime || "")}</span>`;
            if (d.policy) {
                html += `<span class="decision-policy">${escapeHtml(d.policy)}</span>`;
            }
            if (d.checkpoint_saved) {
                html += `<span class="decision-ckpt">CHECKPOINT</span>`;
            }
            html += `</div>`; // header

            if (hasAction) {
                const a = d.action;
                html += `<div class="decision-action">`;
                html += `<span class="action-param">${escapeHtml(a.param || a.type || "")}</span>`;
                if (a.before !== undefined && a.after !== undefined) {
                    html += `<span class="action-before">${formatActionVal(a.before)}</span>`;
                    html += `<span class="action-arrow">&rarr;</span>`;
                    html += `<span class="action-after">${formatActionVal(a.after)}</span>`;
                    if (a.change_pct !== undefined) {
                        const sign = a.change_pct >= 0 ? "+" : "";
                        html += `<span class="action-pct">(${sign}${a.change_pct.toFixed(1)}%)</span>`;
                    }
                }
                html += `</div>`;
            }

            if (d.justification) {
                html += `<div class="decision-justification collapsed" onclick="this.classList.toggle('collapsed')">${escapeHtml(d.justification)}</div>`;
            }

            html += `<div class="decision-timestamp">${formatTimestamp(d.timestamp)}</div>`;
            html += `</div>`;
        }

        container.innerHTML = html;
        countEl.textContent = `Showing ${filtered.length} of ${decisions.length} events`;
    }

    function formatActionVal(v) {
        if (v === null || v === undefined) return "--";
        if (typeof v === "number") {
            if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2);
            return v.toFixed(4);
        }
        return String(v);
    }

    // -----------------------------------------------------------------------
    // Checkpoints
    // -----------------------------------------------------------------------
    async function fetchCheckpoints() {
        const data = await apiFetch("/api/checkpoints");
        if (!data) return;

        const latest = data.latest_checkpoint || null;
        const latestCkptEl = document.getElementById("stat-last-checkpoint");
        if (latestCkptEl) {
            latestCkptEl.textContent = latest && latest.filename ? latest.filename : "--";
        }

        const tbody = document.getElementById("checkpoints-tbody");
        if (!data.checkpoints || data.checkpoints.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No checkpoints found.</td></tr>';
            return;
        }

        let html = "";
        for (const c of data.checkpoints) {
            const reasonClass = "reason-" + (c.reason || "scheduled");
            html += `<tr>
                <td>${escapeHtml(c.filename)}</td>
                <td>${c.step !== null ? formatNumber(c.step) : "--"}</td>
                <td>${c.size_mb}</td>
                <td>${formatTimestamp(c.timestamp)}</td>
                <td><span class="reason-badge ${reasonClass}">${escapeHtml(c.reason || "")}</span></td>
            </tr>`;
        }
        tbody.innerHTML = html;
    }

    // -----------------------------------------------------------------------
    // Polling management
    // -----------------------------------------------------------------------
    function startPolling() {
        stopPolling();

        // Immediate first load
        fetchStatus();
        fetchMetrics();
        fetchCheckpoints();
        fetchHardware();

        // Status: fast interval
        statusInterval = setInterval(fetchStatus, refreshRate * 1000);

        // Metrics + decisions: 3x the status interval
        const metricsRate = Math.max(refreshRate * 3, 15) * 1000;
        metricsInterval = setInterval(fetchMetrics, metricsRate);

        // Checkpoints: slower
        checkpointsInterval = setInterval(fetchCheckpoints, refreshRate * 1000);

        // Hardware: same as status interval
        hardwareInterval = setInterval(fetchHardware, refreshRate * 1000);
    }

    function stopPolling() {
        if (statusInterval) clearInterval(statusInterval);
        if (metricsInterval) clearInterval(metricsInterval);
        if (checkpointsInterval) clearInterval(checkpointsInterval);
        if (hardwareInterval) clearInterval(hardwareInterval);
        statusInterval = null;
        metricsInterval = null;
        checkpointsInterval = null;
        hardwareInterval = null;
    }

    // -----------------------------------------------------------------------
    // Last updated counter
    // -----------------------------------------------------------------------
    function updateLastUpdated() {
        const ago = Math.floor((Date.now() - lastUpdateTs) / 1000);
        document.getElementById("last-updated-ago").textContent = ago;
    }

    // -----------------------------------------------------------------------
    // Event bindings
    // -----------------------------------------------------------------------
    function bindEvents() {
        // Refresh button
        document.getElementById("btn-refresh").addEventListener("click", function () {
            fetchStatus();
            fetchMetrics();
            fetchCheckpoints();
            fetchHardware();
        });

        // Refresh interval selector
        document.getElementById("refresh-interval").addEventListener("change", function () {
            refreshRate = parseInt(this.value) || 10;
            startPolling();
        });

        // X-axis mode toggle
        document.getElementById("x-axis-mode").addEventListener("change", function () {
            xAxisMode = this.value;
            // Rebuild charts with cached data
            if (cachedMetrics) {
                buildLossChart(cachedMetrics, cachedDecisions);
                buildLRChart(cachedMetrics, cachedDecisions);
                buildTokensChart(cachedMetrics);
            }
            if (cachedGradNorms) {
                buildGradChart(cachedGradNorms);
            }
        });

        // Reset zoom
        document.getElementById("btn-reset-zoom").addEventListener("click", function () {
            [chartLoss, chartLR, chartGrad, chartTokens].forEach(function (c) {
                if (c) c.resetZoom();
            });
        });

        // Policy filter buttons
        document.querySelectorAll("#policy-filters .btn-filter").forEach(function (btn) {
            btn.addEventListener("click", function () {
                document.querySelectorAll("#policy-filters .btn-filter").forEach(function (b) {
                    b.classList.remove("active");
                });
                btn.classList.add("active");
                currentPolicyFilter = btn.dataset.policy;
                if (cachedDecisions) {
                    renderDecisions(cachedDecisions.events || cachedDecisions.decisions);
                }
            });
        });

        // Step range filter (debounced)
        let stepFilterTimeout = null;
        function onStepFilterChange() {
            clearTimeout(stepFilterTimeout);
            stepFilterTimeout = setTimeout(function () {
                if (cachedDecisions) {
                    renderDecisions(cachedDecisions.events || cachedDecisions.decisions);
                }
            }, 500);
        }
        document.getElementById("filter-step-min").addEventListener("input", onStepFilterChange);
        document.getElementById("filter-step-max").addEventListener("input", onStepFilterChange);
    }

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------
    function init() {
        setupChartDefaults();
        bindEvents();
        startPolling();
        // Update "last updated" counter every second
        setInterval(updateLastUpdated, 1000);
    }

    // Wait for DOM
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
