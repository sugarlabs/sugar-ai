(function () {
    const cfg = () => window.SUGAR_AI_CONFIG || {};

    let bannerEl = null;
    let bannerStatusEl = null;
    let wakeButtonEl = null;
    let polling = false;

    function ensureBanner() {
        if (bannerEl) return bannerEl;
        bannerEl = document.createElement("div");
        bannerEl.id = "wake-banner";
        bannerEl.style.cssText = [
            "position:fixed", "top:0", "left:0", "right:0",
            "background:#f39c12", "color:#fff", "padding:12px 20px",
            "font-family:Arial,sans-serif", "z-index:9999",
            "display:flex", "align-items:center", "justify-content:space-between",
            "gap:16px", "box-shadow:0 2px 8px rgba(0,0,0,0.2)"
        ].join(";");

        bannerStatusEl = document.createElement("span");
        bannerStatusEl.textContent = "The AI server is asleep.";

        wakeButtonEl = document.createElement("button");
        wakeButtonEl.textContent = "Wake Up Server";
        wakeButtonEl.style.cssText = "background:#fff;color:#2c3e50;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:bold;";
        wakeButtonEl.addEventListener("click", wakeServer);

        bannerEl.appendChild(bannerStatusEl);
        bannerEl.appendChild(wakeButtonEl);
        document.body.appendChild(bannerEl);
        return bannerEl;
    }

    function setStatus(text) {
        ensureBanner();
        bannerStatusEl.textContent = text;
    }

    function hideBanner() {
        if (bannerEl) {
            bannerEl.remove();
            bannerEl = null;
            bannerStatusEl = null;
            wakeButtonEl = null;
        }
    }

    function onBackendUnreachable() {
        if (polling) return;
        ensureBanner();
        setStatus("The AI server is asleep or unreachable.");
        if (!cfg().WAKE_LAMBDA_URL) {
            wakeButtonEl.style.display = "none";
            setStatus("The AI server is unreachable. Please try again later.");
        }
    }

    async function wakeServer() {
        if (polling) return;
        if (!cfg().WAKE_LAMBDA_URL) {
            setStatus("Wake server URL is not configured.");
            return;
        }
        wakeButtonEl.disabled = true;
        wakeButtonEl.style.opacity = "0.6";
        setStatus("Starting the AI server... this usually takes 2-3 minutes.");

        try {
            await fetch(cfg().WAKE_LAMBDA_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Wake-Token": cfg().WAKE_TOKEN || ""
                },
                body: JSON.stringify({ source: "sugar-ai-frontend" })
            });
        } catch (err) {
            setStatus("Could not reach the wake service: " + err.message);
            wakeButtonEl.disabled = false;
            wakeButtonEl.style.opacity = "1";
            return;
        }

        await pollUntilAlive();
    }

    async function pollUntilAlive() {
        polling = true;
        const start = Date.now();
        const interval = cfg().WAKE_POLL_INTERVAL_MS || 5000;
        const timeout = cfg().WAKE_POLL_TIMEOUT_MS || 300000;

        while (Date.now() - start < timeout) {
            const elapsed = Math.round((Date.now() - start) / 1000);
            setStatus("Waiting for the AI server to come online... (" + elapsed + "s elapsed)");
            try {
                const resp = await fetch(window.SugarApi.backendUrl("/api/health"), { method: "GET" });
                if (resp.ok) {
                    setStatus("Server is online. Reloading...");
                    polling = false;
                    setTimeout(() => window.location.reload(), 800);
                    return;
                }
            } catch (_) {
                // keep polling; backend still asleep
            }
            await new Promise(r => setTimeout(r, interval));
        }

        polling = false;
        setStatus("Timed out waiting for the server. Try again in a minute.");
        if (wakeButtonEl) {
            wakeButtonEl.disabled = false;
            wakeButtonEl.style.opacity = "1";
        }
    }

    window.SugarWake = { onBackendUnreachable, wakeServer, hideBanner };
})();
