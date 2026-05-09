(function () {
    const cfg = () => window.SUGAR_AI_CONFIG || {};

    function backendUrl(path) {
        const base = (cfg().API_BASE_URL || "").replace(/\/$/, "");
        return base + (path.startsWith("/") ? path : "/" + path);
    }

    async function apiFetch(path, options = {}) {
        const headers = Object.assign({}, options.headers || {});
        const apiKey = window.SugarAuth && window.SugarAuth.getApiKey();
        if (apiKey && !headers["X-API-Key"]) headers["X-API-Key"] = apiKey;

        try {
            return await fetch(backendUrl(path), Object.assign({}, options, { headers }));
        } catch (err) {
            // TypeError from fetch means the backend is unreachable (likely
            // asleep). Hand off to the wake banner before rethrowing.
            if (window.SugarWake) window.SugarWake.onBackendUnreachable(err);
            throw err;
        }
    }

    async function apiJson(path, options = {}) {
        const resp = await apiFetch(path, options);
        const text = await resp.text();
        let body = null;
        try { body = text ? JSON.parse(text) : null; } catch (_) { body = text; }
        if (!resp.ok) {
            const detail = (body && body.detail) || resp.statusText;
            const err = new Error(resp.status + ": " + detail);
            err.status = resp.status;
            err.body = body;
            throw err;
        }
        return body;
    }

    window.SugarApi = { backendUrl, apiFetch, apiJson };
})();
