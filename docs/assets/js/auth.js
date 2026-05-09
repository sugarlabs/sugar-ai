(function () {
    const STORAGE_KEY = "sugar_ai_api_key";

    function getApiKey() {
        return localStorage.getItem(STORAGE_KEY);
    }

    function setApiKey(key) {
        if (key) localStorage.setItem(STORAGE_KEY, key);
    }

    function clearApiKey() {
        localStorage.removeItem(STORAGE_KEY);
    }

    function captureKeyFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const key = params.get("api_key");
        if (!key) return null;
        setApiKey(key);
        // Strip api_key from the visible URL so it is not bookmarked or leaked
        // via Referer.
        params.delete("api_key");
        const query = params.toString();
        const newUrl = window.location.pathname + (query ? "?" + query : "") + window.location.hash;
        window.history.replaceState({}, document.title, newUrl);
        return key;
    }

    function requireApiKey(redirectTo) {
        const key = captureKeyFromUrl() || getApiKey();
        if (!key) {
            window.location.href = redirectTo || "oauth-login.html";
            return null;
        }
        return key;
    }

    function logout() {
        clearApiKey();
        window.location.href = "index.html";
    }

    window.SugarAuth = {
        getApiKey,
        setApiKey,
        clearApiKey,
        captureKeyFromUrl,
        requireApiKey,
        logout
    };
})();
