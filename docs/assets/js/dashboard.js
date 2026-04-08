document.addEventListener("DOMContentLoaded", async function () {
    const apiKey = window.SugarAuth.requireApiKey("oauth-login.html");
    if (!apiKey) return;

    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const chatSubmit = document.getElementById("chat-submit");
    const quotaRemaining = document.getElementById("quota-remaining");
    const quotaTotal = document.getElementById("quota-total");
    const quotaBar = document.querySelector(".quota-bar");
    const apiKeyField = document.getElementById("api-key-field");
    const toggleApiKeyBtn = document.getElementById("toggle-api-key");
    const copyApiKeyBtn = document.getElementById("copy-api-key");
    const welcomeText = document.getElementById("welcome-text");
    const userAvatar = document.getElementById("user-avatar");
    const adminLink = document.getElementById("admin-link");
    const logoutBtn = document.getElementById("logout-btn");

    const endpointRadios = document.querySelectorAll('input[name="endpoint-choice"]');
    const customPromptOptions = document.getElementById("custom-prompt-options");
    const customPromptTextarea = document.getElementById("custom-prompt");
    const presetBtns = document.querySelectorAll(".preset-btn");

    const temperatureInput = document.getElementById("temperature");
    const topPInput = document.getElementById("top-p");
    const topKInput = document.getElementById("top-k");
    const maxLengthInput = document.getElementById("max-length");
    const repetitionPenaltyInput = document.getElementById("repetition-penalty");
    const truncationInput = document.getElementById("truncation");

    apiKeyField.value = apiKey;

    logoutBtn.addEventListener("click", () => window.SugarAuth.logout());

    function updateQuota(quota) {
        if (!quota) return;
        quotaRemaining.textContent = quota.remaining;
        quotaTotal.textContent = quota.total;
        const pct = (quota.remaining / quota.total) * 100;
        quotaBar.style.width = `${pct}%`;
        if (pct < 20) quotaBar.style.backgroundColor = "#e74c3c";
        else if (pct < 50) quotaBar.style.backgroundColor = "#f39c12";
        else quotaBar.style.backgroundColor = "#2ecc71";
    }

    try {
        const user = await window.SugarApi.apiJson("/api/user");
        welcomeText.textContent = `Welcome, ${user.name || user.email || "User"}`;
        if (user.picture) {
            userAvatar.src = user.picture;
            userAvatar.style.display = "inline-block";
        } else {
            userAvatar.src = "https://ui-avatars.com/api/?name=" + encodeURIComponent(user.name || user.email || "User");
            userAvatar.style.display = "inline-block";
        }
        if (user.can_change_model) adminLink.style.display = "inline-block";
        updateQuota(user.quota);
    } catch (err) {
        if (err.status === 401) {
            window.SugarAuth.clearApiKey();
            window.location.href = "oauth-login.html";
            return;
        }
    }

    function addUserMessage(message) {
        const el = document.createElement("div");
        el.style.textAlign = "right";
        el.style.marginBottom = "10px";
        const span = document.createElement("span");
        span.style.background = "#3498db";
        span.style.color = "white";
        span.style.padding = "8px 12px";
        span.style.borderRadius = "15px 15px 0 15px";
        span.style.display = "inline-block";
        span.style.maxWidth = "80%";
        span.textContent = message;
        el.appendChild(span);
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addAIMessage(message) {
        const el = document.createElement("div");
        el.style.textAlign = "left";
        el.style.marginBottom = "10px";
        const span = document.createElement("span");
        span.style.background = "#f0f0f0";
        span.style.padding = "8px 12px";
        span.style.borderRadius = "15px 15px 15px 0";
        span.style.display = "inline-block";
        span.style.maxWidth = "80%";
        const formatted = message
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/```(.*?)```/gs, "<pre><code>$1</code></pre>")
            .replace(/`(.*?)`/g, "<code>$1</code>")
            .replace(/\n/g, "<br>");
        span.innerHTML = formatted;
        el.appendChild(span);
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const el = document.createElement("div");
        el.id = "typing-indicator";
        el.style.textAlign = "left";
        el.style.marginBottom = "10px";
        const span = document.createElement("span");
        span.style.background = "#f0f0f0";
        span.style.padding = "8px 12px";
        span.style.borderRadius = "15px";
        span.style.display = "inline-block";
        span.innerHTML = 'Sugar-AI is thinking<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
        el.appendChild(span);
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        const dots = span.querySelectorAll(".dot");
        let i = 0;
        const anim = setInterval(() => {
            dots.forEach(d => d.style.opacity = "0.2");
            dots[i].style.opacity = "1";
            i = (i + 1) % dots.length;
        }, 300);
        return { remove: () => { clearInterval(anim); el.remove(); } };
    }

    function getSelectedEndpoint() {
        const r = document.querySelector('input[name="endpoint-choice"]:checked');
        return r ? r.value : "ask";
    }

    function handleEndpointChange() {
        customPromptOptions.style.display =
            getSelectedEndpoint() === "ask-llm-prompted" ? "block" : "none";
    }

    function applyPreset(type) {
        if (!temperatureInput || !topPInput || !repetitionPenaltyInput) return;
        const presets = {
            code:     { t: "0.3", p: "0.8", r: "1.1" },
            creative: { t: "0.8", p: "0.9", r: "1.2" },
            factual:  { t: "0.4", p: "0.7", r: "1.0" }
        };
        const p = presets[type];
        if (!p) return;
        temperatureInput.value = p.t;
        topPInput.value = p.p;
        repetitionPenaltyInput.value = p.r;
    }

    async function sendMessage(message) {
        addUserMessage(message);
        const typing = showTypingIndicator();

        try {
            const endpoint = getSelectedEndpoint();
            let data;

            if (endpoint === "ask-llm-prompted") {
                const body = {
                    question: message,
                    custom_prompt: customPromptTextarea.value || "You are a helpful assistant. Provide clear and detailed answers.",
                    temperature: parseFloat(temperatureInput.value),
                    top_p: parseFloat(topPInput.value),
                    top_k: parseInt(topKInput.value),
                    max_length: parseInt(maxLengthInput.value),
                    repetition_penalty: parseFloat(repetitionPenaltyInput.value),
                    truncation: truncationInput.checked
                };
                data = await window.SugarApi.apiJson("/ask-llm-prompted", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
            } else {
                const path = endpoint === "ask" ? "/ask" : "/ask-llm";
                data = await window.SugarApi.apiJson(
                    `${path}?question=${encodeURIComponent(message)}`,
                    { method: "POST" }
                );
            }

            updateQuota(data.quota);
            typing.remove();
            const answer = data.answer || (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
            addAIMessage(answer);
        } catch (err) {
            typing.remove();
            addAIMessage(`Error: ${err.message}`);
        }
    }

    chatSubmit.addEventListener("click", () => {
        const msg = chatInput.value.trim();
        if (msg) { sendMessage(msg); chatInput.value = ""; }
    });
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            const msg = chatInput.value.trim();
            if (msg) { sendMessage(msg); chatInput.value = ""; }
        }
    });

    toggleApiKeyBtn.addEventListener("click", function () {
        if (apiKeyField.type === "password") {
            apiKeyField.type = "text";
            this.textContent = "Hide Key";
        } else {
            apiKeyField.type = "password";
            this.textContent = "Show Key";
        }
    });

    copyApiKeyBtn.addEventListener("click", async function () {
        try {
            await navigator.clipboard.writeText(apiKeyField.value);
        } catch (_) {
            const prev = apiKeyField.type;
            apiKeyField.type = "text";
            apiKeyField.select();
            document.execCommand("copy");
            apiKeyField.type = prev;
            window.getSelection().removeAllRanges();
        }
        const original = this.textContent;
        this.textContent = "Copied!";
        setTimeout(() => { this.textContent = original; }, 2000);
    });

    endpointRadios.forEach(r => r.addEventListener("change", handleEndpointChange));
    presetBtns.forEach(b => b.addEventListener("click", function () { applyPreset(this.dataset.preset); }));
    handleEndpointChange();
});
