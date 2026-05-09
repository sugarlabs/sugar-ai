document.addEventListener("DOMContentLoaded", async function () {
    const apiKey = window.SugarAuth.requireApiKey("oauth-login.html");
    if (!apiKey) return;

    const messageEl = document.getElementById("message");
    const tabButtons = document.querySelectorAll(".tablinks");
    const tabContents = document.querySelectorAll(".tab-content");
    document.getElementById("logout-btn").addEventListener("click", () => window.SugarAuth.logout());

    tabButtons.forEach(btn => {
        btn.addEventListener("click", function () {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.style.display = "none");
            this.classList.add("active");
            document.getElementById(this.dataset.tab).style.display = "block";
        });
    });

    try {
        const user = await window.SugarApi.apiJson("/api/user");
        if (!user.can_change_model) {
            window.location.href = "dashboard.html";
            return;
        }
    } catch (err) {
        if (err.status === 401) {
            window.SugarAuth.clearApiKey();
            window.location.href = "oauth-login.html";
            return;
        }
    }

    function showMessage(text, success = true) {
        messageEl.className = "message " + (success ? "success" : "error");
        messageEl.textContent = text;
        messageEl.style.display = "block";
        setTimeout(() => { messageEl.style.display = "none"; }, 4000);
    }

    function escape(str) {
        return (str || "").replace(/[&<>"']/g, s => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        })[s]);
    }

    async function refresh() {
        try {
            const keys = await window.SugarApi.apiJson("/api/admin/keys");
            renderPending(keys.pending || []);
            renderApproved(keys.approved || []);
            renderDenied(keys.denied || []);
        } catch (err) {
            showMessage("Failed to load keys: " + err.message, false);
        }
    }

    function renderPending(items) {
        const body = document.getElementById("pending-body");
        body.innerHTML = "";
        for (const k of items) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${escape(k.name)}</td>
                <td>${escape(k.email)}</td>
                <td>${escape(k.request_reason)}</td>
                <td>${escape(k.created_at)}</td>
                <td>
                    <button class="action-btn approve" data-id="${k.id}" data-action="approve">Approve</button>
                    <button class="action-btn deny" data-id="${k.id}" data-action="deny">Deny</button>
                </td>`;
            body.appendChild(tr);
        }
    }

    function renderApproved(items) {
        const body = document.getElementById("approved-body");
        body.innerHTML = "";
        for (const k of items) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${escape(k.name)}</td>
                <td>${escape(k.email)}</td>
                <td>${escape(k.key)}</td>
                <td>${k.can_change_model ? "Yes" : "No"}</td>
                <td>${k.is_active ? "Active" : "Inactive"}</td>
                <td>${escape(k.created_at)}</td>
                <td>
                    <button class="action-btn toggle-admin" data-id="${k.id}" data-action="toggle-admin">Toggle Admin</button>
                    <button class="action-btn toggle-status" data-id="${k.id}" data-action="toggle-status">${k.is_active ? "Deactivate" : "Activate"}</button>
                </td>`;
            body.appendChild(tr);
        }
    }

    function renderDenied(items) {
        const body = document.getElementById("denied-body");
        body.innerHTML = "";
        for (const k of items) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${escape(k.name)}</td>
                <td>${escape(k.email)}</td>
                <td>${escape(k.request_reason)}</td>
                <td>${escape(k.created_at)}</td>
                <td>
                    <button class="action-btn approve" data-id="${k.id}" data-action="approve">Approve</button>
                </td>`;
            body.appendChild(tr);
        }
    }

    document.body.addEventListener("click", async function (e) {
        const btn = e.target.closest(".action-btn");
        if (!btn) return;
        const id = btn.dataset.id;
        const action = btn.dataset.action;
        btn.disabled = true;
        try {
            await window.SugarApi.apiJson(`/admin/${action}/${id}`, { method: "POST" });
            showMessage("Action completed.", true);
            await refresh();
        } catch (err) {
            showMessage("Action failed: " + err.message, false);
            btn.disabled = false;
        }
    });

    refresh();
});
