// tab functionality for admin panel
window.openTab = function openTab(evt, tabName) {
  const tabcontent = document.getElementsByClassName("tab-content");
  for (let i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }
  
  const tablinks = document.getElementsByClassName("tablinks");
  for (let i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
};

function modelPayloadFromForm() {
  const maxLengthValue = document.getElementById("model-max-length").value.trim();
  const apiKeyValue = document.getElementById("model-api-key").value.trim();

  return {
    name: document.getElementById("model-name").value.trim(),
    provider_type: document.getElementById("model-provider-type").value.trim(),
    base_url: document.getElementById("model-base-url").value.trim(),
    model_name: document.getElementById("model-model-name").value.trim(),
    api_key: apiKeyValue || null,
    max_model_length: maxLengthValue ? Number(maxLengthValue) : null,
    is_active: document.getElementById("model-is-active").checked,
  };
}

function getModelModal() {
  return document.getElementById("model-modal");
}

function setModelFormMode(isEditing) {
  document.getElementById("model-form-title").textContent = isEditing ? "Edit Model" : "Add Model";
  document.getElementById("model-submit-btn").textContent = isEditing ? "Save Changes" : "Add Model";
}

function resetModelForm() {
  document.getElementById("model-form").reset();
  document.getElementById("model-id").value = "";
  document.getElementById("model-provider-type").value = "openai_compatible";
  setModelFormMode(false);
}

function openModelModal() {
  const modal = getModelModal();
  if (!modal) {
    return;
  }
  if (typeof modal.showModal === "function") {
    modal.showModal();
  } else {
    modal.setAttribute("open", "open");
  }
}

window.openCreateModelModal = function openCreateModelModal() {
  resetModelForm();
  openModelModal();
};

window.closeModelModal = function closeModelModal() {
  const modal = getModelModal();
  if (!modal) {
    return;
  }
  resetModelForm();
  if (typeof modal.close === "function") {
    modal.close();
  } else {
    modal.removeAttribute("open");
  }
};

window.editModel = function editModel(model) {
  openModelModal();
  document.getElementById("model-id").value = model.id;
  document.getElementById("model-name").value = model.name || "";
  document.getElementById("model-provider-type").value = model.provider_type || "openai_compatible";
  document.getElementById("model-base-url").value = model.base_url || "";
  document.getElementById("model-model-name").value = model.model_name || "";
  document.getElementById("model-api-key").value = model.api_key || "";
  document.getElementById("model-max-length").value = model.max_model_length || "";
  document.getElementById("model-is-active").checked = Boolean(model.is_active);
  setModelFormMode(true);
};

async function submitModelForm(event) {
  event.preventDefault();

  const modelId = document.getElementById("model-id").value;
  const payload = modelPayloadFromForm();
  const url = modelId ? `/admin/models/${modelId}` : "/admin/models";
  const method = modelId ? "PUT" : "POST";

  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(data.detail || "Failed to save model.");
    return;
  }

  closeModelModal();
  window.location.reload();
}

window.activateModel = async function activateModel(modelId) {
  const response = await fetch(`/admin/models/${modelId}/activate`, {
    method: "POST",
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(data.detail || "Failed to activate model.");
    return;
  }
  window.location.reload();
};

window.deleteModel = async function deleteModel(modelId, modelName) {
  if (!window.confirm(`Delete model "${modelName}"?`)) {
    return;
  }

  const response = await fetch(`/admin/models/${modelId}`, {
    method: "DELETE",
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    alert(data.detail || "Failed to delete model.");
    return;
  }
  window.location.reload();
};

// initialize first tab as active when page loads
document.addEventListener('DOMContentLoaded', function() {
  const firstTab = document.querySelector('.tablinks.active');
  if (firstTab) {
    firstTab.click();
  }
  const modelForm = document.getElementById("model-form");
  if (modelForm) {
    modelForm.addEventListener("submit", submitModelForm);
    resetModelForm();
  }
  document.addEventListener("keydown", function(event) {
    const modal = getModelModal();
    if (event.key === "Escape" && modal && modal.hasAttribute("open")) {
      closeModelModal();
    }
  });
});
