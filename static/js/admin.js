// tab functionality for admin panel
function openTab(evt, tabName) {
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
}

// send an admin action in the background and reload the panel, instead of
// leaving the admin on the endpoint's raw JSON response
async function submitAdminAction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("button");
  button.disabled = true;

  try {
    const response = await fetch(form.action, { method: "POST", credentials: "same-origin" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `${response.status} ${response.statusText}`);
    }
    window.location.reload();
  } catch (error) {
    alert(`Action failed: ${error.message}`);
    button.disabled = false;
  }
}

// initialize first tab as active when page loads
document.addEventListener('DOMContentLoaded', function() {
  document.querySelector('.tablinks.active').click();

  document.querySelectorAll('form[action^="/admin/"]').forEach(function(form) {
    form.addEventListener("submit", submitAdminAction);
  });
});
