document.addEventListener('DOMContentLoaded', function() {
  // Close alert messages when the close button is clicked
  document.querySelectorAll('.alert .close').forEach(button => {
    button.addEventListener('click', function() {
      this.parentElement.style.display = 'none';
    });
  });
  
  setTimeout(function() {
    document.querySelectorAll('.alert:not(.alert-persistent)').forEach(alert => {
      alert.style.display = 'none';
    });
  }, 5000);
});
