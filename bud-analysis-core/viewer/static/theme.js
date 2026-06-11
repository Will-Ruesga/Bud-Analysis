const root = document.documentElement;
const toggle = document.querySelector("[data-theme-toggle]");
const icon = toggle?.querySelector("[data-theme-icon]");

function render() {
  if (icon) {
    icon.textContent = root.dataset.theme === "dark" ? "☀" : "☾";
  }
}

toggle?.addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("theme", root.dataset.theme);
  render();
});

render();
