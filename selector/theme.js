const button = document.querySelector(".lights");

function sync() {
  const dark = document.documentElement.classList.contains("dark");
  button.setAttribute("aria-pressed", String(dark));
  button.setAttribute("aria-label", dark ? "turn lights on" : "turn lights off");
}

button.addEventListener("click", () => {
  const dark = !document.documentElement.classList.contains("dark");
  document.documentElement.classList.toggle("dark", dark);
  localStorage.setItem("theme", dark ? "dark" : "light");
  sync();
});

sync();
