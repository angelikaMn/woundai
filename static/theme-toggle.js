/**
 * Theme Toggle Script for WoundAI
 * Handles light/dark mode switching with smooth animations
 */

// Get theme from localStorage or default to 'light'
const getTheme = () => {
  return localStorage.getItem("theme") || "light";
};

// Set theme and save to localStorage
const setTheme = (theme) => {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);

  // Update meta theme-color for mobile browsers
  const metaThemeColor = document.querySelector('meta[name="theme-color"]');
  if (metaThemeColor) {
    metaThemeColor.setAttribute(
      "content",
      theme === "dark" ? "#2d2528" : "#f7f1de",
    );
  }
};

// Toggle between light and dark theme
const toggleTheme = () => {
  const currentTheme = getTheme();
  const newTheme = currentTheme === "light" ? "dark" : "light";
  setTheme(newTheme);

  // Add a subtle animation ripple effect
  createRipple();
};

// Create ripple animation effect on toggle
const createRipple = () => {
  const toggle = document.querySelector(".theme-toggle");
  if (!toggle) return;

  const ripple = document.createElement("span");
  ripple.style.cssText = `
        position: absolute;
        border-radius: 50%;
        background: var(--color-primary);
        opacity: 0.5;
        width: 10px;
        height: 10px;
        animation: ripple 0.6s ease-out;
        pointer-events: none;
    `;

  toggle.appendChild(ripple);

  setTimeout(() => {
    ripple.remove();
  }, 600);
};

// Add ripple animation keyframes dynamically
const addRippleAnimation = () => {
  if (!document.querySelector("#ripple-keyframes")) {
    const style = document.createElement("style");
    style.id = "ripple-keyframes";
    style.textContent = `
            @keyframes ripple {
                from {
                    transform: scale(0);
                    opacity: 0.5;
                }
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
        `;
    document.head.appendChild(style);
  }
};

// Initialize theme on page load
const initTheme = () => {
  // Prevent transition flash on page load
  document.documentElement.classList.add("no-transition");

  // Set initial theme
  const theme = getTheme();
  setTheme(theme);

  // Remove no-transition class after a brief delay
  setTimeout(() => {
    document.documentElement.classList.remove("no-transition");
  }, 50);

  // Add ripple animation
  addRippleAnimation();

  // Add click event listener to theme toggle button
  const toggleButton = document.querySelector(".theme-toggle");
  if (toggleButton) {
    toggleButton.addEventListener("click", toggleTheme);

    // Add keyboard accessibility
    toggleButton.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleTheme();
      }
    });
  }
};

// Initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initTheme);
} else {
  initTheme();
}

// Expose toggleTheme globally for manual triggers if needed
window.toggleTheme = toggleTheme;
