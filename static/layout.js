// layout.js
// Dynamically calculate the vertical space above the main content
// and set the CSS variable --panel-top-offset so panels can size
// themselves to leave a fixed bottom gap (see styles.css).

(function () {
  const bottomGapVar = '--bottom-gap';
  const topOffsetVar = '--panel-top-offset';

  function updatePanelTopOffset() {
    const mainGrid = document.querySelector('.main-grid');
    if (!mainGrid) return;

    // mainGrid.getBoundingClientRect().top returns the distance from the
    // top of the viewport to the top edge of the main grid. Use that
    // as the top offset so panels occupy the remaining viewport space.
    const rect = mainGrid.getBoundingClientRect();
    const top = Math.max(0, Math.floor(rect.top));

    document.documentElement.style.setProperty(topOffsetVar, top + 'px');
  }

  // Run on load and resize/orientationchange. Also observe DOM changes
  // that could affect layout (images loading, dynamic content).
  window.addEventListener('resize', updatePanelTopOffset);
  window.addEventListener('orientationchange', updatePanelTopOffset);
  window.addEventListener('load', updatePanelTopOffset);
  document.addEventListener('DOMContentLoaded', updatePanelTopOffset);

  // MutationObserver to catch async DOM changes that might shift the layout
  // (e.g., images injected after fetch). Keep this lightweight.
  const mo = new MutationObserver(() => {
    updatePanelTopOffset();
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  // Expose for debugging if needed
  window.__updatePanelTopOffset = updatePanelTopOffset;
})();
