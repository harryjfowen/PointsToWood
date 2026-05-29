const container = document.getElementById('slider');
const afterWrap = container.querySelector('.img-after-wrap');
const afterImg  = container.querySelector('.img-after');
const handle    = document.getElementById('handle');

function setPosition(x) {
  const rect = container.getBoundingClientRect();
  const pct  = Math.min(Math.max((x - rect.left) / rect.width, 0), 1);
  afterWrap.style.width = pct * 100 + '%';
  afterImg.style.width  = (1 / pct) * 100 + '%';
  handle.style.left     = pct * 100 + '%';
}

// Init at 50%
afterImg.addEventListener('load', () => setPosition(
  container.getBoundingClientRect().left + container.getBoundingClientRect().width * 0.5
));

let dragging = false;
container.addEventListener('mousedown',  e => { dragging = true; setPosition(e.clientX); });
container.addEventListener('touchstart', e => { dragging = true; setPosition(e.touches[0].clientX); }, { passive: true });
window.addEventListener('mousemove',  e => { if (dragging) setPosition(e.clientX); });
window.addEventListener('touchmove',  e => { if (dragging) setPosition(e.touches[0].clientX); }, { passive: true });
window.addEventListener('mouseup',   () => dragging = false);
window.addEventListener('touchend',  () => dragging = false);
