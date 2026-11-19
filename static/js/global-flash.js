/* Only transient, general server flashes. Inspection alerts are independent. */
(() => {
  'use strict';
  const storageKey = 'astrid.closed-general-flashes.v1';
  let closed;
  try { closed = new Set(JSON.parse(sessionStorage.getItem(storageKey) || '[]')); }
  catch { closed = new Set(); }
  const remember = id => {
    if (!id) return;
    closed.add(id);
    try { sessionStorage.setItem(storageKey, JSON.stringify([...closed].slice(-200))); }
    catch { /* Storage unavailable: closing and timers still work. */ }
  };
  const protectedNotice = notice => {
    if (notice.matches('[data-persistent], [data-critical], [data-inspection-result], .persistent, .critical, .warning, .danger, .inspection-result, .defect')) return true;
    const text = notice.querySelector('.global-flash-message')?.textContent || notice.textContent;
    const normalized = text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
    return /defectos? detectad|anomalias? detectad|resultados? de inspeccion|intervencion humana|requiere (revision|atencion)|alerta critica/.test(normalized);
  };
  const remove = notice => {
    const wrap = notice.closest('.global-flash-wrap');
    notice.remove();
    if (wrap && !wrap.querySelector('.global-flash')) wrap.remove();
  };
  document.querySelectorAll('.global-flash-wrap .global-flash').forEach(notice => {
    if (notice.dataset.flashInitialized) return;
    notice.dataset.flashInitialized = 'true';
    const category = notice.dataset.flashCategory;
    const duration = category === 'error' ? 8000 : ['success', 'info'].includes(category) ? 5000 : null;
    if (duration === null || protectedNotice(notice)) return;
    const id = notice.dataset.flashEvent;
    if (id && closed.has(id)) { remove(notice); return; }
    const button = notice.querySelector('.global-flash-close');
    if (!button) return;
    button.hidden = false;
    let remaining = duration, started = 0, timer = null, leaving = false;
    const pauses = new Set();
    const dismiss = () => {
      if (leaving || protectedNotice(notice)) return;
      leaving = true;
      clearTimeout(timer);
      remember(id);
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { remove(notice); return; }
      notice.classList.add('is-leaving');
      notice.addEventListener('transitionend', event => {
        if (event.target === notice && event.propertyName === 'opacity') remove(notice);
      });
      window.setTimeout(() => remove(notice), 240);
    };
    const schedule = () => {
      if (leaving || pauses.size || timer !== null) return;
      started = performance.now();
      timer = window.setTimeout(() => { timer = null; dismiss(); }, Math.max(0, remaining));
    };
    const pause = reason => {
      if (timer !== null) {
        clearTimeout(timer);
        remaining = Math.max(0, remaining - (performance.now() - started));
        timer = null;
      }
      pauses.add(reason);
    };
    const resume = reason => { pauses.delete(reason); schedule(); };
    notice.addEventListener('pointerenter', event => {
      if (event.pointerType === 'mouse' || event.pointerType === 'pen') pause('hover');
    });
    notice.addEventListener('pointerleave', () => resume('hover'));
    notice.addEventListener('focusin', () => pause('focus'));
    notice.addEventListener('focusout', event => {
      if (!notice.contains(event.relatedTarget)) resume('focus');
    });
    button.addEventListener('click', dismiss);
    if (window.matchMedia('(hover: hover)').matches && notice.matches(':hover')) pause('hover');
    if (notice.contains(document.activeElement)) pause('focus');
    schedule();
  });
})();
