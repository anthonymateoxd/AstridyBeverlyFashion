(() => {
        const initializeFlashes = () => {
          const wrapper = document.querySelector('.global-flash-wrap');
          if (!wrapper) return;
          wrapper.querySelectorAll('.global-flash').forEach((item) => {
            let dismissed = false;
            let timer;
            const category = (item.dataset.flashCategory || '').toLowerCase();
            const lifetime = category === 'error' ? 10000 : 6500;
            const dismiss = () => {
              if (dismissed) return;
              dismissed = true;
              window.clearTimeout(timer);
              item.classList.add('is-leaving');
              window.setTimeout(() => {
                item.remove();
                if (!wrapper.querySelector('.global-flash')) wrapper.remove();
              }, 230);
            };
            const schedule = (ms = lifetime) => {
              window.clearTimeout(timer);
              if (!dismissed) timer = window.setTimeout(dismiss, ms);
            };
            const close = item.querySelector('.global-flash-close');
            if (close) {
              close.hidden = false;
              close.addEventListener('click', dismiss);
            }
            item.addEventListener('mouseenter', () => window.clearTimeout(timer));
            item.addEventListener('mouseleave', () => schedule(3000));
            item.addEventListener('focusin', () => window.clearTimeout(timer));
            item.addEventListener('focusout', () => schedule(3000));
            schedule();
          });
        };
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', initializeFlashes, { once: true });
        } else {
          initializeFlashes();
        }
      })();

/*
 * Informes: conservar la posicion al cambiar entre Hoy / 7 dias / 30 dias.
 * No altera filtros, solicitudes GET, estadisticas ni exportaciones.
 */
(() => {
  const storageKey = 'astrid.informes.quickFilters.scroll.v1';
  const quickSelector = '.ana-quick';
  const detailsId = 'analisis-detallado';

  function initializeQuickFilters() {
    const quick = document.querySelector(quickSelector);
    if (!quick) return;

    // Recuperar la misma seccion despues del GET de filtros.
    let saved = null;
    try {
      const raw = window.sessionStorage.getItem(storageKey);
      if (raw) saved = JSON.parse(raw);
      window.sessionStorage.removeItem(storageKey);
    } catch (_) {
      // Almacenamiento no disponible: conservar comportamiento nativo.
    }

    if (saved && Number.isFinite(saved.timestamp) &&
        Date.now() - saved.timestamp >= 0 &&
        Date.now() - saved.timestamp < 120000 &&
        saved.pathname === window.location.pathname) {
      const params = new URLSearchParams(window.location.search);
      const datesMatch =
        params.get('date_from') === saved.from &&
        params.get('date_to') === saved.to;
      if (datesMatch) {
        const details = document.getElementById(detailsId);
        if (details && details.tagName.toLowerCase() === 'details') {
          details.open = true;
        }
        window.requestAnimationFrame(() => {
          window.requestAnimationFrame(() => {
            const anchor = document.querySelector(quickSelector);
            if (anchor && Number.isFinite(saved.offsetTop)) {
              const difference =
                anchor.getBoundingClientRect().top - saved.offsetTop;
              window.scrollTo({ top: Math.max(0, window.scrollY + difference), behavior: 'auto' });
            } else if (Number.isFinite(saved.scrollY)) {
              window.scrollTo({ top: Math.max(0, saved.scrollY), behavior: 'auto' });
            }
          });
        });
      }
    }

    quick.querySelectorAll('a[href]').forEach((link) => {
      link.addEventListener('click', (event) => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.ctrlKey || event.metaKey || event.shiftKey || event.altKey ||
            (link.target && link.target !== '_self')) return;
        const next = new URL(link.href, window.location.href);
        if (next.origin !== window.location.origin ||
            next.pathname !== window.location.pathname) return;
        try {
          window.sessionStorage.setItem(storageKey, JSON.stringify({
            pathname: next.pathname,
            from: next.searchParams.get('date_from'),
            to: next.searchParams.get('date_to'),
            scrollY: window.scrollY,
            offsetTop: quick.getBoundingClientRect().top,
            timestamp: Date.now()
          }));
        } catch (_) {
          // Un navegador sin sessionStorage seguira navegando normalmente.
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeQuickFilters, { once: true });
  } else {
    initializeQuickFilters();
  }
})();
