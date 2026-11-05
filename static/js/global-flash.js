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
