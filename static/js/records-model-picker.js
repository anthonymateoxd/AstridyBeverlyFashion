/* Progressive enhancement confined to the inspection-history model filter. */
(() => {
  'use strict';
  const select = document.getElementById('recordsModel');
  if (!select || select.dataset.searchReady) return;
  const root = select.closest('.records-model-picker');
  if (!root) return;
  const label = document.querySelector('label[for="recordsModel"]');
  const options = Array.from(select.options, option => ({
    value: option.value, label: option.textContent.trim().replace(/\s+/g, ' ')
  }));
  const normalize = text => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase('es');
  const input = document.createElement('input');
  input.id = 'recordsModelSearch';
  input.type = 'text';
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.placeholder = 'Buscar código o nombre';
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-autocomplete', 'list');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-controls', 'recordsModelOptions');
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'records-model-toggle';
  toggle.textContent = '▾';
  toggle.setAttribute('aria-label', 'Mostrar todos los modelos');
  toggle.setAttribute('aria-controls', 'recordsModelOptions');
  toggle.setAttribute('aria-expanded', 'false');
  const popup = document.createElement('div');
  popup.className = 'records-model-popup';
  popup.hidden = true;
  const list = document.createElement('div');
  list.id = 'recordsModelOptions';
  list.setAttribute('role', 'listbox');
  list.setAttribute('aria-label', 'Modelos');
  const status = document.createElement('div');
  status.className = 'records-model-status';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  popup.append(list, status);
  root.append(input, toggle, popup);
  let visible = [], active = -1;
  const selectedLabel = () => options.find(option => option.value === select.value)?.label || 'Todos';
  const highlight = index => {
    active = index;
    Array.from(list.children).forEach((item, i) => item.classList.toggle('is-active', i === active));
    const item = list.children[active];
    if (item) {
      input.setAttribute('aria-activedescendant', item.id);
      item.scrollIntoView({ block: 'nearest' });
    } else input.removeAttribute('aria-activedescendant');
  };
  const close = () => {
    popup.hidden = true;
    input.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-expanded', 'false');
    input.removeAttribute('aria-activedescendant');
    input.value = selectedLabel();
  };
  const choose = index => {
    if (!visible[index]) return;
    select.value = visible[index].value;
    select.dispatchEvent(new Event('change', { bubbles: true }));
    close();
  };
  const open = (query = '') => {
    const terms = normalize(query.trim()).split(/\s+/).filter(Boolean);
    visible = options.filter(option => !option.value || terms.every(term => normalize(option.label).includes(term)));
    list.replaceChildren();
    visible.forEach((option, index) => {
      const item = document.createElement('div');
      item.id = `recordsModelOption-${index}`;
      item.className = 'records-model-option';
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(option.value === select.value));
      item.textContent = option.label;
      item.addEventListener('click', () => { choose(index); input.focus(); close(); });
      list.append(item);
    });
    status.textContent = terms.length && visible.length === 1 ? 'Sin modelos coincidentes' : '';
    status.hidden = !status.textContent;
    popup.hidden = false;
    input.setAttribute('aria-expanded', 'true');
    toggle.setAttribute('aria-expanded', 'true');
    const selected = visible.findIndex(option => option.value === select.value);
    highlight(terms.length ? Math.min(1, visible.length - 1) : Math.max(0, selected));
  };
  input.addEventListener('focus', () => { open(); input.select(); });
  input.addEventListener('click', () => { if (popup.hidden) open(); });
  input.addEventListener('input', () => open(input.value));
  input.addEventListener('keydown', event => {
    if (event.isComposing) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (popup.hidden) open();
      else highlight(Math.max(0, Math.min(visible.length - 1, active + (event.key === 'ArrowDown' ? 1 : -1))));
    } else if (event.key === 'Enter' && !popup.hidden) {
      event.preventDefault();
      choose(active);
    } else if (event.key === 'Escape' && !popup.hidden) {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === 'Tab') close();
  });
  // Keep the input focused while choosing an option with mouse or touch.
  popup.addEventListener('mousedown', event => event.preventDefault());
  toggle.addEventListener('click', () => {
    const wasOpen = !popup.hidden;
    input.focus();
    if (wasOpen) close();
    else open();
  });
  root.addEventListener('focusout', event => { if (!root.contains(event.relatedTarget)) close(); });
  document.addEventListener('pointerdown', event => { if (!root.contains(event.target)) close(); });
  select.addEventListener('change', () => { input.value = selectedLabel(); });
  select.form?.addEventListener('reset', () => setTimeout(close, 0));
  input.value = selectedLabel();
  if (label) label.htmlFor = input.id;
  select.hidden = true;
  select.dataset.searchReady = 'true';
})();
