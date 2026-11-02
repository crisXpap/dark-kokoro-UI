/**
 * voice-select.js — custom, accessible dropdown for the voice picker.
 *
 * Wraps the native <select> so the rest of the app can keep reading
 * select.value and listening for the "change" event, while the visible UI
 * is a dark, glassmorphic listbox with keyboard support.
 */
(function () {
  const select = document.getElementById('voiceSelect');
  if (!select || select.dataset.enhanced === 'true') return;

  const root = select.closest('.select');
  if (!root) return;

  select.dataset.enhanced = 'true';

  const baseId = select.id || 'voice-select';
  const valueId = `${baseId}-value`;
  const menuId = `${baseId}-menu`;

  /* ---------------------------------------------------------------- trigger */
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'select-trigger';
  trigger.setAttribute('aria-haspopup', 'listbox');
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-controls', menuId);
  trigger.innerHTML = `
    <span class="select-value" id="${valueId}"></span>
    <svg class="select-chevron" viewBox="0 0 12 8" aria-hidden="true" focusable="false">
      <path d="M1 1.5 6 6.5 11 1.5" fill="none" stroke="currentColor"
        stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
    </svg>`;

  /* ----------------------------------------------------------------- menu */
  const menu = document.createElement('ul');
  menu.className = 'select-menu';
  menu.id = menuId;
  menu.setAttribute('role', 'listbox');
  menu.tabIndex = -1;

  const label = select.closest('.card')?.querySelector('.label');
  if (label) {
    if (!label.id) label.id = `${baseId}-label`;
    trigger.setAttribute('aria-labelledby', `${label.id} ${valueId}`);
    menu.setAttribute('aria-labelledby', label.id);
  } else {
    trigger.setAttribute('aria-label', 'Voice');
    menu.setAttribute('aria-label', 'Voice');
  }

  root.appendChild(trigger);
  root.appendChild(menu);
  root.classList.add('is-enhanced');

  const valueEl = trigger.querySelector('.select-value');

  let isOpen = false;
  let activeIndex = -1;

  const options = () => Array.from(select.options);

  function selectedIndex() {
    const index = options().findIndex((option) => option.selected);
    return index < 0 ? 0 : index;
  }

  function syncSelection() {
    const all = options();
    const index = all.findIndex((option) => option.selected);
    const current = index >= 0 ? all[index] : all[0];

    valueEl.textContent = current ? current.textContent : 'No voices available';

    Array.from(menu.children).forEach((item, i) => {
      item.setAttribute('aria-selected', i === index ? 'true' : 'false');
    });
  }

  function rebuild() {
    menu.innerHTML = '';

    options().forEach((option, index) => {
      const item = document.createElement('li');
      item.id = `${baseId}-option-${index}`;
      item.className = 'select-option';
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', 'false');
      item.textContent = option.textContent;

      item.addEventListener('click', () => commit(index));
      item.addEventListener('pointerenter', (event) => {
        if (event.pointerType === 'mouse') setActive(index, false);
      });

      menu.appendChild(item);
    });

    syncSelection();

    if (isOpen) setActive(selectedIndex(), false);
  }

  function setActive(index, scroll = true) {
    const items = menu.children;
    if (!items.length) {
      activeIndex = -1;
      menu.removeAttribute('aria-activedescendant');
      return;
    }

    activeIndex = ((index % items.length) + items.length) % items.length;

    Array.from(items).forEach((item, i) => {
      item.classList.toggle('is-active', i === activeIndex);
    });

    const active = items[activeIndex];
    menu.setAttribute('aria-activedescendant', active.id);

    if (scroll) {
      const top = active.offsetTop;
      const bottom = top + active.offsetHeight;
      if (top < menu.scrollTop) {
        menu.scrollTop = top - 6;
      } else if (bottom > menu.scrollTop + menu.clientHeight) {
        menu.scrollTop = bottom - menu.clientHeight + 6;
      }
    }
  }

  function open(index = selectedIndex()) {
    if (!menu.children.length) return;

    isOpen = true;
    root.classList.add('is-open');
    trigger.setAttribute('aria-expanded', 'true');
    setActive(index);
    menu.focus({ preventScroll: true });
  }

  function close(returnFocus = false) {
    if (!isOpen) return;

    isOpen = false;
    root.classList.remove('is-open');
    trigger.setAttribute('aria-expanded', 'false');
    menu.removeAttribute('aria-activedescendant');

    if (returnFocus) trigger.focus({ preventScroll: true });
  }

  function commit(index) {
    const option = select.options[index];
    if (!option) return;

    if (select.value !== option.value) {
      select.value = option.value;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }

    syncSelection();
    close(true);
  }

  /* --------------------------------------------------------------- events */
  trigger.addEventListener('click', (event) => {
    event.preventDefault();
    if (isOpen) close(true);
    else open();
  });

  trigger.addEventListener('keydown', (event) => {
    if (
      event.key === 'ArrowDown' ||
      event.key === 'ArrowUp' ||
      event.key === 'Enter' ||
      event.key === ' '
    ) {
      event.preventDefault();
      if (!isOpen) open();
      return;
    }

    if (event.key === 'Escape' && isOpen) {
      event.preventDefault();
      close();
    }
  });

  menu.addEventListener('keydown', (event) => {
    const items = menu.children;
    if (!items.length) return;

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setActive(activeIndex + 1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        setActive(activeIndex - 1);
        break;
      case 'Home':
        event.preventDefault();
        setActive(0);
        break;
      case 'End':
        event.preventDefault();
        setActive(items.length - 1);
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (activeIndex >= 0) commit(activeIndex);
        break;
      case 'Escape':
        event.preventDefault();
        close(true);
        break;
      case 'Tab':
        close();
        break;
    }
  });

  document.addEventListener(
    'pointerdown',
    (event) => {
      if (isOpen && !root.contains(event.target)) close();
    },
    true
  );

  document.addEventListener('focusin', (event) => {
    if (isOpen && !root.contains(event.target)) close();
  });

  /* Keep the menu in sync when app.js repopulates the native <select>. */
  let rebuildQueued = false;
  const queueRebuild = () => {
    if (rebuildQueued) return;
    rebuildQueued = true;
    queueMicrotask(() => {
      rebuildQueued = false;
      rebuild();
    });
  };

  new MutationObserver(queueRebuild).observe(select, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  rebuild();
})();
