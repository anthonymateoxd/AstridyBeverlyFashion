/* Automatic reference presentation. The selected original File stays intact. */
document.querySelectorAll('form[data-presentation-form]').forEach((form) => {
  const input = form.querySelector('input[type="file"]');
  const submit = form.querySelector('[type="submit"]');
  if (!input || !submit) return;
  const panel = document.createElement('div');
  panel.style.cssText = 'flex:1 0 100%;display:grid;gap:8px;margin:12px 0;max-width:600px';
  const image = document.createElement('img');
  image.alt = 'Imagen de referencia del modelo';
  image.style.cssText = 'display:none;width:100%;max-height:480px;object-fit:contain;background:white';
  const original = document.createElement('a');
  original.textContent = 'Ver original';
  original.target = '_blank'; original.rel = 'noopener'; original.hidden = true;
  const status = document.createElement('small');
  status.setAttribute('aria-live', 'polite');
  panel.append(image, original, status); input.after(panel);
  let sequence = 0, processing = false, sourceURL = null, previewURL = null;
  input.addEventListener('change', async () => {
    const current = ++sequence;
    if (sourceURL) URL.revokeObjectURL(sourceURL);
    if (previewURL) URL.revokeObjectURL(previewURL);
    sourceURL = previewURL = null;
    const file = input.files[0];
    processing = !!file; submit.disabled = processing;
    original.hidden = !file; image.style.display = file ? 'block' : 'none';
    status.textContent = '';
    if (!file) return;
    sourceURL = URL.createObjectURL(file);
    original.href = sourceURL; image.src = sourceURL;
    status.textContent = 'Quitando el fondo de la fotografía…';
    try {
      const body = new FormData(); body.append('photo', file, file.name);
      const response = await fetch(form.dataset.presentationEndpoint, {
        method: 'POST', credentials: 'same-origin', cache: 'no-store', body,
      });
      if (!response.ok) throw new Error('No se pudo separar el fondo. Se conserva el original.');
      const blob = await response.blob();
      if (blob.type !== 'image/png' || !blob.size) throw new Error('Respuesta de imagen inválida.');
      if (current !== sequence) return;
      previewURL = URL.createObjectURL(blob); image.src = previewURL;
      status.textContent = 'Prenda sobre blanco. El original se conserva intacto.';
    } catch (error) {
      if (current === sequence) status.textContent = 'No se pudo separar el fondo. Puede guardar el original o tomar otra foto.';
    } finally {
      if (current === sequence) { processing = false; submit.disabled = false; }
    }
  });
  form.addEventListener('submit', (event) => {
    if (processing) {
      event.preventDefault(); event.stopImmediatePropagation();
      status.textContent = 'Espere a que termine el procesamiento de la fotografía.';
    }
  }, true);
});
