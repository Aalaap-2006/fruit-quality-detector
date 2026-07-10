(() => {
  'use strict';

  // ---------- Elements ----------
  const tabs = document.querySelectorAll('.tab');
  const tabIndicator = document.getElementById('tabIndicator');
  const panelUpload = document.getElementById('panel-upload');
  const panelCamera = document.getElementById('panel-camera');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');

  const video = document.getElementById('video');
  const captureBtn = document.getElementById('captureBtn');
  const cameraHint = document.getElementById('cameraHint');
  const canvas = document.getElementById('canvas');

  const previewWrap = document.getElementById('previewWrap');
  const previewImg = document.getElementById('previewImg');
  const scanLine = document.getElementById('scanLine');
  const scanStatus = document.getElementById('scanStatus');
  const predictBtn = document.getElementById('predictBtn');
  const resetBtn = document.getElementById('resetBtn');

  const resultWrap = document.getElementById('resultWrap');
  const verdictBadge = document.getElementById('verdictBadge');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictText = document.getElementById('verdictText');
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugeValue = document.getElementById('gaugeValue');
  const resultNote = document.getElementById('resultNote');
  const scanAgainBtn = document.getElementById('scanAgainBtn');

  const errorWrap = document.getElementById('errorWrap');
  const errorText = document.getElementById('errorText');
  const errorResetBtn = document.getElementById('errorResetBtn');

  const scannerCard = document.querySelector('.scanner-card');

  let currentBlob = null;
  let stream = null;
  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 86; // matches r=86 in SVG

  // ---------- Tabs ----------
  function setActiveTab(name) {
    tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    tabIndicator.style.transform = name === 'camera' ? 'translateX(100%)' : 'translateX(0)';
    panelUpload.classList.toggle('active', name === 'upload');
    panelCamera.classList.toggle('active', name === 'camera');

    if (name === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
  }

  tabs.forEach(tab => {
    tab.addEventListener('click', () => setActiveTab(tab.dataset.tab));
  });

  // ---------- Camera ----------
  async function startCamera() {
    if (stream) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      video.srcObject = stream;
      cameraHint.textContent = 'Frame the fruit, then tap the shutter';
    } catch (err) {
      cameraHint.textContent = 'Camera access denied or unavailable';
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
  }

  captureBtn.addEventListener('click', () => {
    if (!stream) return;
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      if (blob) showPreview(blob);
    }, 'image/jpeg', 0.92);
  });

  // ---------- Upload ----------
  dropzone.addEventListener('click', (e) => {
    // label already triggers input via 'for', but keep explicit for clarity when JS intercepts
  });

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (file) showPreview(file);
  });

  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) showPreview(file);
  });

  // ---------- Preview ----------
  function showPreview(blob) {
    currentBlob = blob;
    previewImg.src = URL.createObjectURL(blob);
    scannerCard.querySelectorAll('.tabs, .tab-panel').forEach(el => el.style.display = 'none');
    previewWrap.hidden = false;
    resultWrap.hidden = true;
    errorWrap.hidden = true;
    stopCamera();
  }

  function resetToStart() {
    currentBlob = null;
    previewWrap.hidden = true;
    resultWrap.hidden = true;
    errorWrap.hidden = true;
    fileInput.value = '';
    scannerCard.querySelectorAll('.tabs, .tab-panel').forEach(el => el.style.display = '');
    const activeTab = document.querySelector('.tab.active');
    setActiveTab(activeTab ? activeTab.dataset.tab : 'upload');
  }

  resetBtn.addEventListener('click', resetToStart);
  scanAgainBtn.addEventListener('click', resetToStart);
  errorResetBtn.addEventListener('click', resetToStart);

  // ---------- Predict ----------
  predictBtn.addEventListener('click', async () => {
    if (!currentBlob) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing…';
    scanLine.classList.add('active');
    scanStatus.classList.add('active');

    const formData = new FormData();
    formData.append('image', currentBlob, 'capture.jpg');

    try {
      const response = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.error || `Server responded with ${response.status}`);
      }
      const data = await response.json();
      showResult(data);
    } catch (err) {
      showError(err.message || 'Could not reach the prediction service.');
    } finally {
      predictBtn.disabled = false;
      predictBtn.textContent = 'Analyze Freshness';
      scanLine.classList.remove('active');
      scanStatus.classList.remove('active');
    }
  });

  function showResult(data) {
    const isFresh = data.quality === 'Fresh';
    const confidence = Math.max(0, Math.min(100, Number(data.confidence) || 0));

    previewWrap.hidden = true;
    resultWrap.hidden = false;

    verdictBadge.classList.remove('fresh', 'rotten');
    verdictBadge.classList.add(isFresh ? 'fresh' : 'rotten');
    verdictIcon.textContent = isFresh ? '●' : '●';
    verdictText.textContent = isFresh ? 'Fresh' : 'Rotten';

    gaugeFill.classList.remove('fresh', 'rotten');
    gaugeFill.classList.add(isFresh ? 'fresh' : 'rotten');

    gaugeFill.style.strokeDasharray = `${GAUGE_CIRCUMFERENCE}`;
    gaugeFill.style.strokeDashoffset = `${GAUGE_CIRCUMFERENCE}`;
    // force reflow so the transition plays
    void gaugeFill.getBoundingClientRect();
    const offset = GAUGE_CIRCUMFERENCE * (1 - confidence / 100);
    gaugeFill.style.strokeDashoffset = `${offset}`;

    animateValue(gaugeValue, 0, Math.round(confidence), 1100);

    resultNote.textContent = isFresh
      ? 'Surface color and texture are consistent with recently harvested produce.'
      : 'Surface markers indicate decay — texture and discoloration patterns are past peak freshness.';
  }

  function animateValue(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(from + (to - from) * eased);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function showError(message) {
    previewWrap.hidden = true;
    resultWrap.hidden = true;
    errorWrap.hidden = false;
    errorText.textContent = message;
  }

  // ---------- Init ----------
  setActiveTab('upload');
})();
