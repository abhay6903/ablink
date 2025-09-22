// Theme toggle
const toggleBtn = document.getElementById('themeToggle');
const root = document.documentElement;

toggleBtn.addEventListener('click', () => {
  const isDark = root.getAttribute('data-theme') === 'dark';
  const newTheme = isDark ? 'light' : 'dark';
  root.setAttribute('data-theme', newTheme);
  try {
    localStorage.setItem('theme', newTheme);
  } catch(e) {
    console.warn('Could not save theme preference');
  }
  
  toggleBtn.style.transform = 'rotate(360deg)';
  setTimeout(() => {
    toggleBtn.style.transform = '';
  }, 300);
});

try {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) root.setAttribute('data-theme', savedTheme);
} catch(e) {
  console.warn('Could not load theme preference');
}

// Form elements
const connectForm = document.getElementById('connectForm');
const connectStatus = document.getElementById('connectStatus');
const schemaSelect = document.getElementById('schemaSelect');
const schemaInput = document.getElementById('schemaInput');
const tableSelect = document.getElementById('tableSelect');
const tableInput = document.getElementById('tableInput');
const runBtn = document.getElementById('runBtn');
const spinner = document.getElementById('spinner');
const barFill = document.getElementById('barFill');
const pct = document.getElementById('pct');
const runStatus = document.getElementById('runStatus');
const downloadLink = document.getElementById('downloadLink');
const showProfileBtn = document.getElementById('showProfileBtn');
const profileContainer = document.getElementById('profileContainer');
const profileFrame = document.getElementById('profileFrame');
const profileWidePanel = document.getElementById('profileWidePanel');
const profileFrameWide = document.getElementById('profileFrameWide');
const recordForm = document.getElementById('recordForm');
const resetBtn = document.getElementById('resetBtn');
const progressContainer = document.getElementById('progressContainer');
const checkDuplicateBtn = document.getElementById('checkDuplicateBtn');
const schemaBox = document.getElementById('schemaBox');
const tableBox = document.getElementById('tableBox');

let lastJobId = null;
let creds = null;
let jobId = null;
let running = false;
let sessionId = null;

function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'fadeOut 0.3s ease forwards';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

function updateStatus(element, message, type = '') {
  element.textContent = message;
  element.className = `status ${type}`;
  if(message) element.classList.add('fade-in');
}

function getCredsFromForm() {
  const data = Object.fromEntries(new FormData(connectForm));
  return data;
}

function maybeEnableRun() {
  const schema = schemaSelect.value || schemaInput.value.trim();
  const table = tableSelect.value || tableInput.value.trim();
  const canRun = !!(schema && table);
  
  runBtn.disabled = !canRun;
  runBtn.setAttribute('data-tooltip', canRun ? 'Run deduplication process' : 'Connect and select schema/table first');
  checkDuplicateBtn.disabled = !canRun;
  
  if (canRun) {
    runBtn.classList.remove('loading');
  }
}

connectForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const submitBtn = connectForm.querySelector('button');
  submitBtn.classList.add('loading');
  submitBtn.disabled = true;
  
  if (!sessionId) {
    try {
      const sres = await fetch('/session');
      const sjson = await sres.json();
      if (sjson.ok) sessionId = sjson.session_id;
    } catch(e) {
      console.warn('Could not get session ID');
    }
  }
  
  const data = Object.fromEntries(new FormData(connectForm));
  creds = data;
  updateStatus(connectStatus, 'Connecting to database...', 'warning');
  
  try {
    const res = await fetch('/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    
    const json = await res.json();
    
    if (!json.ok) {
      updateStatus(connectStatus, 'Connection failed: ' + json.error, 'error');
      schemaSelect.innerHTML = '<option value="">-- Connection failed --</option>';
      showNotification('Database connection failed', 'error');
      return;
    }
    
    updateStatus(connectStatus, '✅ Connected successfully', 'success');
    schemaBox.classList.add('connected');
    
    schemaSelect.innerHTML = '<option value="">-- Select schema --</option>' +
      json.schemas.map(s => `<option value="${s}">${s}</option>`).join('');
    
    showNotification('Database connected successfully!', 'success');
    maybeEnableRun();
    
  } catch (error) {
    updateStatus(connectStatus, 'Connection error: ' + error.message, 'error');
    showNotification('Connection error occurred', 'error');
  } finally {
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
  }
});

schemaSelect.addEventListener('change', async () => {
  tableSelect.innerHTML = '<option value="">-- Loading tables... --</option>';
  tableBox.classList.remove('connected');
  
  const schema = schemaSelect.value || schemaInput.value;
  if (!schema || !creds) return;
  
  updateStatus(runStatus, 'Loading tables...', 'warning');
  
  try {
    const res = await fetch('/tables', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...creds, schema})
    });
    
    const json = await res.json();
    
    if (!json.ok) {
      updateStatus(runStatus, 'Error loading tables: ' + json.error, 'error');
      tableSelect.innerHTML = '<option value="">-- Error loading tables --</option>';
      return;
    }
    
    updateStatus(runStatus, '', '');
    tableSelect.innerHTML = '<option value="">-- Select table --</option>' +
      json.tables.map(t => `<option value="${t}">${t}</option>`).join('');
    
    tableBox.classList.add('connected');
    maybeEnableRun();
    
  } catch (error) {
    updateStatus(runStatus, 'Error: ' + error.message, 'error');
  }
});

schemaInput.addEventListener('input', maybeEnableRun);
tableInput.addEventListener('input', maybeEnableRun);
schemaSelect.addEventListener('change', maybeEnableRun);
tableSelect.addEventListener('change', maybeEnableRun);

async function loadColumns() {
  const schema = schemaSelect.value || schemaInput.value.trim();
  const table = tableSelect.value || tableInput.value.trim();
  if (!schema || !table) return;
  
  const dataCreds = creds || getCredsFromForm();
  updateStatus(runStatus, 'Loading table columns...', 'warning');
  
  try {
    const res = await fetch('/columns', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...dataCreds, schema, table})
    });
    
    const json = await res.json();
    if (!json.ok) throw new Error(json.error || 'Failed to load columns');
    
    recordForm.innerHTML = json.columns.map(c => `
      <div class="field">
        <label>${c.name} <span style="color: var(--muted);">(${c.type})</span></label>
        <input data-col="${c.name}" placeholder="Enter ${c.name} value" />
      </div>
    `).join('');
    
    updateStatus(runStatus, `✅ Loaded ${json.columns.length} columns`, 'success');
    setTimeout(() => updateStatus(runStatus, '', ''), 2000);
    
  } catch(err) {
    updateStatus(runStatus, 'Error loading columns: ' + String(err), 'error');
  }
}

schemaSelect.addEventListener('change', loadColumns);
tableSelect.addEventListener('change', loadColumns);
schemaInput.addEventListener('blur', loadColumns);
tableInput.addEventListener('blur', loadColumns);

runBtn.addEventListener('click', async () => {
  if (running) return;
  if (!sessionId){
    try{
      const sres = await fetch('/session');
      const sjson = await sres.json();
      if (sjson.ok) sessionId = sjson.session_id;
    }catch{}
  }

  const schema = schemaSelect.value || schemaInput.value.trim();
  const table = tableSelect.value || tableInput.value.trim();

  if (!schema || !table) {
    updateStatus(runStatus, 'Please provide both a schema and a table.', 'error');
    return;
  }
  if (!creds) creds = getCredsFromForm();

  running = true;
  runBtn.disabled = true;
  progressContainer.classList.remove('hidden');
  spinner.classList.remove('hidden');
  barFill.style.width = '0%';
  pct.textContent = '0%';
  updateStatus(runStatus, 'Starting deduplication job...', 'warning');
  downloadLink.classList.add('disabled');

  try {
    const res = await fetch('/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...creds, schema, table, session_id: sessionId})
    });
    const json = await res.json();
    if (!json.ok) throw new Error(json.error);
    
    jobId = json.job_id;
    lastJobId = jobId;
    pollProgress();
  } catch(err) {
    updateStatus(runStatus, 'Error starting job: ' + err.message, 'error');
    running = false;
    runBtn.disabled = false;
    spinner.classList.add('hidden');
  }
});

async function pollProgress(){
  if (!jobId) return;
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/progress/${jobId}`);
      const json = await res.json();
      if (!json.ok) { 
        updateStatus(runStatus, 'Error fetching progress: ' + json.error, 'error');
        clearInterval(timer);
        return;
      }
      
      const p = json.progress ?? 0;
      barFill.style.width = `${p}%`;
      pct.textContent = `${p}%`;
      
      if (json.status === 'completed'){
        spinner.classList.add('hidden');
        updateStatus(runStatus, '✅ Deduplication complete!', 'success');
        downloadLink.classList.remove('disabled');
        running = false;
        runBtn.disabled = false;
        clearInterval(timer);
        showProfileBtn.disabled = false;
        showNotification('Deduplication job finished!', 'success');
      } else if (json.status === 'error'){
        spinner.classList.add('hidden');
        updateStatus(runStatus, 'Job failed: ' + (json.error || 'unknown error'), 'error');
        running = false;
        runBtn.disabled = false;
        clearInterval(timer);
        showNotification('Deduplication job failed.', 'error');
      } else {
        updateStatus(runStatus, json.status, 'warning');
      }
    } catch(err) {
      updateStatus(runStatus, 'Polling error: ' + err.message, 'error');
      running = false;
      runBtn.disabled = false;
      clearInterval(timer);
    }
  }, 2000);
}

showProfileBtn.addEventListener('click', async () => {
  if (!lastJobId){
    updateStatus(runStatus, 'You must run a deduplication job first.', 'warning');
    return;
  }
  updateStatus(runStatus, 'Loading data profile...', 'warning');
  // Load below the button in wide scrollable panel
  if (profileFrameWide) {
    profileFrameWide.src = `/profile_html/${lastJobId}`;
    profileFrameWide.onload = () => {
      updateStatus(runStatus, '', '');
      profileWidePanel.classList.remove('hidden');
      profileWidePanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    profileFrameWide.onerror = () => {
      updateStatus(runStatus, 'The data profile could not be loaded.', 'error');
    };
  } else {
    // Fallback to right panel if wide panel not present
    profileFrame.src = `/profile_html/${lastJobId}`;
    profileFrame.onload = () => { updateStatus(runStatus, '', ''); profileContainer.classList.remove('hidden'); };
    profileFrame.onerror = () => { updateStatus(runStatus, 'The data profile could not be loaded.', 'error'); };
  }
});

// ================== MODIFIED FUNCTION FOR DIRTY RECORDS ==================
checkDuplicateBtn.addEventListener('click', async () => {
  if (!lastJobId){
    updateStatus(runStatus, 'You must run a job before checking a record.', 'warning');
    showNotification('Run a job first', 'warning');
    return;
  }
  
  const record = {};
  let hasValue = false;
  recordForm.querySelectorAll('input[data-col]').forEach(inp => {
    const val = inp.value.trim();
    if(val) {
      record[inp.getAttribute('data-col')] = val;
      hasValue = true;
    }
  });

  if (!hasValue) {
    updateStatus(runStatus, 'Please enter at least one value to check.', 'warning');
    return;
  }
  
  checkDuplicateBtn.classList.add('loading');
  updateStatus(runStatus, 'Checking record...', 'warning');
  
  try {
    const res = await fetch('/check_record', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ job_id: lastJobId, record })
    });
    const json = await res.json();
    if (!json.ok){
      throw new Error(json.error || 'Could not check record');
    }
    
    // NEW: Handle the three possible outcomes from the backend
    if (json.result === 'duplicate'){
      let detail = `Result: Duplicate (Cluster: ${json.cluster_id})`;
      updateStatus(runStatus, detail, 'success');
      showNotification('Entered Record is a duplicate.', 'success');
    } else if (json.result === 'potential_duplicate') {
      // Show this as a 'warning' status to indicate it's not a perfect match
      const probPercent = (json.match_probability * 100).toFixed(1);
      let detail = `Result: Potential Duplicate (Match Score: ${probPercent}%) Cluster: ${json.cluster_id}`;
      updateStatus(runStatus, detail, 'warning');
      showNotification('Found a potential duplicate for the entered record.', 'warning');
    }
    else { // 'unique'
      updateStatus(runStatus, 'Result: Unique', 'success');
      showNotification('Entered Record appears to be unique.', 'success');
    }
  } catch(err) {
    updateStatus(runStatus, 'Error: ' + err.message, 'error');
  } finally {
    checkDuplicateBtn.classList.remove('loading');
  }
});

// ================== NEW: ROBUST DOWNLOAD HANDLER ==================
downloadLink.addEventListener('click', async () => {
  if (!lastJobId) {
    showNotification('No job has been run yet.', 'warning');
    return;
  }

  downloadLink.classList.add('loading');
  downloadLink.disabled = true;

  try {
    const res = await fetch(`/report/${lastJobId}`);

    if (!res.ok) {
      // If the server returns an error, show it as a notification
      const errorJson = await res.json();
      throw new Error(errorJson.error || `HTTP error! status: ${res.status}`);
    }

    // Process the successful file download
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'reports.csv'; // The filename for the download
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    a.remove();
    
    showNotification('Report download started.', 'success');

  } catch (err) {
    showNotification(`Failed to download report: ${err.message}`, 'error');
  } finally {
    downloadLink.classList.remove('loading');
    downloadLink.disabled = false;
  }
});


resetBtn.addEventListener('click', async () => {
  try{
    await fetch('/reset', { method: 'POST' });
  }catch{}
  
  creds = null; jobId = null; lastJobId = null; running = false; sessionId = null;
  
  connectForm.reset();
  schemaSelect.innerHTML = '<option value="">-- Connect to see schemas --</option>';
  tableSelect.innerHTML = '<option value="">-- Select schema first --</option>';
  schemaInput.value = '';
  tableInput.value = '';
  recordForm.innerHTML = `<div class="field"><label>No table selected</label><input disabled placeholder="Connect and select a table first" /></div>`;
  
  progressContainer.classList.add('hidden');
  spinner.classList.add('hidden');
  barFill.style.width = '0%';
  pct.textContent = '0%';
  
  runBtn.disabled = true;
  showProfileBtn.disabled = true;
  checkDuplicateBtn.disabled = true;
  downloadLink.classList.add('disabled');
  
  profileContainer.classList.add('hidden');
  profileFrame.src = 'about:blank';
  
  updateStatus(connectStatus, '', '');
  updateStatus(runStatus, '', '');
  
  schemaBox.classList.remove('connected');
  tableBox.classList.remove('connected');
  
  // Representative UI removed; nothing to clear here
  
  showNotification('Application has been reset.', 'warning');
});