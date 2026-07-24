/**
 * TaskFlow — Kanban Task Manager
 * Client-side application logic
 */

const API_URL = '/api/tasks';

// ---- State ----
let tasks = [];
let editingTaskId = null;
let draggedCard = null;

// ---- DOM refs ----
const board = document.getElementById('board');
const modalOverlay = document.getElementById('modal-overlay');
const modalTitle = document.getElementById('modal-title');
const toastContainer = document.getElementById('toast-container');
const formTitle = document.getElementById('form-title');
const formDescription = document.getElementById('form-description');
const formColumn = document.getElementById('form-column');
const formPriority = document.getElementById('form-priority');
const formDueDate = document.getElementById('form-due-date');
const subtaskList = document.getElementById('subtask-list');
const subtaskInput = document.getElementById('subtask-input');
const btnSave = document.getElementById('btn-save');
const btnDelete = document.getElementById('btn-delete');
const statTodo = document.getElementById('stat-todo');
const statProgress = document.getElementById('stat-progress');
const statDone = document.getElementById('stat-done');
const searchInput = document.getElementById('search-input');
const filterPriority = document.getElementById('filter-priority');
const filterDue = document.getElementById('filter-due');
const filterClear = document.getElementById('filter-clear');
const analyticsPanel = document.getElementById('analytics-panel');
const analyticsToggleBtn = document.getElementById('analytics-toggle');

// ---- API Helpers ----
async function apiGet() {
    const res = await fetch(API_URL);
    const data = await res.json();
    return data.tasks || [];
}
async function apiPost(task) {
    const res = await fetch(API_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(task) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    return data;
}
async function apiPut(id, task) {
    const res = await fetch(`${API_URL}?id=${id}`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(task) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    return data;
}
async function apiPatch(id, partial) {
    const res = await fetch(`${API_URL}?id=${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(partial) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    return data;
}
async function apiDelete(id) {
    const res = await fetch(`${API_URL}?id=${id}`, { method: 'DELETE' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    return data;
}

// ---- Toast ----
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        ${type === 'success'
            ? '<svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"/></svg>'
            : '<svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"/></svg>'
        }
        <span>${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ======================================================
//  THEME TOGGLE
// ======================================================
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('taskflow-theme', next);
}

function loadTheme() {
    const saved = localStorage.getItem('taskflow-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
}

// ======================================================
//  SEARCH & FILTER
// ======================================================
function getTaskDueCategory(task) {
    if (!task.dueDate) return 'none';
    const due = new Date(task.dueDate + 'T23:59:59');
    const now = new Date();
    const diffDays = Math.ceil((due - now) / (1000 * 60 * 60 * 24));
    if (diffDays < 0) return 'overdue';
    if (diffDays === 0) return 'today';
    if (diffDays <= 7) return 'week';
    return 'future';
}

function applyFilters() {
    const query = searchInput.value.toLowerCase().trim();
    const priority = filterPriority.value;
    const due = filterDue.value;

    const hasFilters = query || priority || due;
    filterClear.style.display = hasFilters ? 'flex' : 'none';

    document.querySelectorAll('.task-card').forEach(card => {
        const id = parseInt(card.dataset.id);
        const task = tasks.find(t => t.id === id);
        if (!task) return;

        let show = true;

        // Text search
        if (query) {
            const text = (task.title + ' ' + (task.description || '')).toLowerCase();
            if (!text.includes(query)) show = false;
        }

        // Priority filter
        if (priority && task.priority !== priority) show = false;

        // Due date filter
        if (due) {
            const cat = getTaskDueCategory(task);
            if (due === 'none' && cat !== 'none') show = false;
            else if (due === 'overdue' && cat !== 'overdue') show = false;
            else if (due === 'today' && cat !== 'today') show = false;
            else if (due === 'week' && cat !== 'today' && cat !== 'week') show = false;
        }

        card.classList.toggle('filtered-out', !show);
    });
}

function clearFilters() {
    searchInput.value = '';
    filterPriority.value = '';
    filterDue.value = '';
    filterClear.style.display = 'none';
    document.querySelectorAll('.task-card').forEach(c => c.classList.remove('filtered-out'));
}

// ======================================================
//  ANALYTICS
// ======================================================
function toggleAnalytics() {
    analyticsPanel.classList.toggle('open');
    analyticsToggleBtn.classList.toggle('active');
    if (analyticsPanel.classList.contains('open')) updateAnalytics();
}

function updateAnalytics() {
    const total = tasks.length;
    if (total === 0) {
        document.getElementById('ring-fill').style.strokeDashoffset = 326.73;
        document.getElementById('ring-label').textContent = '0%';
        ['todo', 'inprogress', 'done', 'high', 'medium', 'low'].forEach(k => {
            document.getElementById(`bar-${k}`).style.width = '0%';
            document.getElementById(`bar-${k}-val`).textContent = '0';
        });
        ['qs-total', 'qs-overdue', 'qs-due-soon', 'qs-no-date'].forEach(k => {
            document.getElementById(k).textContent = '0';
        });
        return;
    }

    const doneCount = tasks.filter(t => t.column === 'done').length;
    const todoCount = tasks.filter(t => t.column === 'todo').length;
    const ipCount = tasks.filter(t => t.column === 'inprogress').length;
    const pct = Math.round((doneCount / total) * 100);

    // Ring
    const circumference = 326.73;
    document.getElementById('ring-fill').style.strokeDashoffset = circumference - (circumference * pct / 100);
    document.getElementById('ring-label').textContent = pct + '%';

    // Breakdown bars
    const setBar = (id, count) => {
        document.getElementById(`bar-${id}`).style.width = (count / total * 100) + '%';
        document.getElementById(`bar-${id}-val`).textContent = count;
    };
    setBar('todo', todoCount);
    setBar('inprogress', ipCount);
    setBar('done', doneCount);

    const highCount = tasks.filter(t => t.priority === 'high').length;
    const medCount = tasks.filter(t => t.priority === 'medium').length;
    const lowCount = tasks.filter(t => t.priority === 'low').length;
    setBar('high', highCount);
    setBar('medium', medCount);
    setBar('low', lowCount);

    // Quick stats
    document.getElementById('qs-total').textContent = total;
    document.getElementById('qs-overdue').textContent = tasks.filter(t => getTaskDueCategory(t) === 'overdue').length;
    document.getElementById('qs-due-soon').textContent = tasks.filter(t => {
        const cat = getTaskDueCategory(t);
        return cat === 'today' || cat === 'week';
    }).length;
    document.getElementById('qs-no-date').textContent = tasks.filter(t => !t.dueDate).length;
}

// ======================================================
//  CONFETTI
// ======================================================
const confettiCanvas = document.getElementById('confetti-canvas');
const confettiCtx = confettiCanvas.getContext('2d');
let confettiParticles = [];
let confettiAnimating = false;

function resizeConfettiCanvas() {
    confettiCanvas.width = window.innerWidth;
    confettiCanvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeConfettiCanvas);
resizeConfettiCanvas();

function launchConfetti() {
    const colors = ['#e8a84c', '#5cb87a', '#5c9ee0', '#e05c5c', '#b87ae0', '#e0c75c'];
    confettiParticles = [];

    for (let i = 0; i < 120; i++) {
        confettiParticles.push({
            x: window.innerWidth / 2 + (Math.random() - 0.5) * 200,
            y: window.innerHeight / 2,
            vx: (Math.random() - 0.5) * 18,
            vy: -(Math.random() * 14 + 6),
            w: Math.random() * 8 + 4,
            h: Math.random() * 6 + 3,
            color: colors[Math.floor(Math.random() * colors.length)],
            rotation: Math.random() * 360,
            rotationSpeed: (Math.random() - 0.5) * 12,
            gravity: 0.35 + Math.random() * 0.15,
            opacity: 1,
            decay: 0.008 + Math.random() * 0.008
        });
    }

    if (!confettiAnimating) {
        confettiAnimating = true;
        animateConfetti();
    }
}

function animateConfetti() {
    confettiCtx.clearRect(0, 0, confettiCanvas.width, confettiCanvas.height);

    confettiParticles.forEach(p => {
        p.x += p.vx;
        p.vy += p.gravity;
        p.y += p.vy;
        p.vx *= 0.98;
        p.rotation += p.rotationSpeed;
        p.opacity -= p.decay;

        if (p.opacity > 0) {
            confettiCtx.save();
            confettiCtx.translate(p.x, p.y);
            confettiCtx.rotate((p.rotation * Math.PI) / 180);
            confettiCtx.globalAlpha = p.opacity;
            confettiCtx.fillStyle = p.color;
            confettiCtx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
            confettiCtx.restore();
        }
    });

    confettiParticles = confettiParticles.filter(p => p.opacity > 0);

    if (confettiParticles.length > 0) {
        requestAnimationFrame(animateConfetti);
    } else {
        confettiAnimating = false;
        confettiCtx.clearRect(0, 0, confettiCanvas.width, confettiCanvas.height);
    }
}

// ======================================================
//  RENDERING
// ======================================================
function renderBoard() {
    ['todo', 'inprogress', 'done'].forEach(col => {
        const body = document.querySelector(`.column[data-column="${col}"] .column-body`);
        const colTasks = tasks
            .filter(t => t.column === col)
            .sort((a, b) => (a.position ?? 0) - (b.position ?? 0));

        const count = document.querySelector(`.column[data-column="${col}"] .column-count`);
        count.textContent = colTasks.length;

        if (colTasks.length === 0) {
            body.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                    </svg>
                    <p>Drop tasks here</p>
                </div>`;
            return;
        }

        body.innerHTML = colTasks.map(t => renderCard(t)).join('');
        body.querySelectorAll('.task-card').forEach(card => {
            card.addEventListener('dragstart', handleDragStart);
            card.addEventListener('dragend', handleDragEnd);
        });
    });

    updateStats();
    if (analyticsPanel.classList.contains('open')) updateAnalytics();
    applyFilters(); // re-apply any active filters
}

function renderCard(task) {
    const priorityClass = `badge-priority-${task.priority}`;
    const dueInfo = getDueInfo(task.dueDate);
    const subtasksDone = (task.subtasks || []).filter(s => s.completed).length;
    const subtasksTotal = (task.subtasks || []).length;
    const subtaskPct = subtasksTotal > 0 ? (subtasksDone / subtasksTotal) * 100 : 0;

    return `
        <div class="task-card ${task.completed ? 'completed' : ''}"
             data-id="${task.id}" data-priority="${task.priority}" draggable="true">
            <div class="task-top-row">
                ${task.column !== 'done' ? `
                <button class="task-advance-btn" onclick="advanceTask(${task.id})" title="${task.column === 'todo' ? 'Move to In Progress' : 'Mark as Done'}">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </button>` : `
                <button class="task-advance-btn completed" title="Completed" disabled>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                </button>`}
                <div class="task-title">${escapeHtml(task.title)}</div>
                <div class="task-actions">
                    <button class="task-action-btn" onclick="openEditModal(${task.id})" title="Edit">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    <button class="task-action-btn delete" onclick="confirmDelete(${task.id})" title="Delete">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                        </svg>
                    </button>
                </div>
            </div>
            ${task.description ? `<div class="task-description">${escapeHtml(task.description)}</div>` : ''}
            <div class="task-meta">
                <span class="task-badge ${priorityClass}">${task.priority}</span>
                ${task.dueDate ? `
                    <span class="task-due ${dueInfo.class}">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                        </svg>
                        ${dueInfo.label}
                    </span>` : ''}
                ${subtasksTotal > 0 ? `
                    <div class="subtask-progress">
                        <div class="subtask-bar"><div class="subtask-bar-fill" style="width:${subtaskPct}%"></div></div>
                        <span class="subtask-count">${subtasksDone}/${subtasksTotal}</span>
                    </div>` : ''}
            </div>
        </div>`;
}

function getDueInfo(dateStr) {
    if (!dateStr) return { label: '', class: '' };
    const due = new Date(dateStr + 'T23:59:59');
    const now = new Date();
    const diffDays = Math.ceil((due - now) / (1000 * 60 * 60 * 24));
    if (diffDays < 0) return { label: `${Math.abs(diffDays)}d overdue`, class: 'overdue' };
    if (diffDays === 0) return { label: 'Today', class: 'due-soon' };
    if (diffDays === 1) return { label: 'Tomorrow', class: 'due-soon' };
    if (diffDays <= 3) return { label: `${diffDays}d left`, class: 'due-soon' };
    return { label: formatDate(dateStr), class: '' };
}

function formatDate(dateStr) {
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function updateStats() {
    statTodo.textContent = tasks.filter(t => t.column === 'todo').length;
    statProgress.textContent = tasks.filter(t => t.column === 'inprogress').length;
    statDone.textContent = tasks.filter(t => t.column === 'done').length;
}

// ======================================================
//  DRAG & DROP
// ======================================================
function handleDragStart(e) {
    draggedCard = e.target.closest('.task-card');
    draggedCard.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', draggedCard.dataset.id);
}

function handleDragEnd() {
    if (draggedCard) draggedCard.classList.remove('dragging');
    draggedCard = null;
    document.querySelectorAll('.column').forEach(c => c.classList.remove('drag-over'));
    document.querySelectorAll('.drop-placeholder').forEach(p => p.remove());
}

function initColumnDropZones() {
    document.querySelectorAll('.column').forEach(column => {
        const body = column.querySelector('.column-body');

        column.addEventListener('dragover', e => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            column.classList.add('drag-over');
            const afterElement = getDragAfterElement(body, e.clientY);
            let placeholder = body.querySelector('.drop-placeholder');
            if (!placeholder) {
                placeholder = document.createElement('div');
                placeholder.className = 'drop-placeholder';
            }
            if (afterElement) body.insertBefore(placeholder, afterElement);
            else body.appendChild(placeholder);
        });

        column.addEventListener('dragleave', e => {
            if (!column.contains(e.relatedTarget)) {
                column.classList.remove('drag-over');
                const ph = body.querySelector('.drop-placeholder');
                if (ph) ph.remove();
            }
        });

        column.addEventListener('drop', async e => {
            e.preventDefault();
            column.classList.remove('drag-over');
            const ph = body.querySelector('.drop-placeholder');
            const taskId = parseInt(e.dataTransfer.getData('text/plain'));
            const newColumn = column.dataset.column;
            const phIndex = ph ? [...body.children].indexOf(ph) : 0;
            if (ph) ph.remove();

            const task = tasks.find(t => t.id === taskId);
            const wasNotDone = task && task.column !== 'done';

            try {
                const completed = newColumn === 'done';
                await apiPatch(taskId, { column: newColumn, position: phIndex, completed });
                if (task) {
                    task.column = newColumn;
                    task.position = phIndex;
                    task.completed = completed;
                }
                renderBoard();
                showToast('Task moved');

                // Confetti when dragged to Done
                if (newColumn === 'done' && wasNotDone) launchConfetti();
            } catch (err) {
                showToast('Failed to move task', 'error');
            }
        });
    });
}

function getDragAfterElement(container, y) {
    const cards = [...container.querySelectorAll('.task-card:not(.dragging)')];
    return cards.reduce((closest, child) => {
        const box = child.getBoundingClientRect();
        const offset = y - box.top - box.height / 2;
        if (offset < 0 && offset > closest.offset) return { offset, element: child };
        return closest;
    }, { offset: Number.NEGATIVE_INFINITY }).element;
}

// ======================================================
//  MODAL
// ======================================================
let modalSubtasks = [];

function openModal(columnDefault = 'todo') {
    editingTaskId = null;
    modalTitle.textContent = 'New Task';
    formTitle.value = '';
    formDescription.value = '';
    formColumn.value = columnDefault;
    formPriority.value = 'medium';
    formDueDate.value = '';
    modalSubtasks = [];
    renderModalSubtasks();
    btnDelete.style.display = 'none';
    modalOverlay.classList.add('open');
    setTimeout(() => formTitle.focus(), 200);
}

function openEditModal(taskId) {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    editingTaskId = taskId;
    modalTitle.textContent = 'Edit Task';
    formTitle.value = task.title;
    formDescription.value = task.description || '';
    formColumn.value = task.column;
    formPriority.value = task.priority;
    formDueDate.value = task.dueDate || '';
    modalSubtasks = (task.subtasks || []).map(s => ({ ...s }));
    renderModalSubtasks();
    btnDelete.style.display = 'inline-flex';
    modalOverlay.classList.add('open');
}

function closeModal() {
    modalOverlay.classList.remove('open');
    editingTaskId = null;
}

function renderModalSubtasks() {
    subtaskList.innerHTML = modalSubtasks.map((s, i) => `
        <li class="subtask-item">
            <input type="checkbox" class="subtask-checkbox" ${s.completed ? 'checked' : ''} onchange="toggleModalSubtask(${i})">
            <span class="subtask-text ${s.completed ? 'done' : ''}">${escapeHtml(s.title)}</span>
            <button class="subtask-remove" onclick="removeModalSubtask(${i})">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </li>`).join('');
}

function addModalSubtask() {
    const val = subtaskInput.value.trim();
    if (!val) return;
    modalSubtasks.push({ id: Date.now(), title: val, completed: false });
    subtaskInput.value = '';
    renderModalSubtasks();
    subtaskInput.focus();
}
function toggleModalSubtask(index) { modalSubtasks[index].completed = !modalSubtasks[index].completed; renderModalSubtasks(); }
function removeModalSubtask(index) { modalSubtasks.splice(index, 1); renderModalSubtasks(); }

async function saveTask() {
    const title = formTitle.value.trim();
    if (!title) {
        formTitle.style.borderColor = 'var(--accent-red)';
        formTitle.focus();
        setTimeout(() => formTitle.style.borderColor = '', 2000);
        return;
    }

    const taskData = {
        title,
        description: formDescription.value.trim(),
        column: formColumn.value,
        priority: formPriority.value,
        dueDate: formDueDate.value || null,
        subtasks: modalSubtasks,
        completed: formColumn.value === 'done'
    };

    try {
        if (editingTaskId) {
            const oldTask = tasks.find(t => t.id === editingTaskId);
            const wasNotDone = oldTask && oldTask.column !== 'done';
            await apiPut(editingTaskId, taskData);
            const idx = tasks.findIndex(t => t.id === editingTaskId);
            if (idx !== -1) Object.assign(tasks[idx], taskData, { id: editingTaskId });
            showToast('Task updated');
            // Confetti if moved to done via edit
            if (taskData.column === 'done' && wasNotDone) launchConfetti();
        } else {
            const res = await apiPost(taskData);
            tasks.push(res.task);
            showToast('Task created');
            if (taskData.column === 'done') launchConfetti();
        }
        closeModal();
        renderBoard();
    } catch (err) {
        showToast(err.message || 'Failed to save task', 'error');
    }
}

async function confirmDelete(taskId) {
    const id = taskId || editingTaskId;
    if (!id) return;
    if (!confirm('Delete this task?')) return;
    try {
        await apiDelete(id);
        tasks = tasks.filter(t => t.id !== id);
        closeModal();
        renderBoard();
        showToast('Task deleted');
    } catch (err) {
        showToast('Failed to delete task', 'error');
    }
}

// ======================================================
//  ADVANCE TASK
// ======================================================
const nextColumn = { todo: 'inprogress', inprogress: 'done' };

async function advanceTask(taskId) {
    const task = tasks.find(t => t.id === taskId);
    if (!task || !nextColumn[task.column]) return;
    const newCol = nextColumn[task.column];
    const completed = newCol === 'done';

    try {
        await apiPatch(taskId, { column: newCol, completed });
        task.column = newCol;
        task.completed = completed;
        renderBoard();
        const labels = { inprogress: 'In Progress', done: 'Done' };
        showToast(`Moved to ${labels[newCol]}`);
        if (newCol === 'done') launchConfetti();
    } catch (err) {
        showToast(err.message || 'Failed to move task', 'error');
    }
}

// ---- Utilities ----
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ---- Keyboard Shortcuts ----
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeModal();
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') { e.preventDefault(); openModal(); }
    if (e.key === 'Enter' && modalOverlay.classList.contains('open')) {
        if (document.activeElement === subtaskInput) addModalSubtask();
    }
    // Ctrl+/ to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === '/') { e.preventDefault(); searchInput.focus(); }
});

// ---- Init ----
async function init() {
    loadTheme();
    try {
        tasks = await apiGet();
        renderBoard();
        initColumnDropZones();
    } catch (err) {
        showToast('Failed to load tasks', 'error');
        console.error(err);
    }
}

// Expose functions globally
window.advanceTask = advanceTask;
window.openModal = openModal;
window.openEditModal = openEditModal;
window.closeModal = closeModal;
window.saveTask = saveTask;
window.confirmDelete = confirmDelete;
window.addModalSubtask = addModalSubtask;
window.toggleModalSubtask = toggleModalSubtask;
window.removeModalSubtask = removeModalSubtask;
window.toggleTheme = toggleTheme;
window.toggleAnalytics = toggleAnalytics;
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;

init();
