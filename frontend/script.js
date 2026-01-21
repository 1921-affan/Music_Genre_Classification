const API_BASE = '';
let socket;

// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const progressSection = document.getElementById('progressSection');
const resultCard = document.getElementById('resultCard');
const progressBar = document.getElementById('progressBar');
const statusMessage = document.getElementById('statusMessage');
const fileNameDisplay = document.getElementById('fileName');
const fileSizeDisplay = document.getElementById('fileSize');
const genreResult = document.getElementById('genreResult');
const confidenceBadge = document.getElementById('confidenceBadge');

// Step Mapping
const steps = ['upload', 'preprocessing', 'feature_extraction', 'inference', 'complete'];
const stepIndices = {
    'upload': 1,
    'preprocessing': 2,
    'feature_extraction': 3,
    'inference': 4,
    'complete': 4
};

// Initialize Socket.io
function initSocket() {
    socket = io(API_BASE);

    socket.on('connect', () => {
        console.log('Connected to analysis server');
    });

    socket.on('progress', (data) => {
        console.log('Progress Update:', data);
        updateUIProgress(data.step, data.message);
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });
}

// UI Controllers
function updateUIProgress(stepId, message) {
    const index = stepIndices[stepId];
    if (!index) return;

    // Update Progress Bar
    const progress = (index / 4) * 100;
    progressBar.style.width = `${progress}%`;
    statusMessage.innerText = message;

    // Update Steps UI
    document.querySelectorAll('.step').forEach(stepEl => {
        const stepName = stepEl.dataset.step;
        const stepIndex = stepIndices[stepName];

        if (stepIndex < index) {
            stepEl.classList.remove('active');
            stepEl.classList.add('completed');
        } else if (stepIndex === index) {
            stepEl.classList.add('active');
            stepEl.classList.remove('completed');
        } else {
            stepEl.classList.remove('active', 'completed');
        }
    });

    if (stepId === 'complete') {
        progressBar.style.width = '100%';
        progressBar.style.backgroundColor = '#10b981';
    }
}

function showResult(data) {
    progressSection.classList.add('hidden');
    resultCard.classList.remove('hidden');

    genreResult.innerText = data.predicted_class;
    confidenceBadge.innerText = `${data.confidence}% Confidence`;

    // Dynamic color based on confidence
    if (data.confidence > 80) {
        confidenceBadge.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
        confidenceBadge.style.color = '#10b981';
    } else {
        confidenceBadge.style.backgroundColor = 'rgba(234, 179, 8, 0.1)';
        confidenceBadge.style.color = '#eab308';
    }

    // Grad-CAM Display
    const explainabilitySection = document.getElementById('explainabilitySection');
    const gradcamImage = document.getElementById('gradcamImage');

    if (data.explainability) {
        explainabilitySection.classList.remove('hidden');
        gradcamImage.src = `data:image/png;base64,${data.explainability}`;
    } else {
        explainabilitySection.classList.add('hidden');
    }
}

function resetUI() {
    resultCard.classList.add('hidden');
    progressSection.classList.add('hidden');
    dropzone.classList.remove('hidden');
    progressBar.style.width = '0%';
    progressBar.style.backgroundColor = '#a78bfa';
    fileInput.value = '';

    document.querySelectorAll('.step').forEach(el => {
        el.classList.remove('active', 'completed');
    });
}

// File Upload Logic
async function handleFileUpload(file) {
    if (!file) return;

    // Show Progress State
    dropzone.classList.add('hidden');
    progressSection.classList.remove('hidden');
    fileNameDisplay.innerText = file.name;
    fileSizeDisplay.innerText = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;

    updateUIProgress('upload', 'Uploading file to server...');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.status === 'success') {
            showResult(data);
        } else {
            alert(`Error: ${data.message}`);
            resetUI();
        }
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Could not connect to the backend server. Make sure Python app.py is running.');
        resetUI();
    }
}

// Event Listeners
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
        handleFileUpload(file);
    } else {
        alert('Please upload a valid audio file.');
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

// Start
initSocket();
