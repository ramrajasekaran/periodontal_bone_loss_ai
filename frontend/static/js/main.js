// ==============================
// Periodontal Bone Loss AI
// Frontend Logic
// ==============================

const uploadArea = document.getElementById('upload-area');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewImage = document.getElementById('preview-image');
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');

let selectedFile = null;

// ---- Upload Handling ----

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG or PNG).');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        uploadArea.classList.add('has-image');
        analyzeBtn.disabled = false;
        clearBtn.classList.remove('hidden');
        hideError();
        resultsSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewImage.classList.add('hidden');
    uploadPlaceholder.classList.remove('hidden');
    uploadArea.classList.remove('has-image');
    analyzeBtn.disabled = true;
    clearBtn.classList.add('hidden');
    resultsSection.classList.add('hidden');
    hideError();
}

// ---- Analysis ----

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');

    // Show loading state
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    analyzeBtn.disabled = true;
    hideError();
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', document.getElementById('model-select').value);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'An unexpected error occurred.');
            return;
        }

        displayResults(data);
    } catch (err) {
        showError('Failed to connect to the server. Please ensure the backend is running.');
    } finally {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
});

// ---- Display Results ----

function displayResults(data) {
    const gradeEl = document.getElementById('prediction-grade');
    const modelEl = document.getElementById('prediction-model');
    const barsEl = document.getElementById('confidence-bars');
    const gradcamSection = document.getElementById('gradcam-section');

    // Prediction grade
    gradeEl.textContent = data.prediction;
    gradeEl.className = 'prediction-grade grade-' + data.grade;

    // Model used
    const modelNames = { 'efficientnet': 'EfficientNet-B0', 'resnet50': 'ResNet-50', 'cnn': 'Custom CNN' };
    modelEl.textContent = 'Model: ' + (modelNames[data.model_used] || data.model_used);

    // Confidence bars
    barsEl.innerHTML = '';
    const grades = ['Grade 1', 'Grade 2', 'Grade 3'];
    grades.forEach((grade, i) => {
        const pct = data.confidence[grade];
        const row = document.createElement('div');
        row.className = 'confidence-row';
        row.innerHTML = `
            <span class="confidence-label">${grade}</span>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill grade-${i + 1}" style="width: 0%"></div>
            </div>
            <span class="confidence-value">${pct.toFixed(1)}%</span>
        `;
        barsEl.appendChild(row);

        // Animate bar after a small delay
        requestAnimationFrame(() => {
            setTimeout(() => {
                row.querySelector('.confidence-bar-fill').style.width = pct + '%';
            }, 100 + i * 80);
        });
    });

    // Grad-CAM images
    if (data.gradcam && !data.gradcam.error) {
        document.getElementById('gradcam-original').src = 'data:image/png;base64,' + data.gradcam.original;
        document.getElementById('gradcam-heatmap').src = 'data:image/png;base64,' + data.gradcam.heatmap;
        document.getElementById('gradcam-overlay').src = 'data:image/png;base64,' + data.gradcam.overlay;
        gradcamSection.classList.remove('hidden');
    } else {
        gradcamSection.classList.add('hidden');
    }

    resultsSection.classList.remove('hidden');

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Error Handling ----

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}
