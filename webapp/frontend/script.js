const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.getElementById('upload-section');
const loadingState = document.getElementById('loading-state');
const resultsSection = document.getElementById('results-section');
const imgBefore = document.getElementById('img-before');
const imgAfter = document.getElementById('img-after');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');

// --- Event Listeners ---

// Click to upload
dropZone.addEventListener('click', () => fileInput.click());

// Drag & Drop effects
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// File Input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

// Reset
resetBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    fileInput.value = ''; // clear input
});

// Download
downloadBtn.addEventListener('click', () => {
    const link = document.createElement('a');
    link.href = imgAfter.src;
    link.download = 'enhanced_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// --- Logic ---

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    // Show preview of original
    const reader = new FileReader();
    reader.onload = (e) => {
        imgBefore.src = e.target.result;
    };
    reader.readAsDataURL(file);

    uploadImage(file);
}

async function uploadImage(file) {
    // UI State: Loading
    uploadSection.classList.add('hidden');
    loadingState.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const blob = await response.blob();
        const objectURL = URL.createObjectURL(blob);

        // UI State: Results
        imgAfter.src = objectURL;

        // Wait for image to load to avoid layout shifts? (Optional, browser handles it okay)
        imgAfter.onload = () => {
            loadingState.classList.add('hidden');
            resultsSection.classList.remove('hidden');
        };

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to enhance image. Is the backend server running?');
        // Reset state
        loadingState.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }
}
