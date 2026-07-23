// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const filterType = document.getElementById('filterType');
const filterSize = document.getElementById('filterSize');
const sizeDisplay = document.getElementById('sizeDisplay');
const processBtn = document.getElementById('processBtn');
const filterInfo = document.getElementById('filterInfo');

const originalCanvas = document.getElementById('originalCanvas');
const spectrumCanvas = document.getElementById('spectrumCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const filteredCanvas = document.getElementById('filteredCanvas');

let currentImage = null;

// Filter descriptions
const filterDescriptions = {
    lowpass: 'מסנן Low-Pass מעביר תדרים נמוכים ומסנן תדרים גבוהים, יוצר אפקט טשטוש.',
    highpass: 'מסנן High-Pass מעביר תדרים גבוהים ומסנן תדרים נמוכים, מדגיש קצוות ופרטים.',
    bandpass: 'מסנן Band-Pass מעביר טווח ספציפי של תדרים, יוצר אפקט טבעת.'
};

// Event Listeners
imageUpload.addEventListener('change', loadImage);
filterSize.addEventListener('input', () => {
    sizeDisplay.textContent = filterSize.value;
});
filterType.addEventListener('change', () => {
    filterInfo.textContent = filterDescriptions[filterType.value];
});
processBtn.addEventListener('click', processImage);

// Initialize
filterInfo.textContent = filterDescriptions.lowpass;

function loadImage(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (event) {
        const img = new Image();
        img.onload = function () {
            currentImage = img;
            drawOriginal();
            processBtn.disabled = false;
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function drawOriginal() {
    const ctx = originalCanvas.getContext('2d');
    const size = 256;
    originalCanvas.width = size;
    originalCanvas.height = size;

    ctx.drawImage(currentImage, 0, 0, size, size);
}

function processImage() {
    if (!currentImage) {
        alert('אנא העלה תמונה תחילה');
        return;
    }

    const size = 256;
    const ctx = originalCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, size, size);

    // Convert to grayscale
    const grayscale = new Float32Array(size * size);
    for (let i = 0; i < imageData.data.length; i += 4) {
        const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
        grayscale[i / 4] = gray;
    }

    // Apply FFT (simplified - using basic approach)
    const spectrum = applySimplifiedFFT(grayscale, size);

    // Draw spectrum
    drawSpectrum(spectrum, size);

    // Create mask
    const mask = createMask(size, filterType.value, parseInt(filterSize.value));
    drawMask(mask, size);

    // Apply mask
    const filtered = applyMask(spectrum, mask, size);

    // Draw filtered image
    drawFiltered(filtered, size);
}

function applySimplifiedFFT(data, size) {
    // Simplified FFT visualization (not real FFT, just for demo)
    const spectrum = new Float32Array(size * size);
    const centerX = size / 2;
    const centerY = size / 2;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - centerX;
            const dy = y - centerY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            // Simulate frequency domain
            let value = 0;
            for (let ky = 0; ky < size; ky += 4) {
                for (let kx = 0; kx < size; kx += 4) {
                    const angle = 2 * Math.PI * (dx * kx + dy * ky) / size;
                    value += data[ky * size + kx] * Math.cos(angle);
                }
            }

            spectrum[y * size + x] = Math.abs(value);
        }
    }

    return spectrum;
}

function drawSpectrum(spectrum, size) {
    spectrumCanvas.width = size;
    spectrumCanvas.height = size;
    const ctx = spectrumCanvas.getContext('2d');
    const imageData = ctx.createImageData(size, size);

    // Find max for normalization
    let max = 0;
    for (let i = 0; i < spectrum.length; i++) {
        if (spectrum[i] > max) max = spectrum[i];
    }

    // Draw spectrum (log scale)
    for (let i = 0; i < spectrum.length; i++) {
        const value = Math.log(1 + spectrum[i]) / Math.log(1 + max) * 255;
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}

function createMask(size, type, radius) {
    const mask = new Float32Array(size * size);
    const centerX = size / 2;
    const centerY = size / 2;

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const dx = x - centerX;
            const dy = y - centerY;
            const dist = Math.sqrt(dx * dx + dy * dy);

            let value = 0;

            switch (type) {
                case 'lowpass':
                    value = dist <= radius ? 1 : 0;
                    break;
                case 'highpass':
                    value = dist > radius ? 1 : 0;
                    break;
                case 'bandpass':
                    const innerRadius = radius - 20;
                    const outerRadius = radius + 20;
                    value = (dist >= innerRadius && dist <= outerRadius) ? 1 : 0;
                    break;
            }

            mask[y * size + x] = value;
        }
    }

    return mask;
}

function drawMask(mask, size) {
    maskCanvas.width = size;
    maskCanvas.height = size;
    const ctx = maskCanvas.getContext('2d');
    const imageData = ctx.createImageData(size, size);

    for (let i = 0; i < mask.length; i++) {
        const value = mask[i] * 255;
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}

function applyMask(spectrum, mask, size) {
    const filtered = new Float32Array(size * size);

    for (let i = 0; i < spectrum.length; i++) {
        filtered[i] = spectrum[i] * mask[i];
    }

    return filtered;
}

function drawFiltered(filtered, size) {
    filteredCanvas.width = size;
    filteredCanvas.height = size;
    const ctx = filteredCanvas.getContext('2d');
    const imageData = ctx.createImageData(size, size);

    // Inverse FFT (simplified - just normalize and display)
    let max = 0;
    for (let i = 0; i < filtered.length; i++) {
        if (filtered[i] > max) max = filtered[i];
    }

    for (let i = 0; i < filtered.length; i++) {
        const value = (filtered[i] / max) * 255;
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
}
