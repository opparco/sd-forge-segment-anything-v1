(function () {
    const state = {
        points: [],
        labels: [],
        generateTimer: null,
    };

    function findInput(componentId) {
        const root = gradioApp().getElementById(componentId);
        if (!root) return null;
        return root.querySelector("textarea, input");
    }

    function setInputValue(componentId, value) {
        const input = findInput(componentId);
        if (!input) return false;
        input.value = value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }

    function readStateFromInputs() {
        try {
            state.points = JSON.parse(findInput("segment_anything_input_points")?.value || "[]");
            state.labels = JSON.parse(findInput("segment_anything_input_labels")?.value || "[]");
        } catch (_e) {
            state.points = [];
            state.labels = [];
        }
    }

    function writeStateToInputs() {
        setInputValue("segment_anything_input_points", JSON.stringify(state.points));
        setInputValue("segment_anything_input_labels", JSON.stringify(state.labels));
    }

    function scheduleGenerate() {
        clearTimeout(state.generateTimer);
        state.generateTimer = setTimeout(() => {
            const button = gradioApp().getElementById("segment_anything_auto_generate_mask");
            button?.click();
        }, 150);
    }

    function getImageElement(root) {
        return root.querySelector("img") || root.querySelector("canvas");
    }

    function getComponentImageSrc(componentId) {
        const root = gradioApp().getElementById(componentId);
        const image = root?.querySelector("img");
        return image?.src || null;
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            if (!src) {
                reject(new Error("Missing image source."));
                return;
            }
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error("Could not load image source."));
            image.src = src;
        });
    }

    function imageToPngDataUrl(image) {
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth || image.width;
        canvas.height = image.naturalHeight || image.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/png");
    }

    async function maskToForegroundDataUrl(maskSrc, width, height) {
        const maskImage = await loadImage(maskSrc);
        const sourceCanvas = document.createElement("canvas");
        sourceCanvas.width = width;
        sourceCanvas.height = height;
        const sourceCtx = sourceCanvas.getContext("2d");
        sourceCtx.drawImage(maskImage, 0, 0, width, height);
        const sourceData = sourceCtx.getImageData(0, 0, width, height);

        const foregroundCanvas = document.createElement("canvas");
        foregroundCanvas.width = width;
        foregroundCanvas.height = height;
        const foregroundCtx = foregroundCanvas.getContext("2d");
        const foregroundData = foregroundCtx.createImageData(width, height);

        for (let i = 0; i < sourceData.data.length; i += 4) {
            const alpha = Math.max(sourceData.data[i], sourceData.data[i + 1], sourceData.data[i + 2]);
            foregroundData.data[i] = 255;
            foregroundData.data[i + 1] = 255;
            foregroundData.data[i + 2] = 255;
            foregroundData.data[i + 3] = alpha;
        }

        foregroundCtx.putImageData(foregroundData, 0, 0);
        return foregroundCanvas.toDataURL("image/png");
    }

    function findForgeCanvasUuid(elemId) {
        const root = gradioApp().getElementById(elemId);
        const container = root?.querySelector('[id^="container_uuid_"]');
        return container?.id.replace("container_", "") || null;
    }

    function setForgeLogicalImage(uuid, className, dataUrl) {
        const selector = `#${CSS.escape(uuid)}.${className} textarea`;
        const textarea = gradioApp().querySelector(selector);
        if (!textarea) return false;
        textarea.value = dataUrl || "";
        updateInput(textarea);
        textarea.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    }

    async function exportToInpaint() {
        try {
            const refImage = await loadImage(getComponentImageSrc("segment_anything_ref_image"));
            const maskSrc = getComponentImageSrc("segment_anything_mask");
            if (!maskSrc) throw new Error("Generate a mask before exporting.");

            switch_to_inpaint();
            const uuid = findForgeCanvasUuid("img2maskimg");
            if (!uuid) throw new Error("Could not find ForgeCanvas inpaint target.");

            const width = refImage.naturalWidth || refImage.width;
            const height = refImage.naturalHeight || refImage.height;
            const backgroundDataUrl = imageToPngDataUrl(refImage);
            const foregroundDataUrl = await maskToForegroundDataUrl(maskSrc, width, height);

            if (!setForgeLogicalImage(uuid, "logical_image_background", backgroundDataUrl)) {
                throw new Error("Could not set ForgeCanvas background.");
            }
            if (!setForgeLogicalImage(uuid, "logical_image_foreground", foregroundDataUrl)) {
                throw new Error("Could not set ForgeCanvas foreground.");
            }

            setTimeout(() => gradioApp().getElementById("img2img_detect_image_size_btn")?.click(), 500);
        } catch (error) {
            console.error("Segment Anything v1 export failed:", error);
            alert(`Segment Anything v1 export failed: ${error.message}`);
        }
    }

    function naturalSize(el) {
        if (el instanceof HTMLImageElement) {
            return [el.naturalWidth || el.width, el.naturalHeight || el.height];
        }
        if (el instanceof HTMLCanvasElement) {
            return [el.width, el.height];
        }
        return [0, 0];
    }

    function pointFromEvent(event, el) {
        const rect = el.getBoundingClientRect();
        const [naturalWidth, naturalHeight] = naturalSize(el);
        if (!rect.width || !rect.height || !naturalWidth || !naturalHeight) return null;
        const x = Math.round(((event.clientX - rect.left) / rect.width) * naturalWidth);
        const y = Math.round(((event.clientY - rect.top) / rect.height) * naturalHeight);
        return [
            Math.max(0, Math.min(naturalWidth - 1, x)),
            Math.max(0, Math.min(naturalHeight - 1, y)),
        ];
    }

    function ensureOverlay(root, imageEl) {
        let overlay = root.querySelector(".segment-anything-point-overlay");
        if (!overlay) {
            overlay = document.createElement("div");
            overlay.className = "segment-anything-point-overlay";
            overlay.style.pointerEvents = "none";
            root.appendChild(overlay);
        }

        const rootRect = root.getBoundingClientRect();
        const imageRect = imageEl.getBoundingClientRect();
        overlay.style.left = `${imageRect.left - rootRect.left}px`;
        overlay.style.top = `${imageRect.top - rootRect.top}px`;
        overlay.style.width = `${imageRect.width}px`;
        overlay.style.height = `${imageRect.height}px`;
        return overlay;
    }

    function renderPoints() {
        const root = gradioApp().getElementById("segment_anything_ref_image");
        if (!root) return;
        const imageEl = getImageElement(root);
        if (!imageEl) return;
        const overlay = ensureOverlay(root, imageEl);
        const [naturalWidth, naturalHeight] = naturalSize(imageEl);
        overlay.replaceChildren();
        if (!naturalWidth || !naturalHeight) return;

        state.points.forEach(([x, y], index) => {
            const label = state.labels[index];
            const dot = document.createElement("div");
            dot.className = `segment-anything-point ${label === 1 ? "segment-anything-point-fg" : "segment-anything-point-bg"}`;
            dot.style.left = `${(x / naturalWidth) * 100}%`;
            dot.style.top = `${(y / naturalHeight) * 100}%`;
            overlay.appendChild(dot);
        });
    }

    function attachSamPointHandler() {
        const root = gradioApp().getElementById("segment_anything_ref_image");
        if (!root || root.dataset.samPointHandler === "true") return;

        root.dataset.samPointHandler = "true";
        root.style.position = "relative";

        root.addEventListener("contextmenu", (event) => {
            if (getImageElement(root)) event.preventDefault();
        });

        root.addEventListener("mousedown", (event) => {
            if (event.button !== 0 && event.button !== 2) return;
            const imageEl = getImageElement(root);
            if (!imageEl) return;
            const point = pointFromEvent(event, imageEl);
            if (!point) return;

            event.preventDefault();
            readStateFromInputs();
            state.points.push(point);
            state.labels.push(event.button === 0 ? 1 : 0);
            writeStateToInputs();
            renderPoints();
            scheduleGenerate();
        });
    }

    function attachClearHandler() {
        const clearButton = gradioApp().getElementById("segment_anything_clear_points");
        if (!clearButton || clearButton.dataset.samClearHandler === "true") return;
        clearButton.dataset.samClearHandler = "true";
        clearButton.addEventListener("click", () => {
            state.points = [];
            state.labels = [];
            renderPoints();
        });
    }

    function attachExportHandler() {
        const exportButton = gradioApp().getElementById("segment_anything_export_inpaint");
        if (!exportButton || exportButton.dataset.segmentAnythingExportHandler === "true") return;
        exportButton.dataset.segmentAnythingExportHandler = "true";
        exportButton.addEventListener("click", exportToInpaint);
    }

    function initSamMasker() {
        attachSamPointHandler();
        attachClearHandler();
        attachExportHandler();
        readStateFromInputs();
        renderPoints();
    }

    onUiLoaded(initSamMasker);
    onAfterUiUpdate(initSamMasker);
})();
