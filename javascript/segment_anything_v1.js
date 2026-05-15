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

    function initSamMasker() {
        attachSamPointHandler();
        attachClearHandler();
        readStateFromInputs();
        renderPoints();
    }

    onUiLoaded(initSamMasker);
    onAfterUiUpdate(initSamMasker);
})();
