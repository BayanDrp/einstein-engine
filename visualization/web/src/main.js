import { parseCsv } from "./ingest.js";
import { fitToFrame, spanOf } from "./layout.js";
import { createStage } from "./staging.js";

// 1. initialize stages and Canvas
// 2. Discover files in the data folder
// 3. Load the data 
// 4. Prepare simulation
// 5. Load scene
// 6. Playback



function init(options, canvas = document.getElementById("scene")) {
    if (!canvas) {
        throw new Error("Canvas element #scene was not found");
    }

    const stage = createStage(canvas, options);

    return stage;
}

async function discoverFiles(directory) {
    const response = await fetch(
        `/list?f=${encodeURIComponent(directory)}`
    );

    if (!response.ok) {
        throw new Error(`Failed to fetch file list: ${response.statusText}`);
    }

    const data = await response.json();

    return data.files;
}

async function loadDataAndTurnToTracks(files, directory) {
    const results = files.map(async (file) => {
        const url = `${directory}/${file}`;

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
        }

        const csvText = await response.text();

        return parseCsv(csvText, file);
    });

    const parsed = await Promise.all(results);
    const tracks = [];

    for (let i = 0; i < parsed.length; i++) {
        const result = parsed[i];

        if (result.skipped) {
            console.warn(`${files[i]}: ${result.skipped}`);
            continue;
        }

        for (const track of result.tracks) {
            tracks.push(track);
        }
    }

    return tracks;
}

function setupScene(stage, tracks) {
    const frame = fitToFrame(tracks);
    const span = spanOf(tracks);

    stage.load(tracks, frame, span);
}

function playback(stage) {
    let frameIndex = 0;

    function animate() {
        if (stage.done) return;

        stage.update(frameIndex);
        stage.render();

        frameIndex++;

        requestAnimationFrame(animate);
    }

    animate();
}

const params = new URLSearchParams(location.search);
const directory = params.get("data") ?? "data/schwarzschild";

const stage = init({
    title: directory,
    seconds: Number(params.get("seconds")) || 20,
    fps: Number(params.get("fps")) || 30,
    width: Number(params.get("width")) || 1280,
    height: Number(params.get("height")) || 720,
    bloom: !params.has("no-bloom"),
    trails: !params.has("no-trails"),
});

async function start() {
    const files = await discoverFiles(directory);
    const tracks = await loadDataAndTurnToTracks(files, directory);

    if (tracks.length === 0) {
        throw new Error(`No tracks found in ${directory}`);
    }

    setupScene(stage, tracks);

    // capture mode: the render driver calls update/render frame by frame
    if (params.has("capture")) {
        window.stage = stage;
        window.ready = true;
        return;
    }

    playback(stage);
}

start().catch((error) => console.error(error));