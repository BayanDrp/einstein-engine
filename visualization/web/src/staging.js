
import * as THREE from "three";
import { createBody } from "./bodies.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";


// Own the scene, camera, stars, HUD and the playback clock  main.js only has to say "load this data" and then "advance frame i", while staging does everything visual per frame.







export function createStage(canvas, options = {}) {

    // 1. Scene
    const scene = new THREE.Scene();

    scene.background = new THREE.Color(
        options.backgroundColor ?? 0x000000
    );

    // 2. Camera
    const width = options.width ?? 1280;
    const height = options.height ?? 720;

    const cameraOptions = options.camera ?? {};
    const fov = cameraOptions.fov ?? 50;

    const camera = new THREE.PerspectiveCamera(
        fov,
        width / height,
        cameraOptions.near ?? 0.1,
        cameraOptions.far ?? 1000
    );

    // Camera position using spherical coordinates. The distance is fitted
    // to the data after load(); 14 is only the pre-load placeholder.
    let viewRadius = 14;

    // default view: almost straight at the orbital plane, with a slight
    // elevation so the scene still reads as 3D
    const phi =
        (options.phi ?? 80) * Math.PI / 180;

    const theta =
        (options.theta ?? 0) * Math.PI / 180;

    function placeCamera(drift = 0) {
        camera.position.set(
            viewRadius * Math.sin(phi) * Math.sin(theta + drift),
            viewRadius * Math.cos(phi),
            viewRadius * Math.sin(phi) * Math.cos(theta + drift)
        );
        camera.lookAt(0, 0, 0);
    }

    placeCamera();

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true
    });

    renderer.setPixelRatio(1);
    renderer.setSize(width, height);

    // 3b. Bloom: skipped entirely with options.bloom === false
    let composer = null;
    if (options.bloom !== false) {
        composer = new EffectComposer(renderer);
        composer.addPass(new RenderPass(scene, camera));
        composer.addPass(new UnrealBloomPass(
            new THREE.Vector2(width, height),
            0.5, 0.4, 0.9 // strength, radius, threshold
        ));
        // convert the linear composer buffer back to sRGB; without this the
        // bloom path and the plain renderer path disagree about color
        composer.addPass(new OutputPass());
    }

    // 3c. Starfield: seeded, so every render shows the same stars
    function addStars(count) {
        const positions = new Float32Array(count * 3);
        let seed = 1234567;
        const random = () => {
            seed = (seed * 48271) % 2147483647;
            return seed / 2147483647;
        };
        for (let i = 0; i < count; i++) {
            const angle = random() * Math.PI * 2;
            const up = random() * 2 - 1;
            const flat = Math.sqrt(1 - up * up);
            positions[i * 3] = 40 * flat * Math.cos(angle);
            positions[i * 3 + 1] = 40 * up;
            positions[i * 3 + 2] = 40 * flat * Math.sin(angle);
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        const stars = new THREE.Points(geometry, new THREE.PointsMaterial({
            color: 0x8a9bb8,
            size: 0.06,
            transparent: true,
            opacity: 0.8,
        }));
        scene.add(stars);
    }
    addStars(900);

    // 4. Playback clock
    const fps = options.fps ?? 30;
    const seconds = options.seconds ?? 20;
    const totalFrames = Math.max(1, Math.round(seconds * fps));

    let playbackSpan = null; // the global time window, stored by load()
    let step = 0;            // engine-time advance per frame
    let finished = false;    // set true once frameIndex passes the last frame

    // 5. Bodies
    const bodies = [];

    // 5b. HUD: title, legend, time readout, plain DOM above the canvas.
    // Type and margins scale with render height so 4K stays readable.
    const unit = Math.max(10, Math.round(height / 60));
    const pad = Math.round(unit * 1.2);
    const hud = document.createElement("div");
    hud.style.cssText =
        "position:fixed;inset:0;pointer-events:none;" +
        `font-family:monospace;font-size:${unit}px;color:#c8cfdd;` +
        "text-shadow:0 0 4px #000;";
    const titleBox = document.createElement("div");
    titleBox.style.cssText = `position:absolute;top:${pad}px;left:${pad}px;`;
    titleBox.textContent = options.title ?? "";
    const timeBox = document.createElement("div");
    timeBox.style.cssText = `position:absolute;top:${pad}px;right:${pad}px;`;
    const legendBox = document.createElement("div");
    legendBox.style.cssText = `position:absolute;bottom:${pad}px;left:${pad}px;line-height:1.7;`;
    hud.append(titleBox, timeBox, legendBox);
    document.body.append(hud);

    function load(tracks, frame, span) {

        // Remove old bodies from the scene
        for (const body of bodies) {
            scene.remove(body.group);
        }

        // Clear the old references
        bodies.length = 0;

        // trail length in engine time: a small slice of the recording
        const trailWindow = span ? (span.end - span.start) * 0.12 : Infinity;

        // Create a visual body for each track
        for (const [index, track] of tracks.entries()) {

            const body = createBody(track, index, frame, {
                trails: options.trails !== false,
                trailWindow,
            });

            // Skip invisible tracks
            if (!body) continue;

            // Store the controller
            bodies.push(body);

            // Add its Three.js group to the scene
            scene.add(body.group);
        }

        // legend row per visible body: colored dot (ring for dark bodies) + name
        legendBox.textContent = "";
        for (const body of bodies) {
            const row = document.createElement("div");
            const dot = document.createElement("span");
            const dark = body.track.color === "grey" || body.track.color === "black";
            dot.textContent = dark ? "∘ " : "● ";
            dot.style.color = body.color;
            row.append(dot, body.track.name);
            legendBox.append(row);
        }

        // the clock runs over the span spanOf() produced
        playbackSpan = span;
        step = span ? (span.end - span.start) / totalFrames : 0;

        // fit the camera to the data's bounding sphere, not a fixed radius
        viewRadius = 14;
        if (frame.radius > 0) {
            const halfFov = fov * Math.PI / 360;
            viewRadius = frame.radius / Math.sin(halfFov) * 1.2;
        }
        placeCamera();
    }

    // alpha 1..0 for a body whose recording ends before the span does;
    // single-sample markers hold at full strength
    function alphaFor(track, t) {
        if (!playbackSpan || track.times.length < 2) return 1;
        const last = track.times[track.times.length - 1];
        if (t <= last) return 1;
        const rest = playbackSpan.end - last;
        return rest > 0 ? Math.max(1 - (t - last) / rest, 0) : 1;
    }

    function updateBodies(time) {
        for (const body of bodies) {
            body.update(time, alphaFor(body.track, time));
        }
    }

    function render() {
        if (composer) composer.render();
        else renderer.render(scene, camera);
    }

    // advance the stage by one frame: the only time source is the frame
    // number, so every render of frame i is pixel-identical
    function update(frameIndex) {
        const t = playbackSpan ? playbackSpan.start + frameIndex * step : 0;
        updateBodies(t);

        // slow orbit: a fixed 15 degree sweep over the clip, driven by the
        // frame counter, never by engine time
        const progress = totalFrames > 1
            ? Math.min(frameIndex / (totalFrames - 1), 1)
            : 0;
        const drift = -15 * progress * Math.PI / 180;
        placeCamera(drift);

        let label = "t = " + t.toFixed(1);
        if (playbackSpan) {
            const compression = (playbackSpan.end - playbackSpan.start) / seconds;
            label += " / " + playbackSpan.end.toFixed(1) +
                " · playback ×" + compression.toFixed(1);
        }
        timeBox.textContent = label;

        if (frameIndex >= totalFrames) {
            finished = true;
        }
    }

    
    

    // 6. Public API
    return {
        scene,
        camera,
        renderer,
        load,
        updateBodies,
        update,
        render,
        totalFrames,
        // read as a flag: true once the clock passed its last frame
        get done() {
            return finished;
        },
    };
}

