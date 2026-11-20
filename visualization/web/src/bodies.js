import * as THREE from "three";

import { sampleAt, sampleIndexAt } from "./layout.js";

const TRAIL_CAPACITY = 480;

const PALETTE = [
    "#ffd35c",
    "#9fdcff",
    "#ffc2d1",
    "#a5ffd6",
    "#d9c2ff",
    "#f4f8ff",
];

function colorFor(index) {
    return PALETTE[index % PALETTE.length];
}

function createSphere(color, radius) {
    const geometry = new THREE.SphereGeometry(radius, 32, 24);

    const material = new THREE.MeshBasicMaterial({
        color: color,
    });

    const mesh = new THREE.Mesh(geometry, material);

    return mesh;
}

let halo = null; // the shared glow texture, drawn once on first use

function getHalo() {
    if (!halo) {
        const canvas = document.createElement("canvas");
        canvas.width = 64;
        canvas.height = 64;
        const context = canvas.getContext("2d");
        const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
        gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
        gradient.addColorStop(0.35, "rgba(255, 255, 255, 0.75)");
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
        context.fillStyle = gradient;
        context.fillRect(0, 0, 64, 64);
        halo = new THREE.CanvasTexture(canvas);
    }
    return halo;
}

class Track_controller {
    constructor(track, index, frame, options = {}) {
        this.track = track; // the ingested data; sampleAt reads it
        this.color = colorFor(index);
        this.frame = frame; // center + scale; engine sizes are scaled like positions
        this.group = new THREE.Group(); // the visual root that staging adds
        this.group.name = track.name;
        this.trail = null;

        // a grey or black color hint means the body eats light (the engine
        // writes grey for the black hole), so this check comes first
        if (track.color === "grey" || track.color === "black") {
            this.buildDark();
        } else if (track.kind === "matter") {
            this.buildMatter(this.color);
        } else {
            this.buildParticle(this.color);
        }

        // trails show recorded samples only; single markers have no path
        if (options.trails !== false && track.times.length > 1) {
            this.buildTrail(options.trailWindow ?? Infinity);
        }
    }

    // particle: one additive sprite; the round glow texture is bright at the
    // center and fades out, so the dot reads even without bloom
    buildParticle(color) {
        const material = new THREE.SpriteMaterial({
            map: getHalo(),
            color: color,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(0.14, 0.14, 1);
        this.group.add(sprite);
    }

    // matter: one solid sphere, sized by the CSV hint in engine units
    buildMatter(color) {
        const radius = (this.track.displaySize ?? 0.2) * this.frame.scale;
        this.group.add(createSphere(color, radius));
    }

    // dark: black core plus one thin luminous ring just outside it; the ring
    // is what makes a black body readable against the others' glow
    buildDark() {
        const radius = (this.track.displaySize ?? 0.38) * this.frame.scale;
        const core = createSphere("#000000", radius);
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(radius * 1.16, radius * 0.05, 16, 64),
            new THREE.MeshBasicMaterial({ color: "#ffc98a" }),
        );
        this.group.add(core);
        this.group.add(ring);
    }

    // a fading line of recorded past samples: black at the tail, the body
    // color at the head. Positions are stored relative to the group so the
    // body can move the whole trail with one transform.
    buildTrail(window) {
        const source = this.track.positions;
        const world = new Float32Array(source.length);
        for (let i = 0; i < source.length; i += 3) {
            world[i] = (source[i] - this.frame.center[0]) * this.frame.scale;
            world[i + 1] = (source[i + 1] - this.frame.center[1]) * this.frame.scale;
            world[i + 2] = (source[i + 2] - this.frame.center[2]) * this.frame.scale;
        }

        this.trailTimes = this.track.times;
        this.trailWorld = world;
        this.trailWindow = window;
        this.trailCapacity = Math.min(this.trailTimes.length, TRAIL_CAPACITY);
        this.trailPositions = new Float32Array(this.trailCapacity * 3);
        this.trailColors = new Float32Array(this.trailCapacity * 3);
        this.trailHead = new THREE.Color(this.color);

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(this.trailPositions, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(this.trailColors, 3));
        geometry.setDrawRange(0, 0);

        this.trail = new THREE.Line(geometry, new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        }));
        this.trail.frustumCulled = false; // the geometry changes every frame
        this.group.add(this.trail);
    }

    updateTrail(p, t) {
        if (!this.trail) return;

        const end = sampleIndexAt(this.trailTimes, t);
        let start = sampleIndexAt(this.trailTimes, t - this.trailWindow);
        if (end - start + 1 > this.trailCapacity) {
            start = end - this.trailCapacity + 1;
        }

        const count = end - start + 1;
        for (let i = 0; i < count; i++) {
            const sample = start + i;
            const j = i * 3;
            this.trailPositions[j] = this.trailWorld[sample * 3] - p[0];
            this.trailPositions[j + 1] = this.trailWorld[sample * 3 + 1] - p[1];
            this.trailPositions[j + 2] = this.trailWorld[sample * 3 + 2] - p[2];

            const shade = count > 1 ? i / (count - 1) : 1;
            this.trailColors[j] = this.trailHead.r * shade;
            this.trailColors[j + 1] = this.trailHead.g * shade;
            this.trailColors[j + 2] = this.trailHead.b * shade;
        }

        const geometry = this.trail.geometry;
        geometry.setDrawRange(0, count);
        geometry.attributes.position.needsUpdate = true;
        geometry.attributes.color.needsUpdate = true;
    }

    // engine time t -> world position; alpha 1..0 fades the whole body
    update(t, alpha = 1) {
        const p = sampleAt(this.track, t, this.frame);
        this.group.position.set(p[0], p[1], p[2]);
        this.updateTrail(p, t);
        for (const child of this.group.children) {
            if (child.material) {
                if (alpha < 1) child.material.transparent = true;
                child.material.opacity = alpha;
            }
        }
    }
}

// staging calls this once per track; hidden bodies never build visuals
export function createBody(track, index, frame, options = {}) {
    if (!track.visible) return null;
    return new Track_controller(track, index, frame, options);
}
