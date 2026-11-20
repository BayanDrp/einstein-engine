import { test } from "node:test";
import assert from "node:assert/strict";
import { fitToFrame , spanOf ,sampleAt ,sampleIndexAt } from "../web/src/layout.js";

// fake bodies shared by the sampleAt tests: one orbiting, one starting late
const tracks = [
    { times: [0, 1, 2], positions: [0, 0, 0, 1, 1, 1, 2, 2, 2] },
    { times: [3, 4, 5], positions: [3, 3, 3, 4, 4, 4, 5, 5, 5] },
];

test("fitToFrame", () => {
    const tracks = [
        { positions: new Float32Array([0, 0, 0, 1, 1, 1]) },
        { positions: new Float32Array([-1, -1, -1, 2, 2, 2]) }
    ];
    const result = fitToFrame(tracks);
    assert.deepEqual(result.center, [0.5, 0.5, 0.5]);
    assert.equal(result.scale, 4 / 3);
    // farthest corner is sqrt(3 * 1.5^2) in engine units, then scaled
    assert.ok(Math.abs(result.radius - Math.sqrt(6.75) * 4 / 3) < 1e-9);
});

test("sampleIndexAt finds the sample at or before t", () => {
    const times = [0, 1, 2, 4];
    assert.equal(sampleIndexAt(times, -1), 0);  // before the window
    assert.equal(sampleIndexAt(times, 0), 0);   // exact first sample
    assert.equal(sampleIndexAt(times, 1.5), 1); // between samples
    assert.equal(sampleIndexAt(times, 3), 2);   // gap before the next sample
    assert.equal(sampleIndexAt(times, 4), 3);   // exact last sample
    assert.equal(sampleIndexAt(times, 99), 3);  // after the window
});

test("spanOf", () => {
    const tracks = [
        { times: [0, 1, 2] },
        { times: [3, 4, 5] }
    ];
    const result = spanOf(tracks);
    assert.deepEqual(result, { start: 0, end: 5 });
});

test("sampleAt holds before a window, interpolates inside it, holds after", () => {
    // identity frame: no shift, no scaling -> world coords equal engine coords
    const identity = { center: [0, 0, 0], scale: 1 };
    const [a, b] = tracks;
    assert.deepEqual(sampleAt(a, 1.5, identity), [1.5, 1.5, 1.5]); // midpoint
    assert.deepEqual(sampleAt(b, 1.5, identity), [3, 3, 3]);      // holds first sample
    assert.deepEqual(sampleAt(a, 99, identity), [2, 2, 2]);       // holds last sample
});

test("sampleAt maps engine coords through center and scale", () => {
    const frame = { center: [10, 0, 0], scale: 2 };
    // engine (1,1,1): x = (1-10)*2 = -18, y = (1-0)*2 = 2
    assert.deepEqual(sampleAt(tracks[0], 1, frame), [-18, 2, 2]);
});

test("fitToFrame on degenerate data: all points equal -> scale 1", () => {
    const tracks = [{ positions: [5, 5, 5] }];
    const frame = fitToFrame(tracks);
    assert.deepEqual(frame.center, [5, 5, 5]);
    assert.equal(frame.scale, 1);
    assert.equal(frame.radius, 0);
});