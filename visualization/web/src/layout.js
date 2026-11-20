

export function fitToFrame(tracks) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (const track of tracks) {
        const p = track.positions;
        for (let i = 0; i < p.length; i += 3) {
            if (p[i] < minX) minX = p[i];
            if (p[i] > maxX) maxX = p[i];
            if (p[i + 1] < minY) minY = p[i + 1];
            if (p[i + 1] > maxY) maxY = p[i + 1];
            if (p[i + 2] < minZ) minZ = p[i + 2];
            if (p[i + 2] > maxZ) maxZ = p[i + 2];
        }
    }

    if (!Number.isFinite(minX)) {
        return { center: [0, 0, 0], scale: 1, radius: 0 };
    }

    const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    const maxSpan = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const scale = maxSpan > 0 ? 4 / maxSpan : 1;

    // farthest sample from the center; staging uses it to place the camera
    let radiusSq = 0;
    for (const track of tracks) {
        const p = track.positions;
        for (let i = 0; i < p.length; i += 3) {
            const dx = p[i] - center[0];
            const dy = p[i + 1] - center[1];
            const dz = p[i + 2] - center[2];
            const d = dx * dx + dy * dy + dz * dz;
            if (d > radiusSq) radiusSq = d;
        }
    }

    return { center, scale, radius: Math.sqrt(radiusSq) * scale };
}


/* The global time window: smallest first sample to largest last sample over
 * all bodies (single sample markers count). The playback clock runs from
 * start to end. Empty input returns null. */
export function spanOf(tracks) {
    let start = Infinity;
    let end = -Infinity;
    for (const track of tracks) {
        if (track.times.length === 0) continue;
        start = Math.min(start, track.times[0]);
        end = Math.max(end, track.times[track.times.length - 1]);
    }
    if (!Number.isFinite(start)) return null;
    return { start, end };
};


/* Index of the last sample at or before t. Times are sorted, so a binary
 * search is enough; before the first sample it returns 0, after the last
 * sample it returns the last index. */
export function sampleIndexAt(times, t) {
    let lo = 0;
    let hi = times.length - 1;
    if (t <= times[0]) return 0;
    if (t >= times[hi]) return hi;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (times[mid] <= t) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}


/* Engine time t -> world position [x, y, z] for one body: the time is
 * clamped into the body's recorded window (a body whose recording ends
 * early holds its last position; staging fades it instead of inventing
 * motion), linearly interpolated between samples, then centered on the
 * frame center and scaled into the stage. A body with one sample holds
 * that one point. */
export function sampleAt(track, t, frame) {
    const times = track.times;
    const positions = track.positions;
    const lastSample = times.length - 1;

    let time = t;
    if (time < times[0]) time = times[0];
    if (time > times[lastSample]) time = times[lastSample];

    // the interval [lo, lo+1] that holds `time` (never past the last sample)
    const lo = sampleIndexAt(times, time);

    // share = 0 at sample lo, 1 at sample lo+1 (equal times -> share 0)
    const next = Math.min(lo + 1, lastSample); // single sample: next == lo
    let share = 0;
    if (next !== lo && times[next] > times[lo]) {
        share = (time - times[lo]) / (times[next] - times[lo]);
    }

    const x = positions[lo * 3] + share * (positions[next * 3] - positions[lo * 3]);
    const y = positions[lo * 3 + 1] + share * (positions[next * 3 + 1] - positions[lo * 3 + 1]);
    const z = positions[lo * 3 + 2] + share * (positions[next * 3 + 2] - positions[lo * 3 + 2]);

    return [
        (x - frame.center[0]) * frame.scale,
        (y - frame.center[1]) * frame.scale,
        (z - frame.center[2]) * frame.scale,
    ];
};



