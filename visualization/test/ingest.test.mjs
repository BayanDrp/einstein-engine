/* Tests for web/src/ingest.js. Run with:
 *   node --test test/ingest.test.mjs
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { parseCsv, loadFolder } from "../web/src/ingest.js";


test("no naming column: whole file is one body named after the stem", () => {
    const { tracks, skipped } = parseCsv(
        "t,x,y,z\n0,1,2,3\n1,4,5,6\n",
        "planet.csv",
    );
    assert.equal(skipped, null);
    assert.equal(tracks.length, 1);
    assert.equal(tracks[0].name, "planet");
    assert.equal(tracks[0].kind, "particle");
    assert.ok(tracks[0].positions instanceof Float64Array);
    assert.deepEqual(Array.from(tracks[0].times), [0, 1]);
    assert.deepEqual(Array.from(tracks[0].positions), [1, 2, 3, 4, 5, 6]);
});


test("naming column groups rows, sorted alphabetically", () => {
    const { tracks } = parseCsv(
        "label,t,x,y,z\nmoon,0,0,0,0\nsun,0,1,1,1\nmoon,5,2,2,2\n",
        "orbits.csv",
    );
    // rows sharing a label merge into one body (that is how an orbit's
    // thousands of samples stay together); one group per value -> "-0"
    assert.deepEqual(tracks.map((t) => t.name), ["moon-0", "sun-0"]);
    assert.deepEqual(Array.from(tracks[0].times), [0, 5]); // both moon rows together
    assert.deepEqual(Array.from(tracks[1].times), [0]);
});


test("alias 'time' works and an unknown type falls back to particle", () => {
    const { tracks } = parseCsv("time,x,y,z,type\n0,1,0,0,wave\n", "a.csv");
    assert.equal(tracks[0].kind, "particle");
    assert.deepEqual(Array.from(tracks[0].times), [0]);
});


test("hints: type, display_size, color, visible (0/false/no hides)", () => {
    const { tracks } = parseCsv(
        "type,display_size,color,visible,t,x,y,z\nmatter,2.5,grey,0,0,0,0,0\n",
        "hole.csv",
    );
    const body = tracks[0];
    assert.equal(body.kind, "matter");
    assert.equal(body.displaySize, 2.5);
    assert.equal(body.color, "grey");
    assert.equal(body.visible, false);
});


test("missing hint columns get neutral defaults", () => {
    const { tracks } = parseCsv("label,t,x,y,z\np,0,0,0,0\n", "we.csv");
    assert.equal(tracks[0].visible, true);
    assert.equal(tracks[0].displaySize, null);
    assert.equal(tracks[0].color, null);
});


test("rows with non numeric t/x/y/z are dropped, the file still parses", () => {
    const { tracks, skipped } = parseCsv(
        "t,x,y,z\n0,1,2,3\nbad,row,x,here\n1,4,5,6\n",
        "ok.csv",
    );
    assert.equal(skipped, null);
    assert.deepEqual(Array.from(tracks[0].times), [0, 1]);
});


test("rows are sorted by time even when unordered", () => {
    const { tracks } = parseCsv(
        "t,x,y,z\n5,1,0,0\n0,2,0,0\n2,3,0,0\n",
        "jumbled.csv",
    );
    assert.deepEqual(Array.from(tracks[0].times), [0, 2, 5]);
    assert.deepEqual(Array.from(tracks[0].positions), [2, 0, 0, 3, 0, 0, 1, 0, 0]);
});


test("broken files return a skip reason, never a throw", () => {
    assert.equal(parseCsv("", "empty.csv").skipped, "no data rows");
    assert.equal(parseCsv("t,x,y,z\n", "header-only.csv").skipped, "no data rows");
    assert.equal(
        parseCsv("a,b,c\n1,2,3\n", "wrong-columns.csv").skipped,
        "missing required t/x/y/z columns",
    );
    assert.equal(
        parseCsv("t,x,y,z\nbad,row,here,now\n", "all-bad-rows.csv").skipped,
        "no numeric t/x/y/z rows",
    );
});


test("an empty naming value falls back to the file stem", () => {
    const { tracks } = parseCsv("label,t,x,y,z\n,0,0,0,0\n", "dust.csv");
    assert.deepEqual(tracks.map((t) => t.name), ["dust-0"]);
});


test("loadFolder reads the real schwarzschild folder", async () => {
    // resolve the folder from this file's location, not the process cwd
    const folder = fileURLToPath(new URL("../data/schwarzschild", import.meta.url));
    const { tracks, skipped } = await loadFolder(folder);
    assert.deepEqual(
        tracks.map((t) => t.name),
        ["black_hole", "frozen-0", "staged-0", "isco_r6M-0", "photon_sphere_r3M-0"],
    );
    assert.deepEqual(skipped, [
        {
            file: "geodesic_orbits_static_summary.csv",
            reason: "missing required t/x/y/z columns",
        },
    ]);
    assert.equal(tracks[0].kind, "matter");       // black_hole.csv carries type=matter
    assert.equal(tracks[1].times.length, 1);      // frozen-0 marker holds one sample
    assert.equal(tracks[3].times.length, 3695);   // isco ring
});


test("loadFolder throws only for an unreachable folder", async () => {
    await assert.rejects(() => loadFolder("no/such/folder"), /cannot read folder/);
});
