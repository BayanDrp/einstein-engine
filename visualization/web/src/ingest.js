/* CSV -> bodies. Required columns: t|time, x, y, z. Optional naming column
 * label|mode|name groups rows into "<value>-<k>" bodies (file stem otherwise).
 * Optional hints: type|kind (unknown -> particle), display_size, color,
 * visible (0/false/no hides). Broken files are skipped, never a crash. */

const ALIAS = {
  t: ["t", "time"],
  kind: ["type", "kind"],
  name: ["label", "mode", "name"],
};
const HIDDEN = new Set(["0", "false", "no"]);
const KINDS = ["particle", "matter"];

export function parseCsv(text, file) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length < 2) return { tracks: [], skipped: "no data rows" };

  const head = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const at = (aliases) => head.findIndex((h) => aliases.includes(h));
  const col = {
    t: at(ALIAS.t), x: head.indexOf("x"), y: head.indexOf("y"), z: head.indexOf("z"),
    kind: at(ALIAS.kind), name: at(ALIAS.name),
    size: head.indexOf("display_size"), color: head.indexOf("color"), visible: head.indexOf("visible"),
  };
  if (col.t < 0 || col.x < 0 || col.y < 0 || col.z < 0) {
    return { tracks: [], skipped: "missing required t/x/y/z columns" };
  }

  const stem = file.replace(/\.csv$/i, "");
  const num = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  // one pass over data rows; bad numeric rows are dropped
  const rows = [];
  for (const line of lines.slice(1)) {
    const cells = line.split(",").map((v) => v.trim());
    const t = num(cells[col.t]);
    const x = num(cells[col.x]);
    const y = num(cells[col.y]);
    const z = num(cells[col.z]);
    if (t !== null && x !== null && y !== null && z !== null) {
      rows.push({ t, x, y, z, cells });
    }
  }
  if (rows.length === 0) return { tracks: [], skipped: "no numeric t/x/y/z rows" };

  // one body = plain object; hints come from its first row
  const makeTrack = (name, bodyRows) => {
    const first = bodyRows[0];
    bodyRows.sort((a, b) => a.t - b.t); // engine rows are chronological; sort anyway
    const kind = (first.cells[col.kind] || "").trim().toLowerCase();
    const size = num(first.cells[col.size]);
    const color = (first.cells[col.color] || "").trim().toLowerCase();
    const vis = (first.cells[col.visible] || "").trim().toLowerCase();
    const times = [];
    const positions = [];
    for (const r of bodyRows) {
      times.push(r.t);
      positions.push(r.x, r.y, r.z);
    }
    return {
      name,
      kind: KINDS.includes(kind) ? kind : "particle",
      displaySize: size !== null && size > 0 ? size : null,
      color: color === "" ? null : color,
      visible: !HIDDEN.has(vis),
      times: Float64Array.from(times),
      positions: Float64Array.from(positions),
    };
  };

  // no naming column: the file is one body named after the file stem
  if (col.name < 0) {
    return { tracks: [makeTrack(stem, rows)], skipped: null };
  }

  // group rows by the naming column, one group per value in file order
  const groups = [];
  for (const row of rows) {
    const base = row.cells[col.name] || stem;
    let group = groups.find((g) => g.base === base);
    if (!group) {
      group = { base, name: "", rows: [] };
      groups.push(group);
    }
    group.rows.push(row);
  }

  // suffix "<value>-<k>" counting bodies that share the value
  const seen = {};
  for (const group of groups) {
    const k = (seen[group.base] = (seen[group.base] ?? 0) + 1) - 1;
    group.name = group.base + "-" + k;
  }
  groups.sort((a, b) => (a.name < b.name ? -1 : 1)); // stable palette order
  return { tracks: groups.map((g) => makeTrack(g.name, g.rows)), skipped: null };
}

/* Node loader: list the folder, read each CSV, collect skip reasons.
 * node:fs is imported here, not at the top, so the browser page can import
 * this file safely; the browser loader later reuses parseCsv via fetch. */
export async function loadFolder(folder) {
  const fs = await import("node:fs/promises");
  let entries;
  try {
    entries = await fs.readdir(folder, { withFileTypes: true });
  } catch (error) {
    throw new Error(`${folder}: cannot read folder (${error.code})`);
  }

  const files = entries
    .filter((e) => e.isFile() && e.name.toLowerCase().endsWith(".csv"))
    .map((e) => e.name)
    .sort();

  const tracks = [];
  const skipped = [];
  for (const file of files) {
    let result;
    try {
      result = parseCsv(await fs.readFile(`${folder}/${file}`, "utf8"), file);
    } catch {
      result = { tracks: [], skipped: "unreadable" };
    }
    if (result.skipped) skipped.push({ file, reason: result.skipped });
    else tracks.push(...result.tracks);
  }
  return { tracks, skipped };
}
