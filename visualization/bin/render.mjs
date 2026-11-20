/* Render one folder of trajectory CSVs to an MP4.

   node bin/render.mjs data/schwarzschild -q m --seconds 14 --out schwarzschild

   The page runs in capture mode: main.js exposes the stage on window, this
   script steps it frame by frame, screenshots the page, and pipes the PNGs
   into ffmpeg. No wall clock is involved, so every frame is deterministic.
   The absolute MP4 path is the last stdout line. */

import { spawn } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import puppeteer from "puppeteer-core";
import { createServer } from "./server.mjs";
import { loadFolder } from "../web/src/ingest.js";
import { spanOf } from "../web/src/layout.js";

const VIS = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const CHROME = "/usr/bin/google-chrome-stable";

// same presets the old tool exposed: 480p15, 720p30, 1080p60, 4K60
const QUALITY = {
    l: { width: 854, height: 480, fps: 15 },
    m: { width: 1280, height: 720, fps: 30 },
    h: { width: 1920, height: 1080, fps: 60 },
    p: { width: 3840, height: 2160, fps: 60 },
};

function parseCommandLine() {
    const { values, positionals } = parseArgs({
        allowPositionals: true,
        options: {
            quality: { type: "string", short: "q", default: "m" },
            seconds: { type: "string" },
            "no-trails": { type: "boolean", default: false },
            "no-bloom": { type: "boolean", default: false },
            out: { type: "string", default: "output_vid" },
            "dry-run": { type: "boolean", default: false },
        },
    });

    if (positionals.length !== 1) {
        throw new Error("usage: node bin/render.mjs <folder> [options]");
    }
    if (!QUALITY[values.quality]) {
        throw new Error("quality must be one of l, m, h, p");
    }
    if (values.seconds !== undefined) {
        const seconds = Number(values.seconds);
        if (!Number.isFinite(seconds) || seconds <= 0) {
            throw new Error("--seconds must be a positive number");
        }
        values.seconds = seconds;
    }

    return { folder: positionals[0], ...values };
}

// filesystem folder -> route relative to the visualization root ("data/x")
function folderRoute(raw) {
    const relative = path.relative(VIS, path.resolve(raw));
    if (!relative.startsWith("data" + path.sep)) {
        throw new Error("folder must be inside " + path.join(VIS, "data"));
    }
    return relative.split(path.sep).join("/");
}

function printInventory(name, tracks, skipped) {
    console.log(`${name}: ${tracks.length} track(s), ${skipped.length} skipped file(s)`);
    for (const track of tracks) {
        const start = track.times[0];
        const end = track.times[track.times.length - 1];
        console.log(
            `  ${track.name.padEnd(20)} ${track.kind.padEnd(8)} ` +
            `${String(track.times.length).padStart(6)} samples  t ${start}..${end}`,
        );
    }
    if (skipped.length > 0) {
        console.log("skipped: " + skipped.map((entry) => entry.file).join(", "));
    }
}

async function renderFrames(page, quality, totalFrames, output) {
    const ffmpeg = spawn("ffmpeg", [
        "-y",
        "-loglevel", "error",
        "-f", "image2pipe",
        "-framerate", String(quality.fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output,
    ], { stdio: ["pipe", "ignore", "pipe"] });

    let errorText = "";
    ffmpeg.stderr.on("data", (chunk) => {
        errorText += chunk;
    });
    ffmpeg.stdin.on("error", () => {}); // an early ffmpeg exit must not crash the loop
    const closed = once(ffmpeg, "close");

    for (let frame = 0; frame < totalFrames; frame++) {
        await page.evaluate((index) => {
            window.stage.update(index);
            window.stage.render();
        }, frame);

        const png = await page.screenshot({
            type: "png",
            clip: { x: 0, y: 0, width: quality.width, height: quality.height },
            optimizeForSpeed: true,
        });

        if (!ffmpeg.stdin.write(png)) {
            await once(ffmpeg.stdin, "drain");
        }

        process.stderr.write(`\rframe ${frame + 1}/${totalFrames}`);
    }

    ffmpeg.stdin.end();
    const [code] = await closed;
    process.stderr.write("\n");

    if (code !== 0) {
        const firstLine = errorText.trim().split("\n")[0];
        throw new Error("ffmpeg failed: " + (firstLine || "unknown error"));
    }
}

async function main() {
    const options = parseCommandLine();
    const folder = folderRoute(options.folder);
    const { tracks, skipped } = await loadFolder(path.join(VIS, folder));

    if (tracks.length === 0) {
        throw new Error(`no trajectory CSVs found in ${options.folder}`);
    }
    if (options["dry-run"]) {
        printInventory(folder, tracks, skipped);
        return;
    }

    const span = spanOf(tracks);
    if (!span) {
        throw new Error(`no time samples in ${options.folder}`);
    }

    const quality = QUALITY[options.quality];
    const seconds = options.seconds ??
        Math.min(Math.max((span.end - span.start) / 3, 6), 30);
    const totalFrames = Math.max(1, Math.round(seconds * quality.fps));

    const outName = path.basename(options.out).replace(/\.mp4$/i, "");
    if (!outName) {
        throw new Error("--out must not be empty");
    }
    const outDir = path.join(VIS, "out");
    await fs.mkdir(outDir, { recursive: true });
    const output = path.join(outDir, `${outName}.mp4`);

    const server = createServer(0);
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = server.address().port;

    let browser = null;
    try {
        browser = await puppeteer.launch({
            executablePath: CHROME,
            headless: true,
            args: [
                "--no-sandbox",
                "--enable-unsafe-swiftshader",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ],
        });
        const page = await browser.newPage();
        await page.setViewport({
            width: quality.width,
            height: quality.height,
            deviceScaleFactor: 1,
        });

        const url = new URL(`http://127.0.0.1:${port}/`);
        url.searchParams.set("data", folder);
        url.searchParams.set("capture", "1");
        url.searchParams.set("seconds", String(seconds));
        url.searchParams.set("fps", String(quality.fps));
        url.searchParams.set("width", String(quality.width));
        url.searchParams.set("height", String(quality.height));
        if (options["no-bloom"]) url.searchParams.set("no-bloom", "1");
        if (options["no-trails"]) url.searchParams.set("no-trails", "1");

        await page.goto(url.href, { waitUntil: "networkidle0" });
        await page.waitForFunction("window.ready === true", { timeout: 30000 });

        const frames = await page.evaluate(() => window.stage.totalFrames);
        await renderFrames(page, quality, frames, output);
    } finally {
        if (browser) await browser.close();
        server.close();
    }

    console.log(output);
}

main().catch((error) => {
    console.error("error: " + error.message);
    process.exitCode = 1;
});
