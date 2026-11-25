import assert from "node:assert/strict";
import { accessSync, constants } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createServer } from "../../../visualization/bin/server.mjs";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const VISUALIZATION_DIR = path.resolve(SCRIPT_DIR, "../../../visualization");
const require = createRequire(path.join(VISUALIZATION_DIR, "package.json"));
const puppeteer = require("puppeteer-core");

const EXPECTED_TRACKS = [
    "black_hole",
    "frozen-0",
    "isco_r6M-0",
    "photon_sphere_r3M-0",
    "staged-0",
];
const WIDTH = 1280;
const HEIGHT = 720;

function chromePath() {
    const configured = process.env.PUPPETEER_EXECUTABLE_PATH
        || process.env.CHROME_BIN
        || process.env.CHROME_PATH;
    const candidates = configured ? [configured] : [
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ];

    for (const candidate of candidates) {
        try {
            accessSync(candidate, constants.X_OK);
            return candidate;
        } catch {
            if (configured) {
                throw new Error(`Configured Chromium executable is unavailable: ${candidate}`);
            }
        }
    }

    throw new Error(
        "Chromium was not found. Set PUPPETEER_EXECUTABLE_PATH, CHROME_BIN, or CHROME_PATH.",
    );
}

function finite(values) {
    return values.every(Number.isFinite);
}

async function renderFrames(page, baseUrl, noBloom) {
    const url = new URL("/", baseUrl);
    url.searchParams.set("data", "data/schwarzschild");
    url.searchParams.set("capture", "1");
    url.searchParams.set("seconds", "6");
    url.searchParams.set("fps", "30");
    url.searchParams.set("width", String(WIDTH));
    url.searchParams.set("height", String(HEIGHT));
    if (noBloom) url.searchParams.set("no-bloom", "1");

    await page.goto(url.href, { waitUntil: "networkidle0" });
    await page.waitForFunction("window.ready === true", { timeout: 30000 });

    const frameCount = await page.evaluate(() => window.stage.totalFrames);
    assert.ok(frameCount > 2, "playback must contain first, middle, and final frames");
    const frames = [0, Math.floor((frameCount - 1) / 2), frameCount - 1];
    const snapshots = [];

    for (const frame of frames) {
        const snapshot = await page.evaluate((frameIndex) => {
            const stage = window.stage;
            stage.update(frameIndex);
            stage.render();

            const gl = stage.renderer.getContext();
            const bodies = stage.scene.children
                .filter((child) => child.isGroup && child.name)
                .map((body) => ({
                    name: body.name,
                    position: body.position.toArray(),
                }))
                .sort((a, b) => a.name.localeCompare(b.name));

            return {
                contextLost: gl.isContextLost(),
                drawCalls: stage.renderer.info.render.calls,
                canvasSize: [stage.renderer.domElement.width, stage.renderer.domElement.height],
                cameraPosition: stage.camera.position.toArray(),
                bodies,
                text: document.body.innerText,
            };
        }, frame);

        assert.equal(snapshot.contextLost, false, "WebGL context must remain available");
        assert.ok(snapshot.drawCalls > 0, `frame ${frame} must issue WebGL draw calls`);
        assert.deepEqual(snapshot.canvasSize, [WIDTH, HEIGHT]);
        assert.ok(finite(snapshot.cameraPosition), "camera position must remain finite");
        assert.deepEqual(
            snapshot.bodies.map((body) => body.name),
            EXPECTED_TRACKS,
            "all expected Schwarzschild tracks must be visible",
        );
        for (const body of snapshot.bodies) {
            assert.ok(finite(body.position), `${body.name} position must remain finite`);
        }
        snapshots.push(snapshot);
    }

    assert.notEqual(
        snapshots[0].text,
        snapshots[2].text,
        "the time display must change as playback advances",
    );

    await page.evaluate((frameIndex) => {
        window.stage.update(frameIndex);
        window.stage.render();
    }, frames[1]);
    const repeatedFirst = await page.evaluate((frameIndex) => {
        const stage = window.stage;
        stage.update(frameIndex);
        stage.render();
        return {
            cameraPosition: stage.camera.position.toArray(),
            bodies: stage.scene.children
                .filter((child) => child.isGroup && child.name)
                .map((body) => ({
                    name: body.name,
                    position: body.position.toArray(),
                }))
                .sort((a, b) => a.name.localeCompare(b.name)),
            text: document.body.innerText,
        };
    }, frames[0]);

    assert.deepEqual(repeatedFirst, {
        cameraPosition: snapshots[0].cameraPosition,
        bodies: snapshots[0].bodies,
        text: snapshots[0].text,
    }, "the same frame must reproduce the same scene state");
}

async function main() {
    const server = createServer(0);
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
    });

    let browser;
    try {
        browser = await puppeteer.launch({
            executablePath: chromePath(),
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
            width: WIDTH,
            height: HEIGHT,
            deviceScaleFactor: 1,
        });

        const pageErrors = [];
        page.on("pageerror", (error) => pageErrors.push(error.message));
        page.on("console", (message) => {
            if (
                message.type() === "error"
                && !message.location().url.endsWith("/favicon.ico")
            ) {
                pageErrors.push(message.text());
            }
        });
        page.on("response", (response) => {
            const responseUrl = new URL(response.url());
            if (
                response.status() >= 400
                && responseUrl.origin === `http://127.0.0.1:${server.address().port}`
                && responseUrl.pathname !== "/favicon.ico"
            ) {
                pageErrors.push(`HTTP ${response.status()}: ${response.url()}`);
            }
        });
        page.on("requestfailed", (request) => {
            const requestUrl = new URL(request.url());
            if (
                requestUrl.origin === `http://127.0.0.1:${server.address().port}`
                && requestUrl.pathname !== "/favicon.ico"
            ) {
                pageErrors.push(`request failed: ${request.url()} (${request.failure()?.errorText})`);
            }
        });

        const baseUrl = `http://127.0.0.1:${server.address().port}`;
        await renderFrames(page, baseUrl, false);
        await renderFrames(page, baseUrl, true);

        assert.deepEqual(pageErrors, [], "browser must have no page or application request errors");
        console.log("V10 WebGL: bloom and no-bloom frames rendered deterministically");
    } finally {
        try {
            if (browser) await browser.close();
        } finally {
            await new Promise((resolve) => server.close(resolve));
        }
    }
}

main().catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
});
