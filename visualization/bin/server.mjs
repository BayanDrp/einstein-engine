/* Static server shared by bin/dev.mjs and bin/render.mjs. Routes, all
 * confined to folders under the visualization directory:
 *   /                          -> web/index.html
 *   /src/**                    -> web/src/**
 *   /node_modules/three/**     -> node_modules/three/**   (import map targets)
 *   /data/**                   -> data/**                 (CSV bodies)
 *   /list?f=data/<folder>      -> JSON { files: [...csv names...] } sorted
 * Run by hand:  node bin/server.mjs <port>   (default port 8765)
 */
import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const VIS = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

const TYPE = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".csv": "text/csv; charset=utf-8",
    ".png": "image/png",
};

// URL route -> disk path, or null when outside an allowed root
function locate(route) {
    let disk;
    if (route === "/") {
        disk = path.join(VIS, "web", "index.html");
    } else if (route.startsWith("/node_modules/three/") || route.startsWith("/data/")) {
        disk = path.join(VIS, route); // served from the visualization root
    } else {
        disk = path.join(VIS, "web", route.slice(1));
    }
    const clean = path.resolve(disk);
    return clean.startsWith(VIS + path.sep) ? clean : null;
}

// sorted CSV file names of one folder, [] for anything unreadable/outside
async function listCsv(relative) {
    const clean = path.resolve(path.join(VIS, relative));
    if (!clean.startsWith(VIS + path.sep)) return [];
    const entries = await fs.readdir(clean, { withFileTypes: true });
    return entries
        .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".csv"))
        .map((entry) => entry.name)
        .sort();
}

function notFound(response, text) {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end(text);
}

export function createServer(port = 8765) {
    return http.createServer(async (request, response) => {
        try {
            const url = new URL(request.url, `http://localhost:${port}`);
            const route = decodeURIComponent(url.pathname);
            if (route === "/list") {
                response.writeHead(200, { "content-type": "application/json" });
                response.end(JSON.stringify({ files: await listCsv(url.searchParams.get("f") || "") }));
                return;
            }
            const disk = locate(route);
            if (!disk) return notFound(response, "forbidden");
            let body;
            try {
                body = await fs.readFile(disk);
            } catch (error) {
                return notFound(response, error.code === "ENOENT" ? "not found" : "read error");
            }
            response.writeHead(200, {
                "content-type": TYPE[path.extname(disk).toLowerCase()] ?? "application/octet-stream",
            });
            response.end(body);
        } catch (error) {
            notFound(response, error.code === "ENOENT" ? "not found" : "server error");
        }
    });
}

// started directly (node bin/server.mjs): serve and print the URL
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    const port = Number(process.argv[2]) || 8765;
    createServer(port).listen(port, () =>
        console.log(`serving at http://127.0.0.1:${port}`),
    );
}
