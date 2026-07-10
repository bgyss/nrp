// Tiny static file server for the G2 demo: serves the repo root so the page can
// fetch /webgpu/demo/*, /webgpu/shader_gen.mjs, and /out/g2-demo/export*/ blobs.
// (navigator.gpu needs a secure context; http://localhost qualifies, file:// fetch
// of sibling binaries does not.)
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const MIME = {
  ".html": "text/html",
  ".mjs": "text/javascript",
  ".js": "text/javascript",
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".css": "text/css",
};

export function createDemoServer() {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://localhost");
      let file = decodeURIComponent(url.pathname);
      if (file === "/") file = "/webgpu/demo/index.html";
      const full = path.join(repoRoot, file);
      if (!full.startsWith(repoRoot)) throw new Error("path escape");
      const body = await readFile(full);
      res.writeHead(200, { "content-type": MIME[path.extname(full)] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT || 8791);
  createDemoServer().listen(port, () => {
    console.log(`G2 demo: http://localhost:${port}/webgpu/demo/index.html`);
  });
}
