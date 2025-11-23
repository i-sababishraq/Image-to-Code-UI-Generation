import { readdir, stat, readFile } from "node:fs/promises";
import { resolve, relative, join, sep } from "node:path";
import { pythonWorkingDir } from "../services/vlmAgent";

export async function loader({ request }: { request: Request }) {
    const url = new URL(request.url);
    const relativePath = url.searchParams.get("path") || "";

    const outputsDir = resolve(pythonWorkingDir, "Outputs");
    const targetPath = resolve(outputsDir, relativePath);
    console.log(`[Files API] Request for: ${relativePath}`);
    console.log(`[Files API] Resolved outputsDir: ${outputsDir}`);
    console.log(`[Files API] Resolved targetPath: ${targetPath}`);

    // Security check: ensure targetPath is within outputsDir
    if (!targetPath.startsWith(outputsDir)) {
        return new Response("Forbidden", { status: 403 });
    }

    try {
        const stats = await stat(targetPath);

        if (stats.isDirectory()) {
            const entries = await readdir(targetPath, { withFileTypes: true });
            const files = entries.map((entry) => ({
                name: entry.name,
                isDirectory: entry.isDirectory(),
                path: relative(outputsDir, join(targetPath, entry.name)).split(sep).join("/"),
                size: entry.isDirectory() ? 0 : 0, // We'd need to stat each file for size, skipping for speed
            }));

            // Sort: directories first, then files
            files.sort((a, b) => {
                if (a.isDirectory === b.isDirectory) return a.name.localeCompare(b.name);
                return a.isDirectory ? -1 : 1;
            });

            return Response.json({ files });
        } else {
            // Serve file
            const fileContent = await readFile(targetPath);

            // Determine mime type (basic)
            let contentType = "application/octet-stream";
            if (targetPath.endsWith(".html")) contentType = "text/html";
            else if (targetPath.endsWith(".png")) contentType = "image/png";
            else if (targetPath.endsWith(".jpg")) contentType = "image/jpeg";
            else if (targetPath.endsWith(".txt")) contentType = "text/plain";
            else if (targetPath.endsWith(".json")) contentType = "application/json";

            return new Response(fileContent, {
                headers: {
                    "Content-Type": contentType,
                    "Content-Disposition": `inline; filename="${relativePath.split("/").pop()}"`,
                },
            });
        }
    } catch (error) {
        // If the directory doesn't exist yet (e.g. before first run), return empty list for root
        if ((error as { code?: string }).code === "ENOENT" && relativePath === "") {
            return Response.json({ files: [] });
        }
        return new Response("Not Found", { status: 404 });
    }
}
