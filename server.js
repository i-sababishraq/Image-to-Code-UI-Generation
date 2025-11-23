import { createRequestHandler } from "@react-router/express";
import compression from "compression";
import express from "express";
import morgan from "morgan";

const app = express();

app.use(compression());
app.disable("x-powered-by");

// Serve static files from the build directory
// "build/client" contains the public assets
app.use(
    "/assets",
    express.static("build/client/assets", { immutable: true, maxAge: "1y" })
);
app.use(express.static("build/client", { maxAge: "1h" }));

app.use(morgan("tiny"));

// Import the server build
// Note: We use dynamic import to ensure the build exists before importing
const build = await import("./build/server/index.js");

app.all(
    "(.*)",
    createRequestHandler({
        build,
        mode: process.env.NODE_ENV,
    })
);

const port = process.env.PORT || 3000;
const server = app.listen(port, () => {
    console.log(`Express server listening on port ${port}`);
});

// Set server timeout to 60 minutes to match the client-side timeout
server.timeout = 3600000;
