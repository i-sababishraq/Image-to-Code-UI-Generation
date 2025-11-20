import { randomUUID } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { extname, resolve } from "node:path";

import { pythonWorkingDir, runVlmAgent } from "../services/vlmAgent";

const ONBOARDING_TIPS = [
  "Here are a few quick steps to get the most out of Image2Code:",
  "• Capture a clear screenshot of your wireframe (mobile or desktop) before uploading.",
  "• Describe the desired behavior, breakpoints, and components in the prompt field.",
  "• Iterate by tweaking text instructions or swapping in updated wireframes.",
  "• Use Preview to inspect generated files and History to revisit past explorations.",
].join("\n");

type ActionArgs = {
  request: Request;
};

export async function action({ request }: ActionArgs) {
  if (request.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.includes("multipart/form-data")) {
    return Response.json(
      {
        status: {
          kind: "error",
          text: "Unsupported content type.",
          detail: "Send multipart/form-data with prompt and images.",
        },
      },
      { status: 415 },
    );
  }

  const formData = await request.formData();
  const promptValue = formData.get("prompt");
  const prompt = typeof promptValue === "string" ? promptValue.trim() : "";
  const skipImageRequirement = formData.get("skipImageRequirement") === "true";

  if (!prompt) {
    return Response.json(
      {
        status: {
          kind: "error",
          text: "Add a prompt describing the UI you want generated.",
        },
      },
      { status: 400 },
    );
  }

  const fileValues = formData.getAll("images");
  const imageFiles = fileValues.filter((value): value is File => value instanceof File && value.size > 0);

  if (imageFiles.length === 0 && skipImageRequirement) {
    return Response.json(
      {
        messages: [
          {
            id: randomUUID(),
            role: "assistant",
            variant: "accent",
            content: ONBOARDING_TIPS,
          },
        ],
        status: {
          kind: "success",
          text: "Onboarding tips ready.",
        },
        usedFallback: true,
      },
      { status: 200 },
    );
  }

  if (imageFiles.length === 0 && !skipImageRequirement) {
    return Response.json(
      {
        status: {
          kind: "error",
          text: "Attach at least one wireframe image before running the agent.",
        },
      },
      { status: 400 },
    );
  }

  const invalidFile = imageFiles.find((file) => !file.type.startsWith("image/"));
  if (invalidFile) {
    return Response.json(
      {
        status: {
          kind: "error",
          text: "Only image attachments are supported right now.",
          detail: `Unsupported file type: ${invalidFile.type || "unknown"}`,
        },
      },
      { status: 400 },
    );
  }

  const uploadDir = resolve(pythonWorkingDir, "uploads");
  await mkdir(uploadDir, { recursive: true });

  const savedPaths: string[] = [];
  let result;

  try {
    for (const file of imageFiles) {
      const extension = sanitizeExtension(extname(file.name)) || guessExtensionFromType(file.type);
      const filename = `${Date.now()}-${randomUUID()}${extension}`;
      const filePath = resolve(uploadDir, filename);
      const buffer = Buffer.from(await file.arrayBuffer());
      await writeFile(filePath, buffer);
      savedPaths.push(filePath);
    }

    const primaryImage = savedPaths[0];
    result = await runVlmAgent(prompt, primaryImage);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown server error.";
    return Response.json(
      {
        status: {
          kind: "error",
          text: "Unable to run the VLM agent.",
          detail: message,
        },
      },
      { status: 500 },
    );
  } finally {
    await Promise.all(
      savedPaths.map(async (filePath) => {
        try {
          await rm(filePath, { force: true });
        } catch {
          // Ignore cleanup errors.
        }
      }),
    );
  }

  return Response.json(result);
}

function sanitizeExtension(value: string): string {
  if (!value) {
    return "";
  }
  if (!/^\.[a-zA-Z0-9]+$/.test(value)) {
    return "";
  }
  return value.toLowerCase();
}

function guessExtensionFromType(mime: string): string {
  if (mime === "image/png") {
    return ".png";
  }
  if (mime === "image/jpeg" || mime === "image/jpg") {
    return ".jpg";
  }
  if (mime === "image/webp") {
    return ".webp";
  }
  return "";
}
