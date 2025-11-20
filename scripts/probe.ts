import { action } from "../app/routes/api.vlm.ts";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

async function main() {
  const imagePath = resolve("Image-to-Code-UI-Generation/assets/bird-colorful-logo-gradient-vector_343694-1365.jpg");
  const buffer = await readFile(imagePath);
  const blob = new Blob([buffer], { type: "image/jpeg" });
  const file = new File([blob], "wireframe.jpg", { type: "image/jpeg" });

  const formData = new FormData();
  formData.append("prompt", "Probe request from scripts/probe.ts");
  formData.append("images", file);

  const request = new Request("http://localhost/api/vlm", {
    method: "POST",
    body: formData,
  });

  const response = await action({ request });
  console.log("status", response.status);
  console.log(await response.json());
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
