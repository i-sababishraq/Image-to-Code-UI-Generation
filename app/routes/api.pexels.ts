const PEXELS_API_KEY = process.env.PEXELS_API_KEY;

type ActionArgs = {
  request: Request;
};

type LoaderArgs = {
  request: Request;
};

const ALLOWED_ORIENTATIONS = new Set(["landscape", "portrait", "square"]);

export async function action({ request }: ActionArgs) {
  if (request.method !== "POST") {
    return new Response("Method Not Allowed", { status: 405 });
  }

  if (!PEXELS_API_KEY) {
    return Response.json(
      { error: "The Pexels API key has not been configured on the server." },
      { status: 500 },
    );
  }

  let payload: { query?: string; orientation?: string } = {};
  try {
    payload = (await request.json()) as typeof payload;
  } catch {
    return Response.json({ error: "Invalid JSON payload." }, { status: 400 });
  }

  const query = (payload.query ?? "").trim();
  if (!query) {
    return Response.json({ error: "Enter a description to search on Pexels." }, { status: 400 });
  }
  const orientation = normalizeOrientation(payload.orientation);

  const url = new URL("https://api.pexels.com/v1/search");
  url.searchParams.set("query", query);
  url.searchParams.set("per_page", "1");
  url.searchParams.set("orientation", orientation);
  url.searchParams.set("size", "large");

  const response = await fetch(url, {
    headers: {
      Authorization: PEXELS_API_KEY,
    },
  });

  if (!response.ok) {
    return Response.json(
      { error: "Unable to reach the Pexels API. Try again in a moment." },
      { status: 502 },
    );
  }

  const data = await response.json();
  const photo = data?.photos?.[0];

  if (!photo) {
    return Response.json(
      { error: "No matching images were found for that description." },
      { status: 404 },
    );
  }

  const normalized = {
    id: photo.id,
    description: photo.alt || query,
    photographer: photo.photographer,
    previewUrl: photo.src?.large ?? photo.src?.medium ?? photo.src?.original,
    downloadUrl: photo.src?.original ?? photo.src?.large2x ?? photo.src?.large,
    sourceUrl: photo.url,
    width: photo.width,
    height: photo.height,
  };

  return Response.json({ photo: normalized });
}

export async function loader({ request }: LoaderArgs) {
  const url = new URL(request.url);
  const imageUrl = url.searchParams.get("imageUrl");
  if (!imageUrl) {
    return new Response("Missing imageUrl parameter.", { status: 400 });
  }

  try {
    const remote = await fetch(imageUrl);
    if (!remote.ok) {
      return new Response("Unable to download the requested asset.", { status: 502 });
    }
    const arrayBuffer = await remote.arrayBuffer();
    const contentType = remote.headers.get("content-type") ?? "application/octet-stream";
    const filenameParam = url.searchParams.get("filename") ?? "pexels-image";
    const safeFilename = buildDownloadName(filenameParam, contentType);

    return new Response(Buffer.from(arrayBuffer), {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename="${safeFilename}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("Unable to download the requested asset.", { status: 502 });
  }
}

function normalizeOrientation(value?: string | null) {
  const normalized = (value ?? "").toLowerCase();
  return ALLOWED_ORIENTATIONS.has(normalized) ? normalized : "landscape";
}

function buildDownloadName(base: string, contentType: string) {
  const name = base
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "pexels-image";

  const extension = guessExtensionFromContentType(contentType);
  return extension && !name.endsWith(extension) ? `${name}${extension}` : name;
}

function guessExtensionFromContentType(contentType: string) {
  if (contentType.includes("png")) {
    return ".png";
  }
  if (contentType.includes("webp")) {
    return ".webp";
  }
  return ".jpg";
}
