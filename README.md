# Image-to-Code-UI-Generation
Multimodal AI Agent for Image-to-Code UI Generation - CAP 6411 Computer Vision Systems Group Project
by Megan Diehl, Sabab Ishraq, Fazil Kagdi, Tina Tran, Ashmal Vayani

Image2Code is a React Router application that lets designers and engineers collaborate with an agentic AI to turn text prompts and wireframe screenshots into production-ready UI code. The app exposes a chat-first workflow with dedicated preview and history screens so you can iterate quickly, review generated files, and revisit earlier runs.

## Features

- **Chat-led workflow** – Craft prompts, upload reference images, and receive rich assistant responses inside a tailored messenger experience.
- **Live previews** – Inspect generated layouts and files without leaving the app, then export or open them in your editor of choice.
- **Run history** – Keep track of prior generations, tagged with tech stacks and timestamps for quick restoration.
- **Dark-mode UI** – Opinionated visual design optimized for late-night build sessions.
- **Firebase ready** – Client bootstraps Firebase Authentication, Firestore, and (optionally) Analytics for persistence and team features.

# Frontend (React, by Megan)

## Tech Stack

- [React Router](https://reactrouter.com/) 7 (app directory, data APIs, SSR-ready)
- [Vite](https://vitejs.dev/) for fast dev builds
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/) (via `@tailwindcss/vite`)
- [Firebase Web SDK](https://firebase.google.com/docs/web/setup) for auth and data storage

## Prerequisites

- Node.js 20+ (the project currently targets v22)
- npm 10+
- A Firebase project (for authentication and Firestore persistence)
- Firebase CLI (`npm install -g firebase-tools`) if you plan to deploy or use emulators

## Getting Started

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Configure environment variables**

   Copy the provided template, then update the values with your own Firebase project credentials. Avoid committing the real `.env` file.

   ```bash
   cp .env.example .env
   ```

   | Variable                              | Description                                |
   | ------------------------------------- | ------------------------------------------ |
   | `VITE_FIREBASE_API_KEY`               | Web API key from Firebase project settings |
   | `VITE_FIREBASE_AUTH_DOMAIN`           | Auth domain                                 |
   | `VITE_FIREBASE_PROJECT_ID`            | Project ID                                  |
   | `VITE_FIREBASE_STORAGE_BUCKET`        | Storage bucket                              |
   | `VITE_FIREBASE_MESSAGING_SENDER_ID`   | Messaging sender ID                         |
   | `VITE_FIREBASE_APP_ID`                | Web app ID                                  |
   | `VITE_FIREBASE_MEASUREMENT_ID`        | (Optional) Analytics measurement ID         |

   The sample values in `.env.example` correspond to a development sandbox. Replace them with secrets from your Firebase console before shipping.

3. **Run the development server**

   ```bash
   npm run dev
   ```

   The React Router dev server boots on <http://localhost:5173>. Hot Module Reloading is enabled.

## Available Scripts

| Script              | Purpose                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `npm run dev`       | Start the React Router/Vite dev server with HMR                     |
| `npm run build`     | Produce production client & server bundles in `build/`             |
| `npm start`         | Serve the production build (`react-router-serve ./build/server`)   |
| `npm run typecheck` | Generate route types and run the TypeScript compiler               |

To preview a production build locally:

```bash
npm run build
npm start
```

## Firebase Integration

The app provides shared Firebase helpers under `app/lib/`, `app/hooks/`, and `app/services/`:

- `app/lib/firebase.client.ts` initializes the Firebase app, Auth, Firestore, and Analytics (when supported).
- `app/hooks/useFirebaseAuth.ts` exposes a ready-to-use hook for Google sign-in and session tracking.
- `app/services/userData.ts` includes Firestore helpers for storing user profiles and prompt history.

### CLI Login Tips (Windows)

If `firebase login` fails with `spawn cmd ENOENT`, PowerShell cannot find `cmd.exe`. Either ensure `C:\Windows\System32` is on your `PATH`/`ComSpec`, or use the device-code flow:

```bash
firebase login --no-localhost
```

Open the printed URL manually, enter the code, and the CLI will finish authentication.

## Project Structure

```
app/
  lib/              # Firebase bootstrap
  hooks/            # React hooks (auth)
  services/         # Firestore helpers
  routes/           # React Router routes (chat experience lives in welcome/)
  welcome/          # Chat/Preview/History UI components
public/              # Static assets
react-router.config.ts
vite.config.ts
```

## Deployment

1. Build the production assets (`npm run build`).
2. Deploy `build/client` (static assets) and `build/server` (server bundle) on your Node-capable host or container platform.
3. Supply the same Firebase env vars in your hosting environment.

The default Dockerfile in the repo can be used to containerize the server if you prefer.

## Next Steps

- Connect Firebase Auth state to routing (e.g., guard preview/history or show user menus).
- Persist generated prompts via the Firestore helpers.
- Wire the preview panel to real generated assets.

Happy building!
