import { index, route } from "@react-router/dev/routes";

export default [
  index("routes/_index.tsx"),
  route("home", "routes/home.tsx"),
  route("login", "routes/login.tsx"),
  route("register", "routes/register.tsx"),
  route("welcome", "routes/welcome.tsx"),
  route("api/vlm", "routes/api.vlm.ts"),
  route("api/pexels", "routes/api.pexels.ts"),
  route("api/files", "routes/api.files.ts"),
];
