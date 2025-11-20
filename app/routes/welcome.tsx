import { Navigate } from "react-router";

export default function WelcomeRedirect() {
  return <Navigate to="/home" replace />;
}
