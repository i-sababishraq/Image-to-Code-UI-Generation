import { Navigate } from "react-router";

import { useAuth } from "../auth/AuthProvider";

export default function Index() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-100">
        <div className="text-center">
          <div className="mb-2 animate-spin rounded-full border-4 border-[#2F6BFF] border-t-transparent p-6" />
          <p className="text-sm text-slate-400">Checking your session…</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <Navigate to="/home" replace />;
}
