import { Navigate, Outlet, useLocation } from "react-router";
import { type ReactNode } from "react";

import { useAuth } from "./AuthProvider";

type RequireAuthProps = {
  children?: ReactNode;
  redirectTo?: string;
};

export function RequireAuth({ children, redirectTo = "/login" }: RequireAuthProps) {
  const { user, loading, error } = useAuth();
  const location = useLocation();

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
    return (
      <Navigate
        to={redirectTo}
        replace
        state={{
          from: location,
          authError: error?.message,
        }}
      />
    );
  }

  if (children) {
    return <>{children}</>;
  }

  return <Outlet />;
}
