import { useState } from "react";
import type { FormEvent } from "react";
import { Link, Navigate, useLocation } from "react-router";

import { useAuth } from "../auth/AuthProvider";

type LocationState = {
  from?: { pathname?: string };
  authError?: string;
};

export default function Login() {
  const { signInWithEmail, signInWithGoogle, loading, user, error } = useAuth();
  const location = useLocation();
  const locationState =
    typeof location.state === "object" && location.state !== null
      ? (location.state as LocationState)
      : undefined;

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [formError, setFormError] = useState<string | null>(locationState?.authError ?? null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const redirectTo =
    locationState?.from?.pathname && locationState.from.pathname.length > 0
      ? locationState.from.pathname
      : "/home";

  if (user && !loading) {
    return <Navigate to={redirectTo} replace />;
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);

    if (!email || !password) {
      setFormError("Please enter both an email address and password.");
      return;
    }

    try {
      setIsSubmitting(true);
      await signInWithEmail(email, password);
    } catch (authError) {
      if (authError instanceof Error) {
        setFormError(authError.message);
      } else {
        setFormError("Unable to sign in with the provided credentials.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setFormError(null);

    try {
      setIsSubmitting(true);
      await signInWithGoogle();
    } catch (authError) {
      if (authError instanceof Error) {
        setFormError(authError.message);
      } else {
        setFormError("Unable to sign in with Google right now.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-slate-100">
      <section className="w-full max-w-md rounded-2xl border border-slate-800/60 bg-slate-900/70 p-8 shadow-xl shadow-black/40">
        <header className="mb-6 text-center">
          <h1 className="text-2xl font-semibold">Welcome back</h1>
          <p className="mt-2 text-sm text-slate-400">
            Sign in to continue building with <span className="text-[#6FA3FF]">Image2Code</span>.
          </p>
        </header>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-300" htmlFor="email">
              Email address
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 text-sm focus:border-[#2F6BFF] focus:outline-none focus:ring-2 focus:ring-[#2F6BFF]/40"
              placeholder="you@example.com"
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-slate-300" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 text-sm focus:border-[#2F6BFF] focus:outline-none focus:ring-2 focus:ring-[#2F6BFF]/40"
              placeholder="••••••••"
              disabled={isSubmitting}
            />
          </div>

          {(formError || error) && (
            <p className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {formError ?? error?.message}
            </p>
          )}

          <button
            type="submit"
            className="w-full rounded-lg bg-[#2F6BFF] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#2157db] disabled:cursor-not-allowed disabled:bg-[#2F6BFF]/60"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div className="mt-6 flex items-center gap-3 text-xs text-slate-500">
          <span className="h-px flex-1 bg-slate-800" />
          or
          <span className="h-px flex-1 bg-slate-800" />
        </div>

        <button
          type="button"
          onClick={handleGoogleSignIn}
          className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg border border-slate-800 bg-slate-950/60 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-[#2F6BFF]/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-70"
          disabled={isSubmitting}
        >
          Continue with Google
        </button>

        <p className="mt-6 text-center text-sm text-slate-400">
          Don&apos;t have an account?{" "}
          <Link className="text-[#6FA3FF] hover:underline" to="/register">
            Create one
          </Link>
        </p>
      </section>
    </main>
  );
}
