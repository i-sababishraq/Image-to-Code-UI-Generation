import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
  type User,
} from "firebase/auth";

import { auth } from "../lib/firebase.client";

type AuthState = {
  user: User | null;
  loading: boolean;
  error?: Error;
};

export function useFirebaseAuth() {
  const [state, setState] = useState<AuthState>({ user: null, loading: true, error: undefined });
  const provider = useMemo(() => new GoogleAuthProvider(), []);

  useEffect(() => {
    if (!auth) {
      setState((prev) => ({ ...prev, loading: false }));
      return;
    }

    const unsubscribe = onAuthStateChanged(
      auth,
      (user) => setState({ user, loading: false }),
      (error) => setState({ user: null, loading: false, error })
    );

    return () => unsubscribe();
  }, []);

  const withAuthInstance = useCallback(
    async <T,>(action: (currentAuth: NonNullable<typeof auth>) => Promise<T>) => {
      if (!auth) {
        setState((prev) => ({
          ...prev,
          error: new Error("Firebase Auth is not available in this environment."),
        }));
        throw new Error("Firebase Auth is not available in this environment.");
      }

      setState((prev) => ({ ...prev, error: undefined }));

      try {
        return await action(auth);
      } catch (error) {
        const normalizedError =
          error instanceof Error ? error : new Error("An unexpected authentication error occurred.");
        setState((prev) => ({ ...prev, error: normalizedError }));
        throw normalizedError;
      }
    },
    []
  );

  const signInWithGoogle = () =>
    withAuthInstance((currentAuth) => signInWithPopup(currentAuth, provider));

  const signInWithEmail = (email: string, password: string) =>
    withAuthInstance((currentAuth) => signInWithEmailAndPassword(currentAuth, email, password));

  const registerWithEmail = (email: string, password: string) =>
    withAuthInstance((currentAuth) => createUserWithEmailAndPassword(currentAuth, email, password));

  const handleSignOut = () => withAuthInstance((currentAuth) => signOut(currentAuth));

  return {
    ...state,
    signInWithGoogle,
    signInWithEmail,
    registerWithEmail,
    signOut: handleSignOut,
  };
}
