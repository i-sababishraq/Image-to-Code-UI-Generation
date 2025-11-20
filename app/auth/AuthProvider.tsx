import { createContext, useContext, type ReactNode } from "react";

import { useFirebaseAuth } from "../hooks/useFirebaseAuth";

type AuthContextValue = ReturnType<typeof useFirebaseAuth>;

const AuthContext = createContext<AuthContextValue | null>(null);

export { AuthContext };

export function AuthProvider({ children }: { children: ReactNode }) {
  const authState = useFirebaseAuth();

  return <AuthContext.Provider value={authState}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }

  return context;
}

export type { AuthContextValue };
