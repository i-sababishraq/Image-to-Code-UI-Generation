import { render, screen, waitFor } from "@testing-library/react";
import type { User, UserCredential } from "firebase/auth";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { AuthContext, type AuthContextValue } from "../app/auth/AuthProvider";
import { RequireAuth } from "../app/auth/RequireAuth";
import Index from "../app/routes/_index";

const dummyCredential: UserCredential = {
  user: {} as User,
  providerId: null,
  operationType: "signIn",
};

const createAuthValue = (overrides: Partial<AuthContextValue> = {}): AuthContextValue => ({
  user: null,
  loading: false,
  error: undefined,
  signInWithGoogle: vi.fn(async () => dummyCredential),
  signInWithEmail: vi.fn(async () => dummyCredential),
  registerWithEmail: vi.fn(async () => dummyCredential),
  signOut: vi.fn(async () => {}),
  ...overrides,
});

const routes = [
  {
    path: "/",
    element: <Index />,
  },
  {
    path: "/home",
    element: (
      <RequireAuth>
        <div>Protected Area</div>
      </RequireAuth>
    ),
  },
  {
    path: "/login",
    element: <div>Login Screen</div>,
  },
];

const renderWithAuth = (authValue: AuthContextValue, initialEntry = "/") => {
  const router = createMemoryRouter(routes, { initialEntries: [initialEntry] });

  render(
    <AuthContext.Provider value={authValue}>
      <RouterProvider router={router} />
    </AuthContext.Provider>
  );

  return router;
};

describe("RequireAuth routing", () => {
  it("redirects unauthenticated users to /login", async () => {
    const authValue = createAuthValue({ user: null, loading: false });
    const router = renderWithAuth(authValue);

    await waitFor(() => expect(router.state.location.pathname).toBe("/login"));
    expect(screen.getByText("Login Screen")).toBeInTheDocument();
  });

  it("renders protected content when a user is authenticated", async () => {
    const mockUser = {
      uid: "test-user-id",
      displayName: "Casey Creator",
      email: "casey@example.com",
    } as User;

    const authValue = createAuthValue({ user: mockUser, loading: false });
    const router = renderWithAuth(authValue);

    await waitFor(() => expect(screen.getByText("Protected Area")).toBeInTheDocument());
    expect(router.state.location.pathname).toBe("/home");
  });
});
