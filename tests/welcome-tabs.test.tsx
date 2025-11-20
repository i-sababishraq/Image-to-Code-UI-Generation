import { fireEvent, render, screen } from "@testing-library/react";
import type { User, UserCredential } from "firebase/auth";
import { describe, expect, it, vi } from "vitest";

import { AuthContext, type AuthContextValue } from "../app/auth/AuthProvider";
import { Welcome } from "../app/welcome/welcome";

const dummyCredential: UserCredential = {
  user: {} as User,
  providerId: null,
  operationType: "signIn",
};

const baseAuthValue: AuthContextValue = {
  user: {
    uid: "123",
    email: "user@example.com",
  } as User,
  loading: false,
  error: undefined,
  signInWithGoogle: vi.fn(async () => dummyCredential),
  signInWithEmail: vi.fn(async () => dummyCredential),
  registerWithEmail: vi.fn(async () => dummyCredential),
  signOut: vi.fn(async () => {}),
};

describe("Welcome tabs", () => {
  it("switches between chat, preview, and history panels", () => {
    render(
      <AuthContext.Provider value={baseAuthValue}>
        <Welcome />
      </AuthContext.Provider>
    );

    expect(screen.getByText(/Chat Canvas/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Preview/i }));
    expect(
      screen.getByText(/No UI components have been generated yet\./i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /History/i }));
    expect(screen.getByText(/Run history/i)).toBeInTheDocument();
  });
});
