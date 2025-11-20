import { type User } from "firebase/auth";
import {
  Timestamp,
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  onSnapshot,
  orderBy,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  writeBatch,
  type DocumentData,
  type QueryDocumentSnapshot,
  type Unsubscribe,
} from "firebase/firestore";

import { db } from "../lib/firebase.client";

export type UserProfile = {
  displayName: string | null;
  email: string | null;
  photoURL: string | null;
  onboardingComplete?: boolean;
  updatedAt?: Date;
};

function assertDb() {
  if (!db) {
    throw new Error("Firestore is not available in the current environment.");
  }
}

export async function createOrUpdateUserProfile(user: User, profile?: Partial<UserProfile>) {
  assertDb();
  const ref = doc(collection(db, "users"), user.uid);

  await setDoc(
    ref,
    {
      displayName: user.displayName,
      email: user.email,
      photoURL: user.photoURL,
      onboardingComplete: profile?.onboardingComplete ?? false,
      updatedAt: serverTimestamp(),
      ...profile,
    },
    { merge: true }
  );
}

export async function fetchUserProfile(uid: string) {
  assertDb();
  const ref = doc(collection(db, "users"), uid);
  const snapshot = await getDoc(ref);

  if (!snapshot.exists()) return null;

  return snapshot.data() as UserProfile;
}

export type StoredChatAttachment = {
  id: string;
  name: string;
  type: "image" | "file";
  size?: number;
};

export type StoredChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  variant?: "accent" | "subtle";
  attachments?: StoredChatAttachment[];
  renderAsCode?: boolean;
  codeLanguage?: string;
  htmlPath?: string;
  htmlWebPath?: string | null;
};

export type PromptHistoryPayload = {
  title: string;
  summary: string;
  tags?: string[];
  messages: StoredChatMessage[];
  createdAt?: Date;
};

export type PromptHistoryRecord = {
  id: string;
  title: string;
  summary: string;
  tags: string[];
  messages: StoredChatMessage[];
  createdAt: Date;
};

const PROMPTS_COLLECTION_KEY = "prompts";

function getPromptsCollection(uid: string) {
  if (!db) {
    throw new Error("Firestore is not available in the current environment.");
  }
  return collection(db, "users", uid, PROMPTS_COLLECTION_KEY);
}

export async function savePromptHistory(uid: string, promptId: string, payload: PromptHistoryPayload) {
  assertDb();
  const ref = doc(getPromptsCollection(uid), promptId);

  const { title, summary, tags, messages, createdAt } = payload;

  await setDoc(
    ref,
    {
      title,
      summary,
      tags: Array.isArray(tags) ? tags : [],
      messages,
      createdAt: createdAt ?? serverTimestamp(),
      updatedAt: serverTimestamp(),
    },
    { merge: true },
  );
}

export async function updateLastSeen(uid: string) {
  assertDb();
  const ref = doc(collection(db, "users"), uid);
  await updateDoc(ref, { lastSeenAt: serverTimestamp() });
}

export function listenToPromptHistory(
  uid: string,
  onData: (records: PromptHistoryRecord[]) => void,
  onError?: (error: Error) => void,
): Unsubscribe {
  assertDb();
  const historyQuery = query(getPromptsCollection(uid), orderBy("createdAt", "desc"));

  return onSnapshot(
    historyQuery,
    (snapshot) => {
      const records = snapshot.docs.map(transformPromptHistoryDocument);
      onData(records);
    },
    (error) => {
      if (onError) {
        onError(error instanceof Error ? error : new Error("Unknown history listener error."));
      } else {
        console.error("Prompt history listener error", error);
      }
    },
  );
}

export async function deletePromptHistoryEntry(uid: string, promptId: string) {
  assertDb();
  const ref = doc(getPromptsCollection(uid), promptId);
  await deleteDoc(ref);
}

export async function clearPromptHistory(uid: string) {
  assertDb();
  const promptsRef = getPromptsCollection(uid);
  const snapshot = await getDocs(promptsRef);

  if (snapshot.empty) {
    return;
  }

  const batch = writeBatch(db);
  snapshot.docs.forEach((docSnapshot) => {
    batch.delete(docSnapshot.ref);
  });
  await batch.commit();
}

function transformPromptHistoryDocument(
  docSnapshot: QueryDocumentSnapshot<DocumentData>,
): PromptHistoryRecord {
  const data = docSnapshot.data();

  return {
    id: docSnapshot.id,
    title: typeof data.title === "string" ? data.title : "Untitled prompt",
    summary: typeof data.summary === "string" ? data.summary : "",
    tags: Array.isArray(data.tags)
      ? data.tags.filter((tag): tag is string => typeof tag === "string")
      : [],
    messages: Array.isArray(data.messages)
      ? data.messages
          .map(normalizeStoredChatMessage)
          .filter((message): message is StoredChatMessage => message !== null)
      : [],
    createdAt: resolveTimestampValue(data.createdAt),
  };
}

function normalizeStoredChatMessage(
  rawMessage: Partial<StoredChatMessage>,
): StoredChatMessage | null {
  if (
    typeof rawMessage.id !== "string" ||
    (rawMessage.role !== "user" && rawMessage.role !== "assistant") ||
    typeof rawMessage.content !== "string" ||
    typeof rawMessage.timestamp !== "string"
  ) {
    return null;
  }

  const attachments = Array.isArray(rawMessage.attachments)
    ? rawMessage.attachments.filter(isValidStoredAttachment)
    : undefined;

  return {
    id: rawMessage.id,
    role: rawMessage.role,
    content: rawMessage.content,
    timestamp: rawMessage.timestamp,
    variant: rawMessage.variant === "accent" || rawMessage.variant === "subtle" ? rawMessage.variant : undefined,
    attachments,
    renderAsCode: rawMessage.renderAsCode === true,
    codeLanguage:
      rawMessage.codeLanguage && typeof rawMessage.codeLanguage === "string"
        ? rawMessage.codeLanguage
        : undefined,
    htmlPath: typeof rawMessage.htmlPath === "string" ? rawMessage.htmlPath : undefined,
    htmlWebPath:
      typeof rawMessage.htmlWebPath === "string" || rawMessage.htmlWebPath === null
        ? rawMessage.htmlWebPath ?? null
        : undefined,
  };
}

function isValidStoredAttachment(
  attachment: Partial<StoredChatAttachment>,
): attachment is StoredChatAttachment {
  return (
    typeof attachment.id === "string" &&
    typeof attachment.name === "string" &&
    (attachment.type === "image" || attachment.type === "file")
  );
}

function resolveTimestampValue(value: unknown): Date {
  if (value instanceof Timestamp) {
    return value.toDate();
  }
  if (value && typeof (value as Timestamp).toDate === "function") {
    return (value as Timestamp).toDate();
  }
  return new Date(0);
}
