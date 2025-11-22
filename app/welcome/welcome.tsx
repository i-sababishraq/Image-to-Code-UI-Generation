import {
  type ChangeEvent,
  type FormEvent,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Navigate } from "react-router";

import { useAuth } from "../auth/AuthProvider";
import {
  clearPromptHistory,
  deletePromptHistoryEntry,
  listenToPromptHistory,
  savePromptHistory,
  type PromptHistoryPayload,
  type PromptHistoryRecord,
  type StoredChatAttachment,
  type StoredChatMessage,
} from "../services/userData";

type TabKey = "chat" | "preview" | "history" | "files";

type NavItem = {
  key: TabKey;
  label: string;
  description: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: ChatAttachment[];
  variant?: "accent" | "subtle";
  timestamp: string;
  renderAsCode?: boolean;
  codeLanguage?: string;
  htmlPath?: string;
  htmlWebPath?: string | null;
};

type HistoryEntry = {
  id: string;
  title: string;
  summary: string;
  timestamp: string;
  tags: string[];
  messages: Message[];
  createdAt: Date;
};

type RenderedComponentPreview = {
  id: string;
  title: string;
  html: string;
  createdAt: Date;
  previewUrl?: string | null;
  htmlPath?: string;
  htmlWebPath?: string | null;
};

type WireframePreview = {
  id: string;
  url: string;
  name: string;
  size: number;
};

type PexelsImageResult = {
  id: number;
  description: string;
  photographer: string;
  previewUrl: string;
  downloadUrl: string;
  sourceUrl: string;
  width: number;
  height: number;
};

const HISTORY_TITLE_MAX_LENGTH = 80;
const HISTORY_SUMMARY_MAX_LENGTH = 140;

const formatHistoryEntryTimestamp = (date: Date) => {
  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
  }

  const timePart = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const targetDay = new Date(date);
  targetDay.setHours(0, 0, 0, 0);

  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);

  if (targetDay.getTime() === today.getTime()) {
    return `Today · ${timePart}`;
  }
  if (targetDay.getTime() === yesterday.getTime()) {
    return `Yesterday · ${timePart}`;
  }

  const datePart = date.toLocaleDateString([], { month: "short", day: "numeric" });
  return `${datePart} · ${timePart}`;
};

const truncateText = (value: string, limit: number) => {
  if (!value) {
    return "";
  }
  return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
};

const createHistoryTitle = (prompt: string) => {
  const normalized = prompt.trim();
  if (!normalized) {
    return "Untitled prompt";
  }
  return truncateText(normalized, HISTORY_TITLE_MAX_LENGTH);
};

const createHistorySummary = (assistantMessages: Message[]) => {
  const primary = assistantMessages[0]?.content ?? "No response recorded.";
  return truncateText(primary.trim() || "No response recorded.", HISTORY_SUMMARY_MAX_LENGTH);
};

const deriveHistoryTagsFromFiles = (files: File[]) => {
  if (!files || files.length === 0) {
    return ["Prompt"];
  }

  const tags = new Set<string>();

  if (files.some((file) => file.type.startsWith("image/"))) {
    tags.add("Images");
  }
  if (files.some((file) => !file.type.startsWith("image/"))) {
    tags.add("Files");
  }

  if (tags.size === 0) {
    tags.add("Prompt");
  }

  return Array.from(tags);
};

const serializeMessageForHistory = (message: Message): StoredChatMessage => {
  const serialized: StoredChatMessage = {
    id: message.id,
    role: message.role,
    content: message.content,
    timestamp: message.timestamp,
  };

  if (message.variant) {
    serialized.variant = message.variant;
  }

  if (message.attachments && message.attachments.length > 0) {
    const normalizedAttachments: StoredChatAttachment[] = message.attachments.map((attachment) => {
      const normalized: StoredChatAttachment = {
        id: attachment.id,
        name: attachment.name,
        type: attachment.type,
      };

      if (typeof attachment.size === "number" && Number.isFinite(attachment.size)) {
        normalized.size = attachment.size;
      }

      return normalized;
    });

    serialized.attachments = normalizedAttachments;
  }

  if (message.renderAsCode) {
    serialized.renderAsCode = true;
  }

  if (message.codeLanguage) {
    serialized.codeLanguage = message.codeLanguage;
  }

  if (message.htmlPath) {
    serialized.htmlPath = message.htmlPath;
  }

  if (message.htmlWebPath) {
    serialized.htmlWebPath = message.htmlWebPath;
  }

  return serialized;
};

const deserializeStoredMessage = (message: StoredChatMessage): Message => ({
  id: message.id,
  role: message.role,
  content: message.content,
  timestamp: message.timestamp,
  variant: message.variant,
  attachments: message.attachments?.map((attachment) => ({
    id: attachment.id,
    name: attachment.name,
    type: attachment.type,
    size: attachment.size,
  })),
  renderAsCode: message.renderAsCode,
  codeLanguage: message.codeLanguage,
});

const mapPromptHistoryRecordToEntry = (record: PromptHistoryRecord): HistoryEntry => ({
  id: record.id,
  title: record.title,
  summary: record.summary,
  timestamp: formatHistoryEntryTimestamp(record.createdAt),
  tags: record.tags,
  messages: record.messages.map(deserializeStoredMessage),
  createdAt: record.createdAt,
});

type AgentMessageMapOptions = {
  mode?: "default" | "friendly-code";
};

const CODE_SNIPPET_PATTERN =
  /```|<\/?[a-z][^>]*>|class\s+\w+|function\s+\w+|const\s+\w+|import\s+|export\s+/i;

const hasHtmlIndicators = (value: string) =>
  /<!doctype|<html|<body|<header|<section|<main|<footer|<script|<div|<nav/i.test(value);

const isLikelyCodeSnippet = (value: string) => {
  if (!value) {
    return false;
  }
  return CODE_SNIPPET_PATTERN.test(value) || hasHtmlIndicators(value);
};

const KNOWN_FRIENDLY_INTRO =
  "I'm implementing the plan from your wireframe. Here's the latest code output.";

const stripMarkdownFence = (value: string) => value.replace(/```[a-zA-Z0-9]*\s*/g, "").trim();

const trimAgentPreface = (value: string) => {
  const lowered = value.toLowerCase();
  if (lowered.startsWith("generated python snippet:")) {
    return value.slice(value.indexOf(":") + 1).trim();
  }
  if (lowered.startsWith("generated html:")) {
    return value.slice(value.indexOf(":") + 1).trim();
  }
  return value.trim();
};

const extractCodeContent = (value: string) => {
  const withoutFence = stripMarkdownFence(value);
  const trimmed = trimAgentPreface(withoutFence);
  const docIndex = trimmed.indexOf("<!DOCTYPE");
  const htmlIndex = trimmed.indexOf("<html");
  const snippetStart =
    docIndex !== -1 ? docIndex : htmlIndex !== -1 ? htmlIndex : trimmed.indexOf("<");
  if (snippetStart !== -1) {
    return trimmed.slice(snippetStart).trim();
  }
  return trimmed;
};

const detectCodeLanguage = (value: string): string | undefined => {
  const sample = value.slice(0, 200).toLowerCase();
  if (sample.includes("<!doctype") || sample.includes("<html")) {
    return "html";
  }
  if (sample.includes("<svg")) {
    return "svg";
  }
  if (sample.includes("class ") || sample.includes("def ") || sample.includes("import ")) {
    return "python";
  }
  return undefined;
};

const extractThinkingSteps = (agentMessages?: AgentChatMessagePayload[]) => {
  if (!agentMessages || agentMessages.length === 0) {
    return [];
  }
  return agentMessages
    .map((message) => message.content.trim())
    .filter((content) => content.length > 0 && !isLikelyCodeSnippet(content));
};

const mapAgentPayloadToMessages = (
  agentMessages?: AgentChatMessagePayload[],
  options?: AgentMessageMapOptions,
  timestampOverride?: string,
): Message[] => {
  const timestamp = timestampOverride ?? formatTimestamp(new Date());

  if (!agentMessages || agentMessages.length === 0) {
    return [];
  }

  if (options?.mode === "friendly-code") {
    const [, ...rest] = agentMessages;
    const hasAdditionalResponses = rest.length > 0;
    const candidateSources = hasAdditionalResponses ? rest : agentMessages;
    const filteredCodeSources = candidateSources.filter((message) =>
      isLikelyCodeSnippet(message.content),
    );
    const codeSources = filteredCodeSources.length > 0 ? filteredCodeSources : candidateSources;
    const codeContentCandidates = codeSources
      .map((message) => extractCodeContent(message.content))
      .filter((snippet) => snippet.length > 0);
    const combinedCode = codeContentCandidates.join("\n\n").trim();
    const codeLanguage = detectCodeLanguage(combinedCode);
    const htmlSourceWithPath = codeSources.find(
      (message) => message.htmlPath || message.htmlWebPath,
    );
    const previewSource = agentMessages.find(
      (message) => Array.isArray(message.attachments) && message.attachments.length > 0,
    );
    const previewAttachments: ChatAttachment[] | undefined = previewSource?.attachments
      ? previewSource.attachments.map((attachment) => ({
        id: attachment.id,
        name: attachment.name,
        type: attachment.type,
        previewUrl: attachment.previewUrl,
        size: attachment.size,
      }))
      : undefined;

    const codeMessage: Message = {
      id: createMessageId(),
      role: "assistant",
      variant: "accent",
      content:
        combinedCode.length > 0
          ? combinedCode
          : "Image2Code did not return any code for this run. Try refining your prompt and uploading a clearer wireframe.",
      timestamp,
      renderAsCode: true,
      codeLanguage,
      htmlPath: htmlSourceWithPath?.htmlPath,
      htmlWebPath: htmlSourceWithPath?.htmlWebPath,
      attachments: previewAttachments,
    };

    return [
      {
        id: createMessageId(),
        role: "assistant",
        variant: "accent",
        content: KNOWN_FRIENDLY_INTRO,
        timestamp,
      },
      codeMessage,
    ];
  }

  return agentMessages.map((message) => ({
    id: createMessageId(),
    role: "assistant",
    variant: message.variant,
    content: message.content,
    timestamp,
    attachments:
      message.attachments && message.attachments.length > 0
        ? message.attachments.map((attachment) => ({
          id: attachment.id ?? createMessageId(),
          name: attachment.name,
          type: attachment.type,
          previewUrl: attachment.previewUrl,
          size: attachment.size,
        }))
        : undefined,
  }));
};

const createInstructionFallbackMessage = (): Message => ({
  id: createMessageId(),
  role: "assistant",
  variant: "accent",
  content: FALLBACK_INSTRUCTIONS,
  timestamp: formatTimestamp(new Date()),
});

const summarizeAgentLog = (log: string, maxItems = 4): string => {
  const segments = log
    .split(/(?=Stage|Asset resolved|HTML saved|Best score|Error|->)/)
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length === 0) {
    return log.length > 400 ? `${log.slice(0, 400)}…` : log;
  }
  const highlights = segments.slice(0, maxItems).map((item) => `• ${item.replace(/\s+/g, " ")}`);
  return ["Agent summary:", ...highlights].join("\n");
};

const buildPromptHistoryPayload = (
  userMessage: Message,
  assistantMessages: Message[],
  files: File[],
): PromptHistoryPayload => ({
  title: createHistoryTitle(userMessage.content),
  summary: createHistorySummary(assistantMessages),
  tags: deriveHistoryTagsFromFiles(files),
  messages: [userMessage, ...assistantMessages].map(serializeMessageForHistory),
});

const mapEntryToRenderedComponent = (
  entry: HistoryEntry,
): RenderedComponentPreview | null => {
  const codeMessage = entry.messages.find((message) => message.renderAsCode && message.content);

  if (!codeMessage) {
    return null;
  }

  return {
    id: entry.id,
    title: entry.title,
    html: codeMessage.content,
    createdAt: entry.createdAt,
    previewUrl: codeMessage.htmlWebPath ?? null,
    htmlPath: codeMessage.htmlPath,
    htmlWebPath: codeMessage.htmlWebPath ?? null,
  };
};

const releasePreviewUrls = (urls: string[]) => {
  urls.forEach((url) => URL.revokeObjectURL(url));
  urls.length = 0;
};

const buildWireframePreviews = (
  files: File[],
  registerUrl: (url: string) => void,
): WireframePreview[] =>
  files
    .filter((file) => file.type.startsWith("image/"))
    .map((file) => {
      const url = URL.createObjectURL(file);
      registerUrl(url);
      return {
        id: `${file.name}-${file.lastModified}-${Math.random().toString(16).slice(2)}`,
        url,
        name: file.name,
        size: file.size,
      };
    });

type StatusMessage = {
  kind: "success" | "error";
  text: string;
  detail?: string;
};

type AttachmentType = "image" | "file";

type ChatAttachment = {
  id: string;
  name: string;
  type: AttachmentType;
  previewUrl?: string;
  size?: number;
};

type AgentChatMessagePayload = {
  id: string;
  role: "assistant";
  variant: "accent" | "subtle";
  content: string;
  attachments?: ChatAttachment[];
  htmlPath?: string;
  htmlWebPath?: string | null;
};

type AgentRunResponse = {
  messages?: AgentChatMessagePayload[];
  status?: {
    kind: "success" | "error";
    text: string;
    detail?: string;
  };
  usedFallback?: boolean;
  finalHtmlPath?: string;
  finalHtmlWebPath?: string | null;
};

const VLM_AGENT_ENDPOINT = "/api/vlm";

const formatTimestamp = (date: Date) =>
  date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

const createMessageId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

{ key: "chat", label: "Chat", description: "Live" },
{ key: "preview", label: "Preview", description: "Generated UI" },
{ key: "history", label: "History", description: "Previous runs" },
{ key: "files", label: "Files", description: "Output assets" },
];

const IMAGE2CODE_ONBOARDING_PROMPT = [
  "You are Image2Code, an AI assistant that turns uploaded wireframes and natural language prompts into UI code.",
  "The user just opened the product and needs concise onboarding tips.",
  "Provide 3-4 short, actionable bullet points that explain how to capture wireframes, add textual prompts, and iterate to refine results.",
  "Be encouraging, avoid asking questions, and keep the focus on how to interact with Image2Code effectively.",
].join(" ");

const FALLBACK_INSTRUCTIONS = [
  "Welcome to Image2Code! Here's how to get the best results:",
  "• Upload one or more wireframe screenshots (PNG/JPG/WebP) so I can study the layout.",
  "• Describe the behavior, platform, and any edge cases directly in the prompt field.",
  "• Iterate by tweaking your prompt or swapping in new images—I remember prior context during a run.",
  "• Use the Preview and History tabs to inspect generated code or revisit prior explorations.",
].join("\n");

const AGENT_REQUEST_TIMEOUT_MS = 3_600_000; // 6 minutes

export function Welcome() {
  const { user, loading, signOut } = useAuth();

  const [activeTab, setActiveTab] = useState<TabKey>("chat");
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [promptValue, setPromptValue] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [historyItems, setHistoryItems] = useState<HistoryEntry[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(true);
  const [showClearHistoryConfirm, setShowClearHistoryConfirm] = useState(false);
  const [isClearingHistory, setIsClearingHistory] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [renderedComponents, setRenderedComponents] = useState<RenderedComponentPreview[]>([]);
  const [isBootstrappingAssistant, setIsBootstrappingAssistant] = useState(true);
  const [thinkingEntries, setThinkingEntries] = useState<string[]>([]);
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const generatedObjectUrlsRef = useRef<string[]>([]);
  const previewObjectUrlsRef = useRef<string[]>([]);
  const submittedPreviewUrlsRef = useRef<string[]>([]);
  const livePreviewRef = useRef<RenderedComponentPreview | null>(null);
  const [pendingWireframes, setPendingWireframes] = useState<WireframePreview[]>([]);
  const [submittedWireframes, setSubmittedWireframes] = useState<WireframePreview[]>([]);
  const [lastSubmittedFiles, setLastSubmittedFiles] = useState<File[]>([]);
  const onboardingRequestedRef = useRef(false);
  const [pexelsQuery, setPexelsQuery] = useState("");
  const [pexelsResult, setPexelsResult] = useState<PexelsImageResult | null>(null);
  const [isPexelsSearching, setIsPexelsSearching] = useState(false);
  const [isPexelsDownloading, setIsPexelsDownloading] = useState(false);
  const [isPexelsDownloading, setIsPexelsDownloading] = useState(false);
  const [pexelsError, setPexelsError] = useState<string | null>(null);
  const [refineMaxIters, setRefineMaxIters] = useState(3);
  const displayName = user?.displayName ?? user?.email ?? "Anonymous";
  const inspirationPreviews =
    pendingWireframes.length > 0 ? pendingWireframes : submittedWireframes;
  const showingPendingWireframes = pendingWireframes.length > 0;
  const persistSubmittedWireframes = useCallback((files: File[]) => {
    releasePreviewUrls(submittedPreviewUrlsRef.current);
    const previews = buildWireframePreviews(files, (url) =>
      submittedPreviewUrlsRef.current.push(url),
    );
    setSubmittedWireframes(previews);
  }, []);

  const handlePexelsQueryChange = useCallback(
    (value: string) => {
      setPexelsQuery(value);
      if (pexelsError) {
        setPexelsError(null);
      }
    },
    [pexelsError],
  );

  const runPexelsSearch = useCallback(async () => {
    const trimmedQuery = pexelsQuery.trim();
    if (!trimmedQuery) {
      setPexelsError("Describe the layout or asset you want to search on Pexels.");
      setPexelsResult(null);
      return;
    }

    setIsPexelsSearching(true);
    setPexelsError(null);

    try {
      const response = await fetch("/api/pexels", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: trimmedQuery }),
      });

      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const message =
          (payload && typeof payload.error === "string" && payload.error) ||
          "Unable to fetch inspiration from Pexels.";
        throw new Error(message);
      }

      if (payload?.photo) {
        setPexelsResult(payload.photo as PexelsImageResult);
      } else {
        setPexelsResult(null);
        setPexelsError("No matches were found for that description.");
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected error contacting Pexels.";
      setPexelsResult(null);
      setPexelsError(message);
    } finally {
      setIsPexelsSearching(false);
    }
  }, [pexelsQuery]);

  const handlePexelsDownload = useCallback(async () => {
    if (!pexelsResult) {
      setPexelsError("Search for an image before downloading.");
      return;
    }
    if (typeof window === "undefined" || typeof document === "undefined") {
      return;
    }

    setIsPexelsDownloading(true);
    setPexelsError(null);

    try {
      const filenameBase = buildPexelsFilename(pexelsResult.description);
      const params = new URLSearchParams({
        imageUrl: pexelsResult.downloadUrl,
        filename: filenameBase,
      });
      const response = await fetch(`/api/pexels?${params.toString()}`);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Unable to download the selected image.");
      }
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const disposition = response.headers.get("Content-Disposition");
      const finalName =
        extractFilenameFromDisposition(disposition) ?? `${filenameBase}.jpg`;

      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = finalName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unexpected download error.";
      setPexelsError(message);
    } finally {
      setIsPexelsDownloading(false);
    }
  }, [pexelsResult]);

  const registerObjectUrls = (urls: string[]) => {
    if (urls.length === 0) {
      return;
    }
    generatedObjectUrlsRef.current.push(...urls);
  };

  const releaseGeneratedObjectUrls = () => {
    generatedObjectUrlsRef.current.forEach((url) => {
      URL.revokeObjectURL(url);
    });
    generatedObjectUrlsRef.current = [];
  };

  const requestAssistantIntro = useCallback(async () => {
    setIsBootstrappingAssistant(true);

    const requestPayload = new FormData();
    requestPayload.append("prompt", IMAGE2CODE_ONBOARDING_PROMPT);
    requestPayload.append("skipImageRequirement", "true");

    try {
      const response = await fetch(VLM_AGENT_ENDPOINT, {
        method: "POST",
        body: requestPayload,
      });

      let agentResponse: AgentRunResponse | null = null;
      try {
        agentResponse = (await response.json()) as AgentRunResponse;
      } catch (parseError) {
        console.warn("Unable to parse agent response JSON", parseError);
      }

      if (!response.ok) {
        const message =
          agentResponse?.status?.text ?? `Agent request failed with status ${response.status}.`;
        throw new Error(message);
      }

      const statusInfo = agentResponse?.status;
      const isErrorStatus = statusInfo?.kind === "error";

      const assistantMessages = !isErrorStatus
        ? mapAgentPayloadToMessages(agentResponse?.messages)
        : [];

      setChatMessages((current) => {
        if (current.length > 0) {
          return current;
        }
        if (assistantMessages.length > 0) {
          return assistantMessages;
        }
        return [createInstructionFallbackMessage()];
      });

      if (isErrorStatus) {
        setStatus({
          kind: "error",
          text: statusInfo?.text ?? "Unable to load Image2Code instructions.",
          detail: statusInfo?.detail,
        });
      } else {
        setStatus((previous) =>
          previous ?? {
            kind: "success",
            text: "Image2Code shared a few tips to get you started.",
            detail: statusInfo?.detail,
          },
        );
      }
    } catch (error) {
      console.error("Failed to fetch onboarding instructions", error);
      setStatus({
        kind: "error",
        text: "Unable to load Image2Code instructions.",
        detail: error instanceof Error ? error.message : "Unknown onboarding error.",
      });
      setChatMessages((current) =>
        current.length > 0 ? current : [createInstructionFallbackMessage()],
      );
    } finally {
      setIsBootstrappingAssistant(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      releaseGeneratedObjectUrls();
    };
  }, []);

  useEffect(() => {
    return () => {
      releasePreviewUrls(submittedPreviewUrlsRef.current);
      releasePreviewUrls(previewObjectUrlsRef.current);
    };
  }, []);

  useEffect(() => {
    releasePreviewUrls(previewObjectUrlsRef.current);
    const previews = buildWireframePreviews(selectedFiles, (url) =>
      previewObjectUrlsRef.current.push(url),
    );
    setPendingWireframes(previews);

    return () => {
      releasePreviewUrls(previewObjectUrlsRef.current);
    };
  }, [selectedFiles]);

  useEffect(() => {
    if (onboardingRequestedRef.current) {
      return;
    }
    onboardingRequestedRef.current = true;
    void requestAssistantIntro();
  }, [requestAssistantIntro]);

  useEffect(() => {
    if (!user) {
      setHistoryItems([]);
      setIsHistoryLoading(false);
      return;
    }

    setIsHistoryLoading(true);
    let unsubscribe: (() => void) | undefined;

    try {
      unsubscribe = listenToPromptHistory(
        user.uid,
        (records) => {
          setHistoryItems(records.map(mapPromptHistoryRecordToEntry));
          setIsHistoryLoading(false);
        },
        (error) => {
          console.error("Failed to load prompt history", error);
          setIsHistoryLoading(false);
          setStatus({
            kind: "error",
            text: "Unable to load chat history.",
            detail: error.message,
          });
        },
      );
    } catch (error) {
      console.error("Failed to subscribe to prompt history", error);
      setIsHistoryLoading(false);
      setStatus({
        kind: "error",
        text: "Unable to start chat history listener.",
        detail: error instanceof Error ? error.message : "Unknown error.",
      });
    }

    return () => {
      unsubscribe?.();
    };
  }, [user?.uid]);

  useEffect(() => {
    setRenderedComponents((previous) => {
      const derived =
        historyItems
          .map(mapEntryToRenderedComponent)
          .filter((component): component is RenderedComponentPreview => component !== null) ?? [];

      const pending = livePreviewRef.current;
      if (pending) {
        const exists = derived.some(
          (component) =>
            component.html.trim() === pending.html.trim() ||
            (component.id === pending.id && component.title === pending.title),
        );
        if (!exists) {
          return [pending, ...derived];
        }
        livePreviewRef.current = null;
      }

      return derived;
    });
  }, [historyItems]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const textContent = promptValue.trim();

    if (!textContent) {
      setStatus({
        kind: "error",
        text: "Add a prompt describing what you want generated.",
      });
      return;
    }

    const filesToSend = selectedFiles.length > 0 ? selectedFiles : lastSubmittedFiles;

    if (filesToSend.length === 0) {
      setStatus({
        kind: "error",
        text: "Upload a wireframe before sending your first request.",
      });
      return;
    }

    setIsSending(true);
    setStatus(null);

    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    const abortTimeoutId =
      controller && typeof window !== "undefined"
        ? window.setTimeout(() => controller.abort(), AGENT_REQUEST_TIMEOUT_MS)
        : null;
    const clearAbortTimeout = () => {
      if (abortTimeoutId !== null && typeof window !== "undefined") {
        window.clearTimeout(abortTimeoutId);
      }
    };

    try {
      const createdUrls: string[] = [];
      const attachments: ChatAttachment[] = filesToSend.map((file) => {
        const isImage = file.type.startsWith("image/");
        let previewUrl: string | undefined;

        if (isImage) {
          previewUrl = URL.createObjectURL(file);
          createdUrls.push(previewUrl);
        }

        return {
          id: createMessageId(),
          name: file.name,
          type: isImage ? "image" : "file",
          previewUrl,
          size: file.size,
        };
      });

      if (createdUrls.length > 0) {
        registerObjectUrls(createdUrls);
      }

      persistSubmittedWireframes(filesToSend);
      setLastSubmittedFiles([...filesToSend]);

      const messageContent =
        textContent || (attachments.length > 0 ? "Shared design attachments" : "");

      const userMessage: Message = {
        id: createMessageId(),
        role: "user",
        content: messageContent,
        attachments: attachments.length > 0 ? attachments : undefined,
        timestamp: formatTimestamp(new Date()),
      };

      setChatMessages((previous) => [...previous, userMessage]);

      const requestPayload = new FormData();
      requestPayload.append("prompt", textContent);
      requestPayload.append("refine_max_iters", refineMaxIters.toString());
      filesToSend.forEach((file) => {
        requestPayload.append("images", file);
      });

      try {
        const response = await fetch(VLM_AGENT_ENDPOINT, {
          method: "POST",
          body: requestPayload,
          signal: controller?.signal,
        });
        clearAbortTimeout();

        let agentResponse: AgentRunResponse | null = null;
        try {
          agentResponse = (await response.json()) as AgentRunResponse;
        } catch (parseError) {
          console.warn("Unable to parse agent response JSON", parseError);
        }

        if (!response.ok) {
          const message =
            agentResponse?.status?.text ?? `Agent request failed with status ${response.status}.`;
          throw new Error(message);
        }

        const assistantMessages = mapAgentPayloadToMessages(agentResponse?.messages, {
          mode: "friendly-code",
        }).map((message) => {
          if (message.variant === "subtle" && message.content.length > 400) {
            return {
              ...message,
              content: summarizeAgentLog(message.content, 4),
            };
          }
          return message;
        });
        const thinkingSteps = extractThinkingSteps(agentResponse?.messages);
        if (thinkingSteps.length > 0) {
          setThinkingEntries(thinkingSteps);
          setIsThinkingExpanded(false);
        } else {
          setThinkingEntries([]);
        }

        if (assistantMessages.length > 0) {
          setChatMessages((previous) => [...previous, ...assistantMessages]);
        }

        const codeMessage = assistantMessages.find((message) => message.renderAsCode);
        if (codeMessage) {
          const previewUrl = agentResponse?.finalHtmlWebPath ?? codeMessage.htmlWebPath ?? null;
          const component: RenderedComponentPreview = {
            id: `live-${codeMessage.id}`,
            title: createHistoryTitle(userMessage.content),
            html: codeMessage.content,
            createdAt: new Date(),
            previewUrl,
            htmlPath: agentResponse?.finalHtmlPath ?? codeMessage.htmlPath,
            htmlWebPath: previewUrl,
          };
          livePreviewRef.current = component;
          setRenderedComponents((previous) => [component, ...previous]);
        }

        if (assistantMessages.length > 0 && user) {
          const payload = buildPromptHistoryPayload(userMessage, assistantMessages, filesToSend);
          void (async () => {
            try {
              await savePromptHistory(user.uid, createMessageId(), payload);
            } catch (historyError) {
              console.error("Failed to save prompt history", historyError);
              setStatus((previous) => {
                const detailMessage =
                  historyError instanceof Error ? historyError.message : "Unknown history error.";

                if (previous && previous.kind === "success") {
                  return {
                    ...previous,
                    detail: previous.detail
                      ? `${previous.detail} History sync failed: ${detailMessage}`
                      : `History sync failed: ${detailMessage}`,
                  };
                }

                return {
                  kind: "error",
                  text: "Generated response but failed to sync chat history.",
                  detail: detailMessage,
                };
              });
            }
          })();
        }

        if (agentResponse?.status) {
          const detail =
            agentResponse.status.detail ??
            (agentResponse.usedFallback
              ? "Python agent unavailable. Served fallback response."
              : undefined);
          setStatus({
            kind: agentResponse.status.kind,
            text: agentResponse.status.text,
            detail,
          });
        } else {
          setStatus({
            kind: "success",
            text: "Agent responded successfully.",
          });
        }
      } catch (error) {
        clearAbortTimeout();
        console.error("Failed to send message to VLM agent", error);
        const isAbortError = error instanceof DOMException && error.name === "AbortError";
        const detail = isAbortError
          ? "The agent request took longer than expected and was cancelled."
          : error instanceof Error
            ? error.message
            : "Unknown error contacting agent.";

        setThinkingEntries([]);
        setIsThinkingExpanded(false);

        setStatus({
          kind: "error",
          text: isAbortError
            ? "We cancelled this run because the agent timed out."
            : "We had trouble reaching the agent. Please try again shortly.",
          detail,
        });

        const errorMessage: Message = {
          id: createMessageId(),
          role: "assistant",
          variant: "subtle",
          content:
            "We couldn't reach the agent service. None of your context was lost—try again once the connection is back.",
          timestamp: formatTimestamp(new Date()),
        };
        setChatMessages((previous) => [...previous, errorMessage]);
      }
    } catch (error) {
      console.error("Failed to prepare agent request payload", error);
      setThinkingEntries([]);
      setIsThinkingExpanded(false);
      setStatus({
        kind: "error",
        text: "We couldn't prepare your request.",
        detail: error instanceof Error ? error.message : "Unknown client-side error.",
      });
    } finally {
      setPromptValue("");
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setIsSending(false);
      clearAbortTimeout();
    }
  };

  const handleAddImageClick = () => {
    if (isSending) {
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files ? Array.from(event.target.files) : [];
    setSelectedFiles(files);

    if (files.length > 0) {
      setStatus({
        kind: "success",
        text: `${files.length} attachment${files.length > 1 ? "s" : ""} ready to send.`,
      });
    }
  };

  const handleRemoveAttachment = (fileToRemove: File) => {
    setSelectedFiles((previous) => {
      const updated = previous.filter((file) => file !== fileToRemove);

      if (fileInputRef.current) {
        if (typeof DataTransfer !== "undefined") {
          const dataTransfer = new DataTransfer();
          updated.forEach((file) => dataTransfer.items.add(file));
          fileInputRef.current.files = dataTransfer.files;
        } else {
          fileInputRef.current.value = "";
        }
      }

      if (updated.length === 0) {
        if (lastSubmittedFiles.length > 0) {
          setStatus({
            kind: "success",
            text: "No new attachments selected. We'll reuse your latest wireframes.",
          });
        } else {
          setStatus({
            kind: "error",
            text: "You removed all attachments. Add at least one wireframe before sending.",
          });
        }
      }

      return updated;
    });
  };

  const handleClearHistory = () => {
    if (historyItems.length === 0) {
      setStatus({
        kind: "error",
        text: "History is already cleared.",
      });
      return;
    }

    setShowClearHistoryConfirm(true);
  };

  const handleConfirmClearHistory = async () => {
    if (!user) {
      setShowClearHistoryConfirm(false);
      return;
    }

    setIsClearingHistory(true);
    try {
      await clearPromptHistory(user.uid);
      setStatus({
        kind: "success",
        text: "Cleared previous runs from history.",
      });
    } catch (error) {
      console.error("Failed to clear prompt history", error);
      setStatus({
        kind: "error",
        text: "Unable to clear chat history.",
        detail: error instanceof Error ? error.message : "Unknown error clearing history.",
      });
    } finally {
      setIsClearingHistory(false);
      setShowClearHistoryConfirm(false);
    }
  };

  const handleDeleteHistoryEntry = async (entryId: string) => {
    if (!user) {
      return;
    }

    setPendingDeleteId(entryId);
    try {
      await deletePromptHistoryEntry(user.uid, entryId);
      setStatus({
        kind: "success",
        text: "Removed chat from history.",
      });
    } catch (error) {
      console.error("Failed to delete prompt history entry", error);
      setStatus({
        kind: "error",
        text: "Unable to remove chat from history.",
        detail: error instanceof Error ? error.message : "Unknown error deleting history entry.",
      });
    } finally {
      setPendingDeleteId((current) => (current === entryId ? null : current));
    }
  };

  const handleViewConversation = (entry: HistoryEntry) => {
    if (entry.messages.length === 0) {
      setStatus({
        kind: "error",
        text: "This history item does not include any chat messages yet.",
      });
      return;
    }

    releaseGeneratedObjectUrls();
    setChatMessages(entry.messages);
    setActiveTab("chat");
    setStatus({
      kind: "success",
      text: `Loaded the "${entry.title}" conversation into chat.`,
    });
  };

  const renderContent = () => {
    switch (activeTab) {
      case "preview":
        return (
          <PreviewPanel components={renderedComponents} />
        );
      case "history":
        return (
          <HistoryPanel
            entries={historyItems}
            isLoading={isHistoryLoading}
            onClearHistory={handleClearHistory}
            isClearingHistory={isClearingHistory}
            onViewConversation={handleViewConversation}
            onDeleteEntry={handleDeleteHistoryEntry}
            deletingEntryId={pendingDeleteId}
          />
        );
      case "files":
        return <FilesPanel />;
      case "chat":
      default:
        return (
          <ChatPanel
            messages={chatMessages}
            onSubmit={handleSubmit}
            onAddImageClick={handleAddImageClick}
            fileInputRef={fileInputRef}
            promptValue={promptValue}
            onPromptChange={setPromptValue}
            isSending={isSending}
            onFileChange={handleFileChange}
            selectedFiles={selectedFiles}
            onRemoveAttachment={handleRemoveAttachment}
            isBootstrappingAssistant={isBootstrappingAssistant}
            wireframePreviews={inspirationPreviews}
            hasPendingWireframes={showingPendingWireframes}
            hasStoredWireframes={lastSubmittedFiles.length > 0}
            pexelsQuery={pexelsQuery}
            onPexelsQueryChange={handlePexelsQueryChange}
            onPexelsSearch={runPexelsSearch}
            isPexelsSearching={isPexelsSearching}
            pexelsResult={pexelsResult}
            pexelsError={pexelsError}
            onPexelsDownload={handlePexelsDownload}
            isPexelsDownloading={isPexelsDownloading}
            refineMaxIters={refineMaxIters}
            onRefineMaxItersChange={setRefineMaxIters}
          />
        );
    }
  };

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-100">
        <div className="text-center">
          <div className="mb-3 animate-spin rounded-full border-4 border-[#2F6BFF] border-t-transparent p-6" />
          <p className="text-sm text-slate-400">Loading your workspace…</p>
        </div>
      </main>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-8 px-4 py-6 sm:px-8 lg:px-12">
        <header className="flex flex-col gap-6 border-b border-slate-800/80 pb-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-lg font-semibold sm:text-xl text-[#6FA3FF]">Image2Code</p>
            <p className="text-sm text-slate-400">
              Bring wireframes to life with an agentic workflow.
            </p>
          </div>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
            <nav className="grid grid-cols-3 gap-2 sm:max-w-sm">
              {navItems.map((item) => {
                const isActive = item.key === activeTab;
                return (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setActiveTab(item.key)}
                    aria-pressed={isActive}
                    className={`rounded-2xl border px-3 py-2 text-left text-sm font-medium transition-all duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6BFF] ${isActive
                        ? "border-[#2F6BFF] bg-slate-900/70 text-[#D6E2FF]"
                        : "border-slate-800/70 bg-slate-900/50 text-slate-500 hover:border-[#2F6BFF]/40 hover:text-slate-200"
                      }`}
                  >
                    <span className="block text-sm font-semibold">{item.label}</span>
                    <span className="text-xs font-normal text-slate-500">
                      {item.description}
                    </span>
                  </button>
                );
              })}
            </nav>
            <div className="flex items-center justify-between gap-3 rounded-2xl border border-slate-800/70 bg-slate-900/60 px-4 py-2 text-sm text-slate-300 sm:w-auto">
              <span className="truncate">{displayName}</span>
              <button
                type="button"
                onClick={() => void signOut()}
                className="rounded-lg border border-slate-700 px-3 py-1 text-xs font-medium text-slate-200 transition hover:border-rose-500 hover:text-rose-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-400"
              >
                Sign out
              </button>
            </div>
          </div>
        </header>

        {status && (
          <div
            className={`rounded-2xl border px-4 py-3 text-sm ${status.kind === "success"
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
                : "border-rose-500/40 bg-rose-500/10 text-rose-100"
              }`}
          >
            <p className="font-medium">{status.text}</p>
            {status.detail && <p className="mt-1 text-xs opacity-80">{status.detail}</p>}
          </div>
        )}

        {thinkingEntries.length > 0 && (
          <ThinkingPanel
            entries={thinkingEntries}
            isExpanded={isThinkingExpanded}
            onToggle={() => setIsThinkingExpanded((previous) => !previous)}
          />
        )}

        {renderContent()}
        {showClearHistoryConfirm && (
          <ConfirmDialog
            title="Clear chat history?"
            description="This will permanently delete every saved conversation. This action cannot be undone."
            confirmLabel="Yes, clear history"
            cancelLabel="No, keep history"
            onConfirm={handleConfirmClearHistory}
            onCancel={() => {
              if (!isClearingHistory) {
                setShowClearHistoryConfirm(false);
              }
            }}
            isProcessing={isClearingHistory}
          />
        )}
      </div>
    </main>
  );
}

type ChatPanelProps = {
  messages: Message[];
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onAddImageClick: () => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
  promptValue: string;
  onPromptChange: (value: string) => void;
  isSending: boolean;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  selectedFiles: File[];
  onRemoveAttachment: (file: File) => void;
  isBootstrappingAssistant: boolean;
  wireframePreviews: WireframePreview[];
  hasPendingWireframes: boolean;
  hasStoredWireframes: boolean;
  pexelsQuery: string;
  onPexelsQueryChange: (value: string) => void;
  onPexelsSearch: () => void;
  isPexelsSearching: boolean;
  pexelsResult: PexelsImageResult | null;
  pexelsError: string | null;
  onPexelsDownload: () => void;
  isPexelsDownloading: boolean;
  refineMaxIters: number;
  onRefineMaxItersChange: (value: number) => void;
};

function ChatPanel({
  messages,
  onSubmit,
  onAddImageClick,
  fileInputRef,
  promptValue,
  onPromptChange,
  isSending,
  onFileChange,
  selectedFiles,
  onRemoveAttachment,
  isBootstrappingAssistant,
  wireframePreviews,
  hasPendingWireframes,
  hasStoredWireframes,
  pexelsQuery,
  onPexelsQueryChange,
  onPexelsSearch,
  isPexelsSearching,
  pexelsResult,
  pexelsError,
  onPexelsDownload,
  isPexelsDownloading,
  refineMaxIters,
  onRefineMaxItersChange,
}: ChatPanelProps) {
  return (
    <section className="grid gap-6 lg:grid-cols-[0.65fr_minmax(0,_1fr)]">
      <ImageGenerationPanel
        query={pexelsQuery}
        onQueryChange={onPexelsQueryChange}
        onSearch={onPexelsSearch}
        isSearching={isPexelsSearching}
        result={pexelsResult}
        error={pexelsError}
        onDownload={onPexelsDownload}
        isDownloading={isPexelsDownloading}
        wireframePreviews={wireframePreviews}
        hasPendingWireframes={hasPendingWireframes}
      />

      <div className="flex min-h-[480px] flex-col rounded-3xl border border-slate-800/70 bg-slate-900/85 shadow-[0_30px_80px_-45px_rgba(15,23,42,1)]">
        <div className="flex items-center justify-between border-b border-slate-800/70 px-6 py-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-300">Chat Canvas</h2>
            <p className="text-xs text-slate-500">
              Prototype conversations with your build assistant.
            </p>
          </div>
          <span className="rounded-full border border-emerald-500/30 bg-emerald-900/40 px-3 py-1 text-xs font-medium text-emerald-300">
            Connected
          </span>
        </div>

        <ul className="flex flex-1 flex-col gap-5 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <li className="flex flex-1 items-center justify-center rounded-3xl border border-dashed border-slate-800/70 bg-slate-900/40 px-4 py-8 text-center text-sm text-slate-400">
              {isBootstrappingAssistant
                ? "Checking in with Image2Code for quick tips…"
                : "Upload a wireframe image and describe the UI you need to start the conversation."}
            </li>
          ) : (
            messages.map((message) => <MessageBubble key={message.id} message={message} />)
          )}
        </ul>

        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-3 border-t border-slate-800/70 bg-slate-900/80 px-4 py-4 sm:flex-row sm:items-center"
        >
          <div className="flex-1 space-y-2">
            <label className="flex flex-col">
              <span className="sr-only">Prompt the assistant</span>
              <textarea
                className="h-16 w-full resize-none rounded-2xl border border-slate-800 bg-slate-950/60 px-4 py-3 text-sm leading-relaxed text-slate-100 shadow-inner outline-none transition focus:border-[#2F6BFF]/60 focus:ring-2 focus:ring-[#2F6BFF]/30 placeholder:text-slate-500 disabled:cursor-not-allowed disabled:opacity-70"
                placeholder="Describe the UI you want, or paste feedback to iterate."
                value={promptValue}
                onChange={(event) => onPromptChange(event.target.value)}
                disabled={isSending}
              />
            </label>
            {selectedFiles.length > 0 && (
              <ul className="flex flex-wrap gap-2">
                {selectedFiles.map((file) => (
                  <li
                    key={`${file.name}-${file.lastModified}`}
                    className="flex items-center gap-2 rounded-2xl border border-slate-800/70 bg-slate-950/70 px-3 py-1.5 text-xs text-slate-200"
                  >
                    <span className="max-w-[160px] truncate">{file.name}</span>
                    <button
                      type="button"
                      onClick={() => onRemoveAttachment(file)}
                      className="rounded-full border border-slate-700 bg-slate-900 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-slate-400 transition hover:border-rose-500/40 hover:text-rose-200"
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {selectedFiles.length === 0 && hasStoredWireframes && (
              <p className="text-xs text-slate-500">
                Using wireframes from your previous run.
              </p>
            )}
            {selectedFiles.length === 0 && hasStoredWireframes && (
              <p className="text-xs text-slate-500">
                Using wireframes from your previous run.
              </p>
            )}
            <label className="flex items-center gap-2 pt-2">
              <span className="text-xs font-medium text-slate-400">Refinement Iterations:</span>
              <input
                type="number"
                min={1}
                max={10}
                value={refineMaxIters}
                onChange={(e) => onRefineMaxItersChange(parseInt(e.target.value, 10) || 3)}
                className="w-16 rounded-lg border border-slate-800 bg-slate-950 px-2 py-1 text-xs text-slate-200 focus:border-[#2F6BFF]/60 focus:outline-none"
              />
            </label>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={onFileChange}
          />
          <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={onAddImageClick}
              aria-label="Add images"
              disabled={isSending}
              className="flex h-10 w-10 items-center justify-center rounded-2xl border border-slate-800 bg-slate-900 text-slate-200 transition hover:border-slate-700 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5">
                <path d="M10 4a1 1 0 011 1v4h4a1 1 0 110 2h-4v4a1 1 0 11-2 0v-4H5a1 1 0 110-2h4V5a1 1 0 011-1z" />
              </svg>
            </button>
            <button
              type="submit"
              disabled={
                isSending ||
                (promptValue.trim().length === 0 && selectedFiles.length === 0 && !hasStoredWireframes)
              }
              className="flex items-center gap-2 rounded-xl bg-[#2F6BFF] px-4 py-2 text-sm font-semibold text-white shadow-[0_14px_32px_-18px_rgba(47,107,255,0.9)] transition hover:bg-[#2A5FE6] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6BFF] disabled:cursor-not-allowed disabled:bg-[#2F6BFF]/60"
            >
              {isSending ? "Sending..." : "Send"}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4">
                <path d="M2.94 2.94a.75.75 0 01.78-.18l13 5a.75.75 0 010 1.38l-13 5A.75.75 0 012 13.5v-3.318a.75.75 0 01.553-.725L9.5 8.001l-6.947-1.456A.75.75 0 012 5.82V2.5a.75.75 0 01.22-.53z" />
                <path d="M18 10a.75.75 0 00-.75-.75h-7.5a.75.75 0 000 1.5h7.5A.75.75 0 0018 10z" />
              </svg>
            </button>
          </div>
        </form>
        {isSending && (
          <p
            role="status"
            aria-live="polite"
            className="mx-4 mb-4 flex items-center gap-3 rounded-2xl border border-slate-800/60 bg-slate-900/70 px-4 py-3 text-xs font-medium text-slate-300 animate-pulse"
          >
            <span className="inline-flex h-2 w-2 animate-ping rounded-full bg-[#2F6BFF]" />
            Agent is still running. Bigger generations can take a minute—keep this tab open while we finish.
          </p>
        )}
      </div>
    </section>
  );
}

type ImageGenerationPanelProps = {
  query: string;
  onQueryChange: (value: string) => void;
  onSearch: () => void;
  isSearching: boolean;
  result: PexelsImageResult | null;
  error: string | null;
  onDownload: () => void;
  isDownloading: boolean;
  wireframePreviews: WireframePreview[];
  hasPendingWireframes: boolean;
};

function ImageGenerationPanel({
  query,
  onQueryChange,
  onSearch,
  isSearching,
  result,
  error,
  onDownload,
  isDownloading,
  wireframePreviews,
  hasPendingWireframes,
}: ImageGenerationPanelProps) {
  return (
    <aside className="hidden flex-col gap-5 rounded-3xl border border-slate-800/70 bg-slate-900/80 p-6 shadow-[0_18px_60px_-40px_rgba(15,23,42,1)] lg:flex">
      <div className="space-y-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-300">Image inspiration</h2>
          <p className="text-xs text-slate-500">Use the Pexels API to grab hero art or UI mood boards.</p>
        </div>
        <form
          className="space-y-2"
          onSubmit={(event) => {
            event.preventDefault();
            onSearch();
          }}
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              placeholder="e.g. neon fintech dashboard"
              className="flex-1 rounded-xl border border-slate-800/70 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-[#2F6BFF]/40 focus:outline-none focus:ring-2 focus:ring-[#2F6BFF]/20"
              disabled={isSearching}
            />
            <button
              type="submit"
              className="rounded-xl bg-[#2F6BFF] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#1f4dc5] disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isSearching}
            >
              {isSearching ? "Searching…" : "Search"}
            </button>
          </div>
          <p className="text-xs text-slate-500">
            Describe the vibe of the interface you want. We’ll fetch a matching hero image from Pexels.
          </p>
          {error && <p className="text-xs font-medium text-rose-300">{error}</p>}
        </form>

        {result ? (
          <div className="space-y-3 rounded-2xl border border-slate-800/60 bg-slate-950/40 p-3">
            <figure className="relative aspect-video w-full overflow-hidden rounded-xl border border-slate-800/70">
              <img src={result.previewUrl} alt={result.description} className="h-full w-full object-cover" />
              <span className="absolute right-3 top-3 rounded-full bg-slate-950/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-200">
                Pexels
              </span>
            </figure>
            <div className="space-y-1 text-xs text-slate-400">
              <p className="text-sm font-semibold text-slate-200">{result.description}</p>
              <p>
                {result.photographer} · {result.width}×{result.height}
              </p>
              <a
                href={result.sourceUrl}
                target="_blank"
                rel="noreferrer"
                className="text-[#6FA3FF] underline-offset-2 hover:underline"
              >
                View on Pexels
              </a>
            </div>
            <button
              type="button"
              onClick={onDownload}
              className="w-full rounded-xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-100 transition hover:border-[#2F6BFF]/50 hover:text-[#D6E2FF] disabled:cursor-not-allowed disabled:opacity-40"
              disabled={isDownloading || isSearching}
            >
              {isDownloading ? "Preparing download…" : "Download image"}
            </button>
          </div>
        ) : (
          <div className="rounded-2xl border border-dashed border-slate-800/60 bg-slate-950/30 p-4 text-xs text-slate-500">
            No image selected yet. Describe a layout, style, or color palette to get instant reference art.
          </div>
        )}
      </div>

      <div className="space-y-3 rounded-2xl border border-slate-800/70 bg-slate-950/30 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Wireframe tray</h3>
        <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-slate-800/70 bg-slate-900/60">
          {wireframePreviews.length === 0 ? (
            <div className="flex h-full w-full items-center justify-center text-xs font-medium text-slate-500">
              Upload wireframes to preview here
            </div>
          ) : (
            <>
              <img
                src={wireframePreviews[0]?.url}
                alt={wireframePreviews[0]?.name}
                className="h-full w-full object-cover"
              />
              <div className="absolute left-3 top-3 rounded-full bg-slate-950/70 px-3 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-200">
                {wireframePreviews[0]?.name}
              </div>
            </>
          )}
        </div>
        {wireframePreviews.length > 1 && (
          <div className="grid grid-cols-3 gap-2">
            {wireframePreviews.slice(1, 4).map((preview) => (
              <figure
                key={preview.id}
                className="aspect-square overflow-hidden rounded-xl border border-slate-800/60 bg-slate-900/60"
              >
                <img src={preview.url} alt={preview.name} className="h-full w-full object-cover" />
              </figure>
            ))}
          </div>
        )}
        <p className="text-xs leading-relaxed text-slate-400">
          {wireframePreviews.length === 0
            ? "Drop a mobile or desktop mockup. We'll analyze layout, colors, and behavior before writing code."
            : hasPendingWireframes
              ? "These are queued for your next run. Send your prompt when you're ready."
              : "Showing wireframes from your latest run. Upload updated images to iterate further."}
        </p>
      </div>
    </aside>
  );
}

function buildPexelsFilename(value: string) {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return normalized || "pexels-image";
}

function extractFilenameFromDisposition(header: string | null): string | null {
  if (!header) {
    return null;
  }
  const match = /filename\*?=(?:UTF-8'')?\"?([^\";]+)/i.exec(header);
  return match?.[1] ?? null;
}

type PreviewPanelProps = {
  components: RenderedComponentPreview[];
};

function PreviewPanel({ components }: PreviewPanelProps) {
  if (components.length === 0) {
    return (
      <section className="rounded-3xl border border-slate-800/70 bg-slate-900/80 p-8 text-center text-sm text-slate-400 shadow-[0_24px_70px_-50px_rgba(15,23,42,1)]">
        <p>No UI components have been generated yet.</p>
        <p className="mt-2 text-xs text-slate-500">
          Send a prompt with a wireframe to populate this gallery.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      <header>
        <h2 className="text-sm font-semibold text-slate-200">Component gallery</h2>
        <p className="text-xs text-slate-500">
          Each card renders the exact HTML returned by Image2Code for previous runs.
        </p>
      </header>
      <div className="grid gap-6 md:grid-cols-2">
        {components.map((component) => (
          <article
            key={component.id}
            className="flex flex-col gap-4 rounded-3xl border border-slate-800/70 bg-slate-900/75 p-5 shadow-[0_24px_70px_-60px_rgba(15,23,42,1)]"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-100">{component.title}</p>
                <p className="text-xs text-slate-500">
                  {component.createdAt.toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </p>
              </div>
              <button
                type="button"
                onClick={() => openComponentInNewTab(component)}
                className="rounded-full border border-slate-700 px-4 py-1.5 text-xs font-semibold text-slate-200 transition hover:border-[#2F6BFF]/50 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6BFF]"
              >
                Open full view
              </button>
            </div>
            <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
              {component.previewUrl ? (
                <iframe
                  title={`Preview for ${component.title}`}
                  src={component.previewUrl}
                  sandbox="allow-scripts allow-same-origin allow-forms"
                  className="h-[420px] w-full"
                />
              ) : (
                <iframe
                  title={`Preview for ${component.title}`}
                  srcDoc={component.html}
                  sandbox="allow-scripts allow-same-origin allow-forms"
                  className="h-[420px] w-full"
                />
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

type HistoryPanelProps = {
  entries: HistoryEntry[];
  isLoading: boolean;
  onClearHistory: () => void;
  isClearingHistory: boolean;
  onViewConversation: (entry: HistoryEntry) => void;
  onDeleteEntry: (entryId: string) => void;
  deletingEntryId: string | null;
};

function HistoryPanel({
  entries,
  isLoading,
  onClearHistory,
  isClearingHistory,
  onViewConversation,
  onDeleteEntry,
  deletingEntryId,
}: HistoryPanelProps) {
  return (
    <section className="flex flex-col gap-6">
      <header className="flex flex-col gap-2 rounded-3xl border border-slate-800/70 bg-slate-900/80 px-6 py-5 shadow-[0_24px_70px_-50px_rgba(15,23,42,1)] sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-300">Run history</h2>
          <p className="text-xs text-slate-500">
            Revisit previous generations and restore code with a single click.
          </p>
        </div>
        <button
          type="button"
          onClick={onClearHistory}
          disabled={entries.length === 0 || isClearingHistory || isLoading}
          className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-[#2F6BFF]/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isClearingHistory ? "Clearing…" : "Clear history"}
        </button>
      </header>

      {isLoading ? (
        <div className="rounded-3xl border border-slate-800/70 bg-slate-900/70 px-5 py-6 text-sm text-slate-400">
          Loading previous runs…
        </div>
      ) : entries.length === 0 ? (
        <div className="rounded-3xl border border-slate-800/70 bg-slate-900/70 px-5 py-6 text-sm text-slate-400">
          No previous runs yet. Send a prompt to populate your history.
        </div>
      ) : (
        <ul className="space-y-4">
          {entries.map((entry) => (
            <li
              key={entry.id}
              className="flex flex-col gap-3 rounded-3xl border border-slate-800/70 bg-slate-900/70 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex-1">
                <p className="text-sm font-semibold text-slate-200">{entry.title}</p>
                <p className="text-xs text-slate-500">{entry.summary}</p>
                <p className="text-xs text-slate-600">{entry.timestamp}</p>
              </div>
              {entry.tags.length > 0 && (
                <div className="flex flex-wrap items-center gap-2">
                  {entry.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-slate-800/70 px-3 py-1 text-[11px] font-medium uppercase tracking-wide text-slate-400"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <button
                  type="button"
                  onClick={() => onViewConversation(entry)}
                  disabled={entry.messages.length === 0}
                  className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-[#2F6BFF]/40 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  View conversation
                </button>
                <button
                  type="button"
                  onClick={() => onDeleteEntry(entry.id)}
                  disabled={deletingEntryId === entry.id}
                  className="rounded-xl border border-slate-800/70 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-rose-500/40 hover:text-rose-200 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {deletingEntryId === entry.id ? "Removing…" : "Remove"}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

type ConfirmDialogProps = {
  title: string;
  description: string;
  confirmLabel: string;
  cancelLabel: string;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
  isProcessing?: boolean;
};

function ConfirmDialog({
  title,
  description,
  confirmLabel,
  cancelLabel,
  onConfirm,
  onCancel,
  isProcessing = false,
}: ConfirmDialogProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 px-4 py-6">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-2xl"
      >
        <h3 id="confirm-dialog-title" className="text-lg font-semibold text-slate-100">
          {title}
        </h3>
        <p className="mt-2 text-sm text-slate-400">{description}</p>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={isProcessing}
            className="rounded-xl border border-slate-700 px-4 py-2 text-sm font-medium text-slate-200 transition hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isProcessing}
            className="rounded-xl bg-rose-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-rose-400 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-300 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isProcessing ? "Clearing…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

type ThinkingPanelProps = {
  entries: string[];
  isExpanded: boolean;
  onToggle: () => void;
};

function ThinkingPanel({ entries, isExpanded, onToggle }: ThinkingPanelProps) {
  return (
    <section className="rounded-3xl border border-slate-800/70 bg-slate-900/70 px-5 py-4 text-sm text-slate-300 shadow-[0_18px_60px_-50px_rgba(15,23,42,1)]">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between text-left font-semibold text-slate-200 transition hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6BFF]"
      >
        <span>Thinking log</span>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-180" : ""}`}
        >
          <path
            fillRule="evenodd"
            d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>
      {isExpanded && (
        <ol className="mt-4 space-y-3 text-sm text-slate-400">
          {entries.map((entry, index) => (
            <li key={`${entry.slice(0, 20)}-${index}`}>
              <span className="mr-2 text-xs font-semibold text-slate-500">{index + 1}.</span>
              <span className="whitespace-pre-line text-slate-300">{entry}</span>
            </li>
          ))}
        </ol>
      )}
      {!isExpanded && (
        <p className="mt-3 text-xs text-slate-500">
          Toggle to review the agent&apos;s intermediate steps for this run.
        </p>
      )}
    </section>
  );
}

function openComponentInNewTab(component: RenderedComponentPreview) {
  const resolvedUrl = resolvePreviewUrl(component);
  if (resolvedUrl) {
    window.open(resolvedUrl, "_blank", "noopener,noreferrer");
    return;
  }
  const blob = new Blob([component.html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank", "noopener,noreferrer");
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

type MessageBubbleProps = {
  message: Message;
};

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isCode = message.renderAsCode === true;
  const bubbleBase =
    "max-w-[min(520px,100%)] rounded-3xl px-5 py-4 text-sm leading-relaxed shadow-sm sm:text-base";

  const bubbleTone = isUser
    ? "bg-slate-800 text-slate-100 border border-slate-700"
    : isCode
      ? "bg-slate-950 text-slate-100 border border-slate-800 px-0 py-0"
      : message.variant === "subtle"
        ? "bg-slate-800/70 text-slate-300 border border-slate-800"
        : "bg-[#2F6BFF] text-white shadow-[0_20px_40px_-24px_rgba(47,107,255,1)]";

  const avatarTone = isUser
    ? "bg-slate-800 text-[#7DA6FF]"
    : message.variant === "accent"
      ? "bg-[#2F6BFF] text-white"
      : "bg-slate-800 text-slate-200";

  const containerDirection = isUser ? "flex-row-reverse" : "flex-row";
  const alignment = isUser ? "items-end" : "items-start";
  const nameLabel = isUser ? "You" : "Image2Code";

  return (
    <li className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex ${containerDirection} ${alignment} gap-3`}>
        <div
          className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-sm font-semibold ${avatarTone}`}
        >
          {isUser ? "UX" : "I2C"}
        </div>
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
            {nameLabel} · {message.timestamp}
          </span>
          <div className={`${bubbleBase} ${bubbleTone}`}>
            {isCode ? (
              <>
                <CodePreview content={message.content} language={message.codeLanguage} />
                <AttachmentGallery attachments={message.attachments} />
              </>
            ) : (
              <>
                <p className="whitespace-pre-wrap">{message.content}</p>
                <AttachmentGallery attachments={message.attachments} />
              </>
            )}
          </div>
        </div>
      </div>
    </li>
  );
}

function AttachmentGallery({ attachments }: { attachments?: ChatAttachment[] }) {
  if (!attachments || attachments.length === 0) {
    return null;
  }

  return (
    <ul className="mt-4 flex flex-wrap gap-3 text-sm font-medium">
      {attachments.map((attachment) => (
        <li key={attachment.id}>
          {attachment.type === "image" && attachment.previewUrl ? (
            <figure className="group relative flex h-28 w-32 overflow-hidden rounded-xl border border-slate-800/80 bg-slate-900/60">
              <img
                src={attachment.previewUrl}
                alt={attachment.name}
                className="h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
              />
              <figcaption className="absolute bottom-0 w-full bg-slate-950/80 px-2 py-1 text-[11px] font-medium text-slate-200 backdrop-blur">
                {attachment.name}
              </figcaption>
            </figure>
          ) : (
            <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2 text-[13px] text-slate-200">
              <FileGlyph />
              <span className="max-w-[140px] truncate" title={attachment.name}>
                {attachment.name}
              </span>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function resolvePreviewUrl(component: RenderedComponentPreview): string | null {
  const candidate =
    component.previewUrl ?? component.htmlWebPath ?? deriveWebPathFromFile(component.htmlPath);

  if (!candidate) {
    return null;
  }

  try {
    return new URL(candidate, window.location.origin).toString();
  } catch {
    return null;
  }
}

function deriveWebPathFromFile(filePath?: string | null): string | null {
  if (!filePath) {
    return null;
  }
  const normalized = filePath.replace(/\\/g, "/");
  const marker = "/Image-to-Code-UI-Generation/";
  const markerIndex = normalized.indexOf(marker);
  if (markerIndex === -1) {
    return null;
  }
  return normalized.slice(markerIndex);
}

function CodePreview({ content, language }: { content: string; language?: string }) {
  const [copied, setCopied] = useState(false);
  const normalized = content || "// No code available";
  const lines = normalized.split("\n");
  const digits = String(lines.length).length;
  const highlighted = useMemo(
    () => buildHighlightedMarkup(normalized, language),
    [normalized, language],
  );

  const handleCopy = async () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      return;
    }
    try {
      await navigator.clipboard.writeText(normalized);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-900 bg-slate-950 shadow-[0_25px_60px_-45px_rgba(15,23,42,1)]">
      <div className="flex items-center justify-between border-b border-slate-800/70 bg-slate-950/80 px-4 py-2 text-xs uppercase tracking-wide text-slate-400">
        <span>{language ?? "code"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="rounded-full border border-slate-700 px-3 py-1 text-[11px] font-semibold text-slate-200 transition hover:border-[#2F6BFF]/50 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2F6BFF]"
        >
          {copied ? "Copied!" : "Copy code"}
        </button>
      </div>
      <div className="flex">
        <pre className="bg-slate-950/70 px-4 py-4 text-xs font-mono leading-6 text-slate-600">
          {lines.map((_, index) => (
            <span key={`line-${index}`} className="block text-right tabular-nums">
              {String(index + 1).padStart(digits, " ")}
            </span>
          ))}
        </pre>
        <pre className="flex-1 overflow-auto px-4 py-4 text-xs font-mono leading-6 text-slate-100">
          <code
            className="block whitespace-pre text-left"
            dangerouslySetInnerHTML={{ __html: highlighted }}
          />
        </pre>
      </div>
    </div>
  );
}

function buildHighlightedMarkup(value: string, language?: string) {
  const escaped = escapeCodeHtml(value);
  if (language === "html") {
    return highlightHtmlSyntax(escaped);
  }
  return escaped;
}

function escapeCodeHtml(value: string) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function highlightHtmlSyntax(escaped: string) {
  let result = escaped.replace(
    /(&lt;!--[\s\S]*?--&gt;)/g,
    '<span class="text-slate-500">$1</span>',
  );

  result = result.replace(
    /(&lt;\/?)([a-zA-Z0-9:-]+)([^&]*?)(\/?&gt;)/g,
    (_match, open, tag, attrs, close) => {
      const highlightedAttrs = attrs.replace(
        /([a-zA-Z-:]+)=(&quot;[^&]*?&quot;)/g,
        '<span class="text-amber-200">$1</span>=<span class="text-emerald-200">$2</span>',
      );
      return `<span class="text-slate-500">${open}</span><span class="text-sky-300">${tag}</span>${highlightedAttrs}<span class="text-slate-500">${close}</span>`;
    },
  );

  return result;
}

    </svg >
  );
}

type FileNode = {
  name: string;
  isDirectory: boolean;
  path: string;
  size: number;
};

function FilesPanel() {
  const [files, setFiles] = useState<FileNode[]>([]);
  const [currentPath, setCurrentPath] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchFiles = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
      if (!response.ok) throw new Error("Failed to fetch files");
      const data = await response.json();
      setFiles(data.files);
      setCurrentPath(path);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchFiles("");
  }, [fetchFiles]);

  const handleNavigate = (path: string) => {
    void fetchFiles(path);
  };

  const handleDownload = (path: string) => {
    window.open(`/api/files?path=${encodeURIComponent(path)}`, "_blank");
  };

  const handleUpLevel = () => {
    if (!currentPath) return;
    const parts = currentPath.split("/");
    parts.pop();
    void fetchFiles(parts.join("/"));
  };

  return (
    <section className="flex flex-col gap-6 rounded-3xl border border-slate-800/70 bg-slate-900/80 px-6 py-5 shadow-[0_24px_70px_-50px_rgba(15,23,42,1)]">
      <header>
        <h2 className="text-sm font-semibold text-slate-300">Output Files</h2>
        <p className="text-xs text-slate-500">Browse generated assets and logs.</p>
      </header>

      <div className="flex items-center gap-2 text-sm text-slate-400">
        <button
          onClick={() => void fetchFiles("")}
          className="hover:text-slate-200"
          disabled={loading}
        >
          root
        </button>
        {currentPath.split("/").filter(Boolean).map((part, index, arr) => {
          const path = arr.slice(0, index + 1).join("/");
          return (
            <div key={path} className="flex items-center gap-2">
              <span>/</span>
              <button
                onClick={() => void fetchFiles(path)}
                className="hover:text-slate-200"
                disabled={loading}
              >
                {part}
              </button>
            </div>
          );
        })}
      </div>

      {loading ? (
        <div className="text-sm text-slate-500">Loading...</div>
      ) : error ? (
        <div className="text-sm text-rose-400">{error}</div>
      ) : (
        <ul className="space-y-2">
          {currentPath && (
            <li>
              <button
                onClick={handleUpLevel}
                className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200"
              >
                <span>..</span>
              </button>
            </li>
          )}
          {files.map((file) => (
            <li key={file.path} className="flex items-center justify-between rounded-lg border border-slate-800/50 bg-slate-950/30 px-3 py-2 transition hover:bg-slate-900">
              <div className="flex items-center gap-3">
                <span className="text-slate-500">
                  {file.isDirectory ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5 text-amber-500/70">
                      <path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5 text-slate-400">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                  )}
                </span>
                {file.isDirectory ? (
                  <button
                    onClick={() => handleNavigate(file.path)}
                    className="text-sm font-medium text-slate-200 hover:underline"
                  >
                    {file.name}
                  </button>
                ) : (
                  <span className="text-sm text-slate-300">{file.name}</span>
                )}
              </div>
              {!file.isDirectory && (
                <button
                  onClick={() => handleDownload(file.path)}
                  className="rounded px-2 py-1 text-xs font-medium text-[#6FA3FF] hover:bg-[#2F6BFF]/10"
                >
                  Download
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
