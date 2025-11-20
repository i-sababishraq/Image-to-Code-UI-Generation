import { getApp, getApps, initializeApp, type FirebaseApp } from "firebase/app";
import { getAuth, type Auth } from "firebase/auth";
import { getFirestore, type Firestore } from "firebase/firestore";
import { getAnalytics, isSupported, type Analytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? "AIzaSyAxKUxJmc7ZlFKObTAAlsVaPAmVySU_rS0",
  authDomain:
    import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? "image2code-49c1d.firebaseapp.com",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? "image2code-49c1d",
  storageBucket:
    import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? "image2code-49c1d.firebasestorage.app",
  messagingSenderId:
    import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? "1040280825205",
  appId: import.meta.env.VITE_FIREBASE_APP_ID ?? "1:1040280825205:web:a1166d773bc761a5b3f73c",
  measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID ?? "G-HL44GQ5Q00",
};

const firebaseApp: FirebaseApp = getApps().length ? getApp() : initializeApp(firebaseConfig);

const auth: Auth | undefined = typeof window !== "undefined" ? getAuth(firebaseApp) : undefined;
const db: Firestore = getFirestore(firebaseApp);

const analyticsPromise: Promise<Analytics | undefined> =
  typeof window === "undefined"
    ? Promise.resolve(undefined)
    : isSupported()
        .then((supported) => (supported ? getAnalytics(firebaseApp) : undefined))
        .catch(() => undefined);

export { auth, db, firebaseApp as app, analyticsPromise };
