import "@/lib/auth-env";
import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";

export function useRequiredSession() {
  const { data: session, status } = useSession();

  if (status === "loading") {
    return { session: null, status: "loading" as const };
  }

  if (status === "unauthenticated") {
    redirect("/login");
  }

  return { session, status };
}

export function requireAuth() {
  return async function getSession() {
    const { auth } = await import("./auth");
    const session = await auth();

    if (!session?.user?.id) {
      return null;
    }

    return session;
  };
}
