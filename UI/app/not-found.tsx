import Link from "next/link";
import { FileX, Home } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="glass-strong rounded-2xl p-8 max-w-md w-full text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto">
          <FileX className="w-7 h-7 text-muted-foreground" />
        </div>
        <h1 className="text-xl font-bold">Page not found</h1>
        <p className="text-sm text-muted-foreground">
          The page you are looking for does not exist or has been moved.
        </p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-neon-cyan to-neon-blue text-black font-semibold text-sm hover:opacity-90 transition-all"
        >
          <Home className="w-4 h-4" />
          Go home
        </Link>
      </div>
    </div>
  );
}
