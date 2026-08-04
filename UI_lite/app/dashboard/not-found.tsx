import Link from "next/link";
import { FileX, LayoutDashboard } from "lucide-react";

export default function DashboardNotFound() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="glass-strong rounded-2xl p-8 max-w-md w-full text-center space-y-4">
        <div className="w-14 h-14 rounded-full bg-muted flex items-center justify-center mx-auto">
          <FileX className="w-7 h-7 text-muted-foreground" />
        </div>
        <h1 className="text-xl font-bold">Page not found</h1>
        <p className="text-sm text-muted-foreground">
          This dashboard page does not exist.
        </p>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] text-white font-semibold text-sm hover:opacity-90 transition-all glow-primary"
        >
          <LayoutDashboard className="w-4 h-4" />
          Back to Dashboard
        </Link>
      </div>
    </div>
  );
}
