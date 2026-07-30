"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { signOut, useSession } from "next-auth/react";
import { useState } from "react";
import {
  Zap,
  LayoutDashboard,
  Cpu,
  Waves,
  Globe,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Server,
  PlayCircle,
  BarChart3,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type { ReactNode } from "react";

const tabs = [
  {
    id: "torchtab",
    label: "Torch / DDP",
    icon: Cpu,
    href: "/dashboard/torchtab",
    color: "neon-cyan",
    glow: "shadow-[0_0_15px_rgba(0,240,255,0.15)]",
  },
  {
    id: "deepspeed",
    label: "DeepSpeed",
    icon: Waves,
    href: "/dashboard/deepspeed",
    color: "neon-blue",
    glow: "shadow-[0_0_15px_rgba(0,136,255,0.15)]",
  },
  {
    id: "hivemind",
    label: "Hivemind",
    icon: Globe,
    href: "/dashboard/hivemind",
    color: "neon-purple",
    glow: "shadow-[0_0_15px_rgba(124,58,237,0.15)]",
  },
];

const tabSubNav = [
  { id: "setup", label: "Setup", icon: Settings },
  { id: "nodes", label: "Nodes", icon: Server },
  { id: "config", label: "Configure", icon: PlayCircle },
  { id: "jobs", label: "Jobs", icon: BarChart3 },
];

export default function DashboardShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Auth is already enforced by the server-side parent layout. We render a
  // brief loading state only if the client SessionProvider hasn't hydrated
  // yet for the sidebar's user-email display.
  const activeTab = tabs.find((t) => pathname.startsWith(t.href)) || tabs[0];
  const remainingPath = pathname.replace(activeTab.href, "") || "";
  const activeSubNav = tabSubNav.find((s) => remainingPath.startsWith(`/${s.id}`));

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-50 flex flex-col glass border-r border-border/50 transition-all duration-300 ${
          sidebarOpen ? "w-64" : "w-16"
        } ${mobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
      >
        {/* Logo */}
        <div className="h-16 flex items-center gap-3 px-4 border-b border-border/50">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-cyan to-neon-blue flex items-center justify-center shrink-0">
            <Zap className="w-4 h-4 text-white" />
          </div>
          {sidebarOpen && (
            <span className="font-semibold text-base tracking-tight whitespace-nowrap">
              LLM<span className="text-neon-cyan">Forge</span>
            </span>
          )}
        </div>

        {/* Tab Navigation */}
        <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
          {/* Dashboard Home */}
          <Link
            href="/dashboard"
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
              pathname === "/dashboard"
                ? "bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/20"
                : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
            }`}
          >
            <LayoutDashboard className="w-4 h-4 shrink-0" />
            {sidebarOpen && <span>Overview</span>}
          </Link>

          {/* Divider */}
          {sidebarOpen && (
            <div className="px-3 pt-4 pb-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Backends
              </p>
            </div>
          )}

          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = pathname.startsWith(tab.href);
            return (
              <Link
                key={tab.id}
                href={tab.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
                  isActive
                    ? "text-muted-foreground border"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                }`}
                style={
                  isActive
                    ? {
                        background: `rgba(0, 240, 255, 0.05)`,
                        borderColor: `rgba(0, 240, 255, 0.15)`,
                        color: tab.id === "torchtab" ? "#00f0ff" : tab.id === "deepspeed" ? "#0088ff" : "#7c3aed",
                      }
                    : undefined
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                {sidebarOpen && <span>{tab.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* User & Sign Out */}
        <div className="p-2 border-t border-border/50">
          {sidebarOpen && session?.user && (
            <div className="px-3 py-2 mb-1">
              <p className="text-sm font-medium truncate">{session.user.email}</p>
            </div>
          )}
          <button
            onClick={() => signOut({ callbackUrl: "/" })}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-muted-foreground hover:text-red-400 hover:bg-red-500/10 w-full transition-all"
          >
            <LogOut className="w-4 h-4 shrink-0" />
            {sidebarOpen && <span>Sign Out</span>}
          </button>
        </div>

        {/* Collapse button */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="hidden lg:flex items-center justify-center h-8 border-t border-border/50 text-muted-foreground hover:text-foreground"
        >
          {sidebarOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </aside>

      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top navigation */}
        <header className="glass border-b border-border/50 sticky top-0 z-30">
          <div className="flex items-center justify-between h-16 px-4 lg:px-6">
            {/* Mobile menu toggle */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden text-muted-foreground hover:text-foreground"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>

            {/* Active tab + subnav */}
            <div className="flex items-center gap-4 overflow-x-auto">
              {/* Tab buttons */}
              <div className="flex items-center gap-1 bg-background rounded-lg p-1 border border-border/50">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = pathname.startsWith(tab.href);
                  return (
                    <Link
                      key={tab.id}
                      href={tab.href}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all whitespace-nowrap ${
                        isActive
                          ? "bg-accent text-foreground"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      <span className="hidden sm:inline">{tab.label}</span>
                    </Link>
                  );
                })}
              </div>

              {/* Sub-nav */}
              {activeTab && (
                <div className="flex items-center gap-1">
                  {tabSubNav.map((sub) => {
                    const subHref = `${activeTab.href}/${sub.id}`;
                    const isSubActive = pathname === subHref || (sub.id === "setup" && pathname === activeTab.href);
                    return (
                      <Link
                        key={sub.id}
                        href={subHref}
                        className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all whitespace-nowrap ${
                          isSubActive
                            ? "text-foreground bg-accent/50"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {sub.label}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Right side */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="hidden sm:inline">Connected</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-4 lg:p-6 overflow-auto">
          {/*
            Animate page transitions only on top-level tab changes
            (torchtab / deepspeed / hivemind), not on every micro-pathname
            update. Using `key={pathname}` re-mounts the entire page tree
            on any Next.js client-side router normalization (trailing
            slash, query string, etc.), which tears down long-lived
            resources like the InteractiveShell's SSE stream mid-flight.
          */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab.id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.15 }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
