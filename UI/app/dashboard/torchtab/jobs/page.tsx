"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  PlayCircle,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  AlertTriangle,
  BarChart3,
  ExternalLink,
} from "lucide-react";

// Placeholder — in production this data comes from /api/jobs
const jobs: any[] = [];

export default function TorchtabJobsPage() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Jobs</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Monitor and manage training jobs for Torch DDP
        </p>
      </div>

      {jobs.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-10 border border-border/50 text-center"
        >
          <div className="w-16 h-16 rounded-full bg-accent/30 flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-6 h-6 text-muted-foreground" />
          </div>
          <h3 className="font-semibold mb-2">No Jobs Yet</h3>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">
            Launch a training pipeline to see jobs appear here with real-time status, logs, and metrics.
          </p>
          <Link
            href="/dashboard/torchtab/config"
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
          >
            <PlayCircle className="w-4 h-4" />
            Configure & Launch
          </Link>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job, i) => (
            <motion.div
              key={job.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass rounded-xl p-5 border border-border/50"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[hsl(var(--palette-primary))] to-[hsl(var(--palette-secondary))] flex items-center justify-center shrink-0">
                    <BarChart3 className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">{job.name}</h3>
                      <span className="text-xs px-2 py-0.5 rounded-full bg-accent/30 text-muted-foreground font-mono">
                        {job.type}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Started {job.startedAt}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 text-xs">
                    {job.status === "RUNNING" && (
                      <Loader2 className="w-3.5 h-3.5 text-neon-cyan animate-spin" />
                    )}
                    {job.status === "COMPLETED" && (
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                    )}
                    {job.status === "FAILED" && (
                      <XCircle className="w-3.5 h-3.5 text-red-400" />
                    )}
                    {job.status === "PENDING" && (
                      <Clock className="w-3.5 h-3.5 text-yellow-400" />
                    )}
                    <span className="text-muted-foreground">{job.status}</span>
                  </div>
                  <Link
                    href={`/dashboard/torchtab/jobs/${job.id}`}
                    className="text-muted-foreground hover:text-foreground transition-all"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
