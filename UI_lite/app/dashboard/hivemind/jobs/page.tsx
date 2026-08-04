"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  PlayCircle,
  BarChart3,
} from "lucide-react";

const jobs: any[] = [];

export default function HivemindJobsPage() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold">Hivemind Jobs</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Monitor decentralized training jobs across all peers
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
            Launch a Hivemind training job to track decentralized progress across all connected peers.
          </p>
          <Link
            href="/dashboard/hivemind/config"
            className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-lg glass text-sm font-medium hover:bg-accent/50 transition-all"
          >
            <PlayCircle className="w-4 h-4" />
            Configure & Launch
          </Link>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {jobs.map((job, i) => (
            <motion.div key={job.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }} className="glass rounded-xl p-5 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold">{job.name}</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">{job.type} • {job.peerCount} peers</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">{job.status}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
