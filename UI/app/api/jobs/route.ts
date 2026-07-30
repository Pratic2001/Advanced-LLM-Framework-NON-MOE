import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { buildCommand } from "@/lib/command-builder";
import { JobManager } from "@/lib/job-manager";
import type { ScriptName } from "@/lib/schema";
import { createJobSchema, formatZodErrors } from "@/lib/validations";

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const jobs = await prisma.job.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 50,
    include: {
      metrics: {
        orderBy: { step: "desc" },
        take: 1,
      },
      _count: {
        select: { childJobs: true },
      },
    },
  });

  return NextResponse.json(jobs);
}

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = createJobSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const { type, backend, config, nodeIds, extraArgs } = parsed.data;

  // Look up nodes if specified
  let nodes: { host: string; gpuCount: number }[] = [];
  if (nodeIds && nodeIds.length > 0) {
    const nodeRecords = await prisma.node.findMany({
      where: { id: { in: nodeIds }, userId: session.user.id },
    });
    nodes = nodeRecords.map((n) => ({
      host: n.host,
      gpuCount: n.gpuCount || 1,
    }));
  }

  // Build the CLI command
  const command = buildCommand({
    script: type as ScriptName,
    config,
    backend,
    nodes: nodes.map((n) => n.host),
    nodeCount: Math.max(nodes.length, 1),
    gpuCount: nodes[0]?.gpuCount || 1,
    extraArgs,
  });

  // Create the job record (store extraArgs in config for later use)
  const job = await prisma.job.create({
    data: {
      userId: session.user.id,
      type: type.toUpperCase() as any,
      backend: backend.toUpperCase() as any,
      status: "QUEUED",
      config: (extraArgs ? { ...config, __extraArgs: extraArgs } : config) as any,
    },
  });

  // Spawn the subprocess via JobManager. Fire-and-forget: launch() resolves
  // once the subprocess is up and listeners are attached; stdout/stderr are
  // streamed to WebSocket subscribers independently of this response.
  JobManager.getInstance().launch(job.id, command, []).catch((err) => {
    console.error(`[POST /api/jobs] Failed to launch job ${job.id}:`, err);
  });

  return NextResponse.json(job, { status: 201 });
}
