import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";
import { buildCommand } from "@/lib/command-builder";
import type { ScriptName, BackendType } from "@/lib/schema";

export async function GET() {
  const session = await getServerSession();
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
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const {
    type,
    backend,
    config,
    nodeIds,
    extraArgs,
  } = body as {
    type: string;
    backend: BackendType;
    config: Record<string, any>;
    nodeIds?: string[];
    extraArgs?: string;
  };

  if (!type || !backend) {
    return NextResponse.json(
      { error: "Type and backend are required" },
      { status: 400 }
    );
  }

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
      config: extraArgs ? { ...config, __extraArgs: extraArgs } : config,
    },
  });

  // In production: spawn the subprocess via job-manager
  // const process = await JobManager.getInstance().spawn(job.id, command);

  return NextResponse.json(job, { status: 201 });
}
