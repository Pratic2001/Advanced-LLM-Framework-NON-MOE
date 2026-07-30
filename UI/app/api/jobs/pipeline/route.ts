import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";
import { buildCommand } from "@/lib/command-builder";
import type { ScriptName, BackendType } from "@/lib/schema";

export async function POST(req: Request) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const {
    name,
    backend,
    stages,
    nodeIds,
  } = body as {
    name: string;
    backend: BackendType;
    stages: { type: string; config: Record<string, any> }[];
    nodeIds?: string[];
  };

  if (!name || !backend || !stages || stages.length === 0) {
    return NextResponse.json(
      { error: "Name, backend, and at least one stage are required" },
      { status: 400 }
    );
  }

  // Create pipeline record
  const pipeline = await prisma.pipeline.create({
    data: {
      userId: session.user.id,
      name,
      backend: backend.toUpperCase() as any,
      status: "PENDING",
    },
  });

  // Create pipeline stages and initial jobs
  for (let i = 0; i < stages.length; i++) {
    const stage = stages[i];
    const scriptName = getScriptName(stage.type);

    const command = buildCommand({
      script: scriptName,
      config: stage.config,
      backend,
      nodeCount: nodeIds?.length || 1,
      gpuCount: 1,
    });

    const job = await prisma.job.create({
      data: {
        userId: session.user.id,
        type: stage.type.toUpperCase() as any,
        backend: backend.toUpperCase() as any,
        status: i === 0 ? "QUEUED" : "PENDING",
        config: stage.config,
      },
    });

    await prisma.pipelineStage.create({
      data: {
        pipelineId: pipeline.id,
        stageOrder: i,
        jobType: stage.type.toUpperCase(),
        config: stage.config,
        status: i === 0 ? "WAITING" : "PENDING",
        jobId: job.id,
      },
    });
  }

  const fullPipeline = await prisma.pipeline.findUnique({
    where: { id: pipeline.id },
    include: {
      stages: { orderBy: { stageOrder: "asc" } },
    },
  });

  return NextResponse.json(fullPipeline, { status: 201 });
}

function getScriptName(type: string): ScriptName {
  const map: Record<string, ScriptName> = {
    tokenizer: "tokenizer_train.py",
    packing: "hf_to_packed.py",
    pretrain: "train_pretrain.py",
    sft: "train_sft.py",
    grpo: "train_grpo.py",
    dpo: "train_dpo.py",
  };
  return map[type.toLowerCase()] || "train_pretrain.py";
}
