import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";
import { getPipelineOrchestrator } from "@/lib/pipeline-orchestrator";
import type { BackendType } from "@/lib/schema";

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

  const orchestrator = getPipelineOrchestrator();

  const pipelineId = await orchestrator.launch({
    name,
    userId: session.user.id,
    backend,
    stages: stages.map((s) => ({
      type: normalizeStageType(s.type),
      label: s.type,
      config: s.config,
    })),
    nodeIds,
  });

  const pipeline = await prisma.pipeline.findUnique({
    where: { id: pipelineId },
    include: {
      stages: { orderBy: { stageOrder: "asc" } },
    },
  });

  return NextResponse.json(pipeline, { status: 201 });
}

function normalizeStageType(type: string): string {
  const map: Record<string, string> = {
    tokenizer: "TOKENIZER_TRAIN",
    packing: "PACK_PRETRAIN",
    pretrain: "PRETRAIN",
    sft: "SFT",
    grpo: "GRPO",
    dpo: "DPO",
  };
  return map[type.toLowerCase()] || type.toUpperCase();
}
