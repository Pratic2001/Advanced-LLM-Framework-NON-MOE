import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { getPipelineOrchestrator } from "@/lib/pipeline-orchestrator";
import { pipelineSchema, formatZodErrors } from "@/lib/validations";

export async function POST(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = pipelineSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const { name, backend, stages, nodeIds } = parsed.data;

  const orchestrator = getPipelineOrchestrator();

  const pipelineId = await orchestrator.launch({
    name,
    userId: session.user.id,
    backend,
    stages: stages.map((s) => ({
      type: normalizeStageType(s.type),
      label: s.type,
      config: s.config,
      extraArgs: s.extraArgs,
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
