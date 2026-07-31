import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { createMetricSchema, formatZodErrors } from "@/lib/validations";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = await params;

  const { searchParams } = new URL(req.url);
  const limit = Math.min(parseInt(searchParams.get("limit") || "500"), 5000);
  const after = searchParams.get("after")
    ? parseInt(searchParams.get("after")!)
    : undefined;

  const metrics = await prisma.jobMetric.findMany({
    where: {
      jobId,
      ...(after ? { step: { gt: after } } : {}),
    },
    orderBy: { step: "asc" },
    take: limit,
  });

  return NextResponse.json(metrics);
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = await params;

  const body = await req.json();
  const parsed = createMetricSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const { step, loss, vramUsed, tokensPerSec, lr, gradNorm, throughput, nodeName } = parsed.data;

  const metric = await prisma.jobMetric.create({
    data: {
      jobId,
      step,
      loss,
      vramUsed,
      tokensPerSec,
      lr,
      gradNorm,
      throughput,
      nodeName,
    },
  });

  return NextResponse.json(metric, { status: 201 });
}
