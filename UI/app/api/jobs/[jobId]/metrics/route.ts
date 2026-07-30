import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";

export async function GET(
  req: Request,
  { params }: { params: { jobId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(req.url);
  const limit = Math.min(parseInt(searchParams.get("limit") || "500"), 5000);
  const after = searchParams.get("after")
    ? parseInt(searchParams.get("after")!)
    : undefined;

  const metrics = await prisma.jobMetric.findMany({
    where: {
      jobId: params.jobId,
      ...(after ? { step: { gt: after } } : {}),
    },
    orderBy: { step: "asc" },
    take: limit,
  });

  return NextResponse.json(metrics);
}

export async function POST(
  req: Request,
  { params }: { params: { jobId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { step, loss, vramUsed, tokensPerSec, lr, gradNorm, throughput, nodeName } = body;

  const metric = await prisma.jobMetric.create({
    data: {
      jobId: params.jobId,
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
