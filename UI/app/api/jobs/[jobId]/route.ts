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

  const job = await prisma.job.findFirst({
    where: { id: params.jobId, userId: session.user.id },
    include: {
      metrics: {
        orderBy: { step: "asc" },
        take: 500,
      },
      node: true,
      parentJob: true,
      childJobs: true,
    },
  });

  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  return NextResponse.json(job);
}
