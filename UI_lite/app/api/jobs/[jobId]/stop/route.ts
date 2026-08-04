import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { JobManager } from "@/lib/job-manager";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = await params;

  const job = await prisma.job.findFirst({
    where: { id: jobId, userId: session.user.id },
  });

  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  if (job.status !== "RUNNING" && job.status !== "QUEUED") {
    return NextResponse.json(
      { error: `Cannot stop job in status: ${job.status}` },
      { status: 400 }
    );
  }

  try {
    const killed = await JobManager.getInstance().stop(jobId);
    if (!killed) {
      return NextResponse.json(
        { error: "Process not found or already terminated" },
        { status: 409 }
      );
    }

    const updatedJob = await prisma.job.findUnique({
      where: { id: jobId },
    });

    return NextResponse.json(updatedJob);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Failed to stop job" },
      { status: 500 }
    );
  }
}
