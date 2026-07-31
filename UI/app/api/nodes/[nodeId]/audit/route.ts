import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ nodeId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { nodeId } = await params;

  const node = await prisma.node.findFirst({
    where: { id: nodeId, userId: session.user.id },
    include: { sshKey: true },
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  // Mark as auditing
  await prisma.node.update({
    where: { id: nodeId },
    data: { status: "AUDITING", errorMessage: null },
  });

  try {
    // In production: use ssh-manager to connect and run audit commands
    // For now, return simulated hardware specs
    const auditResult = {
      gpuName: "NVIDIA RTX 4090",
      gpuCount: 1,
      vramGb: 24,
      cpuCores: 16,
      cpuRamGb: 32,
      pythonVersion: "3.11.5",
      cudaVersion: "12.4",
    };

    // Update node with audit results
    const updatedNode = await prisma.node.update({
      where: { id: nodeId },
      data: {
        status: "AUDITED",
        gpuName: auditResult.gpuName,
        gpuCount: auditResult.gpuCount,
        vramGb: auditResult.vramGb,
        cpuCores: auditResult.cpuCores,
        cpuRamGb: auditResult.cpuRamGb,
        lastSeenAt: new Date(),
      },
    });

    return NextResponse.json(updatedNode);
  } catch (error: any) {
    await prisma.node.update({
      where: { id: nodeId },
      data: {
        status: "ERROR",
        errorMessage: error.message || "Audit failed",
      },
    });

    return NextResponse.json(
      { error: error.message || "Audit failed" },
      { status: 500 }
    );
  }
}
