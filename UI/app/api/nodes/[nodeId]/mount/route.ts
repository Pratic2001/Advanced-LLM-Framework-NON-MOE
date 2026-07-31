import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { mountNfsSchema, formatZodErrors } from "@/lib/validations";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ nodeId: string }> }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = mountNfsSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const { nodeId } = await params;
  const { nfsServer, exportPath, mountPoint } = parsed.data;

  const node = await prisma.node.findFirst({
    where: { id: nodeId, userId: session.user.id },
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  try {
    // In production: use ssh-manager to SSH into the node and run mount command
    // const ssh = await getSSHConnection(node);
    // await ssh.exec(`sudo mkdir -p ${mountPoint}`);
    // await ssh.exec(`sudo mount -t nfs ${nfsServer}:${exportPath} ${mountPoint}`);

    // Mark NFS as mounted
    const updatedNode = await prisma.node.update({
      where: { id: nodeId },
      data: {
        nfsMounted: true,
        nfsMountPath: mountPoint || "/mnt/training",
      },
    });

    return NextResponse.json(updatedNode);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "NFS mount failed" },
      { status: 500 }
    );
  }
}

export async function DELETE(
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
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  try {
    // In production: SSH into node and unmount
    // await ssh.exec(`sudo umount ${node.nfsMountPath}`);

    const updatedNode = await prisma.node.update({
      where: { id: nodeId },
      data: {
        nfsMounted: false,
        nfsMountPath: null,
      },
    });

    return NextResponse.json(updatedNode);
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "NFS unmount failed" },
      { status: 500 }
    );
  }
}
