import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";

export async function POST(
  req: Request,
  { params }: { params: { nodeId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { nfsServer, exportPath, mountPoint } = body;

  const node = await prisma.node.findFirst({
    where: { id: params.nodeId, userId: session.user.id },
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
      where: { id: params.nodeId },
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
  { params }: { params: { nodeId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const node = await prisma.node.findFirst({
    where: { id: params.nodeId, userId: session.user.id },
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  try {
    // In production: SSH into node and unmount
    // await ssh.exec(`sudo umount ${node.nfsMountPath}`);

    const updatedNode = await prisma.node.update({
      where: { id: params.nodeId },
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
