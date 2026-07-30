import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";

export async function GET(
  req: Request,
  { params }: { params: { nodeId: string } }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const node = await prisma.node.findFirst({
    where: { id: params.nodeId, userId: session.user.id },
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  return NextResponse.json(node);
}

export async function DELETE(
  req: Request,
  { params }: { params: { nodeId: string } }
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const node = await prisma.node.findFirst({
    where: { id: params.nodeId, userId: session.user.id },
  });

  if (!node) {
    return NextResponse.json({ error: "Node not found" }, { status: 404 });
  }

  await prisma.node.delete({ where: { id: params.nodeId } });

  return NextResponse.json({ success: true });
}
