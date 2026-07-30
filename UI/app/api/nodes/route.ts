import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";

export async function GET() {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const nodes = await prisma.node.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
  });

  return NextResponse.json(nodes);
}

export async function POST(req: Request) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { name, host, port, username, role, sshKeyId } = body;

  if (!name || !host || !username) {
    return NextResponse.json(
      { error: "Name, host, and username are required" },
      { status: 400 }
    );
  }

  const node = await prisma.node.create({
    data: {
      userId: session.user.id,
      name,
      host,
      port: port || 22,
      username,
      role: role || "WORKER",
      status: "AUDITING",
      sshKeyId: sshKeyId || null,
    },
  });

  return NextResponse.json(node, { status: 201 });
}
