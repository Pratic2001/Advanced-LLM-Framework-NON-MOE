import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { createNodeSchema, formatZodErrors } from "@/lib/validations";

export async function GET() {
  const session = await auth();
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
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = createNodeSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const { name, host, port, username, role, sshKeyId } = parsed.data;

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
