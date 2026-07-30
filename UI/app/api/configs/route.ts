import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";

export async function GET() {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const configs = await prisma.configPreset.findMany({
    where: { userId: session.user.id },
    orderBy: { updatedAt: "desc" },
  });

  return NextResponse.json(configs);
}

export async function POST(req: Request) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { name, description, backend, config } = body;

  if (!name || !backend || !config) {
    return NextResponse.json(
      { error: "Name, backend, and config are required" },
      { status: 400 }
    );
  }

  const preset = await prisma.configPreset.create({
    data: {
      userId: session.user.id,
      name,
      description,
      backend: backend.toUpperCase() as any,
      config,
    },
  });

  return NextResponse.json(preset, { status: 201 });
}
