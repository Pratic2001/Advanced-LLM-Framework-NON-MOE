import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import prisma from "@/lib/db";

export async function GET(
  req: Request,
  { params }: { params: { configId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const preset = await prisma.configPreset.findFirst({
    where: { id: params.configId, userId: session.user.id },
  });

  if (!preset) {
    return NextResponse.json({ error: "Config not found" }, { status: 404 });
  }

  return NextResponse.json(preset);
}

export async function PUT(
  req: Request,
  { params }: { params: { configId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const preset = await prisma.configPreset.findFirst({
    where: { id: params.configId, userId: session.user.id },
  });

  if (!preset) {
    return NextResponse.json({ error: "Config not found" }, { status: 404 });
  }

  const updated = await prisma.configPreset.update({
    where: { id: params.configId },
    data: {
      name: body.name ?? preset.name,
      description: body.description ?? preset.description,
      config: body.config ?? preset.config,
    },
  });

  return NextResponse.json(updated);
}

export async function DELETE(
  req: Request,
  { params }: { params: { configId: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const preset = await prisma.configPreset.findFirst({
    where: { id: params.configId, userId: session.user.id },
  });

  if (!preset) {
    return NextResponse.json({ error: "Config not found" }, { status: 404 });
  }

  await prisma.configPreset.delete({ where: { id: params.configId } });

  return NextResponse.json({ success: true });
}
