import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { updateConfigSchema, formatZodErrors } from "@/lib/validations";

export async function GET(
  req: Request,
  { params }: { params: { configId: string } }
) {
  const session = await auth();
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
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = updateConfigSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  const preset = await prisma.configPreset.findFirst({
    where: { id: params.configId, userId: session.user.id },
  });

  if (!preset) {
    return NextResponse.json({ error: "Config not found" }, { status: 404 });
  }

  const updated = await prisma.configPreset.update({
    where: { id: params.configId },
    data: {
      name: parsed.data.name ?? preset.name,
      description: parsed.data.description ?? preset.description,
      config: (parsed.data.config ?? preset.config) as any,
    },
  });

  return NextResponse.json(updated);
}

export async function DELETE(
  req: Request,
  { params }: { params: { configId: string } }
) {
  const session = await auth();
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
