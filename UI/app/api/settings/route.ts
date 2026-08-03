import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import prisma from "@/lib/db";
import { updateSettingsSchema, formatZodErrors } from "@/lib/validations";

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { pythonBin: true },
  });

  return NextResponse.json({ pythonBin: user?.pythonBin ?? "" });
}

export async function PATCH(req: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const parsed = updateSettingsSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      { error: "Validation failed", details: formatZodErrors(parsed.error) },
      { status: 400 }
    );
  }

  // A trimmed empty string clears the setting back to the system default.
  const pythonBin = parsed.data.pythonBin?.trim() || null;

  const user = await prisma.user.update({
    where: { id: session.user.id },
    data: { pythonBin },
    select: { pythonBin: true },
  });

  return NextResponse.json({ pythonBin: user.pythonBin ?? "" });
}
