import { NextResponse } from "next/server";
import { execSync } from "child_process";

export async function GET() {
  try {
    const gpuInfo = execSync(
      'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU found"'
    )
      .toString()
      .trim();

    const cpuInfo = execSync("nproc 2>/dev/null || echo 'unknown'").toString().trim();
    const ramInfo = execSync(
      'free -g 2>/dev/null | awk \'/Mem:/ {print $2 "G"}\' || echo "unknown"'
    )
      .toString()
      .trim();
    const pythonVersion = execSync(
      'python3 --version 2>/dev/null || echo "Python not found"'
    )
      .toString()
      .trim();

    return NextResponse.json({
      gpu: gpuInfo || "No NVIDIA GPU found",
      cpu: `${cpuInfo} cores`,
      ram: ramInfo,
      python: pythonVersion,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return NextResponse.json({
      gpu: "Unable to detect",
      cpu: "Unable to detect",
      ram: "Unable to detect",
      python: "Unable to detect",
      timestamp: new Date().toISOString(),
    });
  }
}
