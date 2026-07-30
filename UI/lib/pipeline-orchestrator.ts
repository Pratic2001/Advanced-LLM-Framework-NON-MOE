/**
 * Pipeline Orchestrator
 * Manages multi-stage pipeline execution (tokenizer → pack → pretrain → SFT → GRPO/DPO)
 */

import prisma from "./db";
import { buildCommand } from "./command-builder";
import { JobManager } from "./job-manager";
import { sendPipelineFailureAlert } from "./mail";
import type { BackendType, ScriptName } from "./schema";

export interface StageDefinition {
  type: string;
  label: string;
  config: Record<string, any>;
  dependsOn?: string[]; // Stage types this depends on
  extraArgs?: string;   // Raw CLI args appended to the command
}

export interface PipelineConfig {
  name: string;
  userId: string;
  backend: BackendType;
  stages: StageDefinition[];
  nodeIds?: string[];
}

export class PipelineOrchestrator {
  private jobManager: JobManager;
  private activePipelines: Map<string, boolean> = new Map();

  constructor() {
    this.jobManager = JobManager.getInstance();
  }

  /**
   * Launch a full pipeline
   */
  async launch(config: PipelineConfig): Promise<any> {
    const { name, userId, backend, stages, nodeIds } = config;

    // Create pipeline record
    const pipeline = await prisma.pipeline.create({
      data: {
        userId,
        name,
        backend: backend.toUpperCase() as any,
        status: "RUNNING",
      },
    });

    this.activePipelines.set(pipeline.id, true);

    // Create stages and launch first stage
    let previousJobId: string | null = null;

    for (let i = 0; i < stages.length; i++) {
      const stage = stages[i];
      const scriptName = this.getScriptName(stage.type);

      // Build the command
      const command = buildCommand({
        script: scriptName,
        config: stage.config,
        backend,
        nodeCount: nodeIds?.length || 1,
        gpuCount: 1,
        extraArgs: stage.extraArgs,
      });

      // Create job record
      const job: any = await prisma.job.create({
        data: {
          userId,
          type: stage.type.toUpperCase() as any,
          backend: backend.toUpperCase() as any,
          status: i === 0 ? "QUEUED" : "PENDING",
          config: (stage.extraArgs
            ? { ...stage.config, __extraArgs: stage.extraArgs }
            : stage.config) as any,
          ...(previousJobId ? { parentJobId: previousJobId } : {}),
        },
      });

      // Create pipeline stage record
      await prisma.pipelineStage.create({
        data: {
          pipelineId: pipeline.id,
          stageOrder: i,
          jobType: stage.type.toUpperCase() as any,
          config: stage.config as any,
          status: i === 0 ? "WAITING" : "PENDING",
          job: { connect: { id: job.id } },
        },
      });

      previousJobId = job.id;
    }

    // Launch the first stage
    await this.launchNextStage(pipeline.id);
    this.emitStatus(pipeline.id, "RUNNING");

    return pipeline.id;
  }

  /**
   * Launch the next stage of a pipeline
   */
  private async launchNextStage(pipelineId: string): Promise<void> {
    // Find the first PENDING or WAITING stage
    const nextStage = await prisma.pipelineStage.findFirst({
      where: {
        pipelineId,
        OR: [{ status: "PENDING" }, { status: "WAITING" }],
      },
      orderBy: { stageOrder: "asc" },
      include: { job: true },
    });

    if (!nextStage || !nextStage.job) {
      // All stages complete
      await prisma.pipeline.update({
        where: { id: pipelineId },
        data: { status: "COMPLETED" },
      });
      this.activePipelines.delete(pipelineId);
      this.emitStatus(pipelineId, "COMPLETED");
      return;
    }

    // Mark stage as running
    await prisma.pipelineStage.update({
      where: { id: nextStage.id },
      data: { status: "RUNNING" },
    });

    // Build and launch the job
    const storedConfig = nextStage.config as Record<string, any> || {};
    const storedExtraArgs = typeof storedConfig.__extraArgs === "string"
      ? storedConfig.__extraArgs
      : undefined;
    // Remove __extraArgs so buildFlags doesn't render it as --__extraArgs
    const { __extraArgs, ...cleanConfig } = storedConfig;

    const command = buildCommand({
      script: this.getScriptName(nextStage.jobType),
      config: cleanConfig,
      backend: (nextStage.job.backend as string).toLowerCase() as BackendType,
      extraArgs: storedExtraArgs,
    });

    const success = await this.jobManager.launch(
      nextStage.job.id,
      command,
      []
    );

    if (!success) {
      await prisma.pipelineStage.update({
        where: { id: nextStage.id },
        data: { status: "FAILED" },
      });
      await prisma.pipeline.update({
        where: { id: pipelineId },
        data: { status: "FAILED" },
      });
      this.activePipelines.delete(pipelineId);
      this.emitStatus(pipelineId, "FAILED");

      // Send pipeline failure alert
      this.sendPipelineAlert(pipelineId, nextStage);
    }

    // Listen for job completion to launch next stage
    // JobManager emits "job:status" with jobId in event data — filter by our job ID
    const onStatus = async (event: any) => {
      if (event.jobId !== nextStage.job!.id) return;
      const status = event.data.status;
      if (status === "COMPLETED") {
        await prisma.pipelineStage.update({
          where: { id: nextStage.id },
          data: { status: "COMPLETED" },
        });
        await this.launchNextStage(pipelineId);
      } else if (status === "FAILED" || status === "CANCELLED") {
        await prisma.pipelineStage.update({
          where: { id: nextStage.id },
          data: { status: "FAILED" },
        });
        await prisma.pipeline.update({
          where: { id: pipelineId },
          data: { status: "FAILED" },
        });
        this.activePipelines.delete(pipelineId);
        this.emitStatus(pipelineId, "FAILED");

        // Send pipeline failure alert
        this.sendPipelineAlert(pipelineId, nextStage);
      }
    };

    this.jobManager.on("job:status", onStatus);
  }

  /**
   * Stop a pipeline
   */
  async stop(pipelineId: string): Promise<void> {
    this.activePipelines.delete(pipelineId);

    const runningStage = await prisma.pipelineStage.findFirst({
      where: {
        pipelineId,
        status: "RUNNING",
      },
      include: { job: true },
    });

    if (runningStage?.job) {
      await this.jobManager.stop(runningStage.job.id);
    }

    await prisma.pipeline.update({
      where: { id: pipelineId },
      data: { status: "CANCELLED" },
    });

    await prisma.pipelineStage.updateMany({
      where: {
        pipelineId,
        status: { in: ["PENDING", "WAITING", "RUNNING"] },
      },
      data: { status: "CANCELLED" },
    });
  }

  /**
   * Send email alert when a pipeline stage fails
   */
  private async sendPipelineAlert(
    pipelineId: string,
    failedStage: { id: string; jobType: string; jobId?: string | null }
  ): Promise<void> {
    try {
      const pipeline = await prisma.pipeline.findUnique({
        where: { id: pipelineId },
        include: { user: { select: { email: true } } },
      });
      if (!pipeline) return;

      let logTail = "";
      let errorMessage: string | null = null;
      if (failedStage.jobId) {
        const job = await prisma.job.findUnique({
          where: { id: failedStage.jobId },
          select: { logTail: true, errorMessage: true },
        });
        if (job) {
          logTail = job.logTail || "";
          errorMessage = job.errorMessage;
        }
      }

      sendPipelineFailureAlert({
        pipelineId: pipeline.id,
        pipelineName: pipeline.name,
        failedStage: failedStage.jobType,
        failedJobId: failedStage.jobId || "",
        errorMessage,
        logTail,
        createdAt: pipeline.createdAt,
        userEmail: pipeline.user.email,
      });
    } catch (err) {
      console.error("[PipelineOrchestrator] Failed to send pipeline alert:", err);
    }
  }

  private getScriptName(type: string): ScriptName {
    const map: Record<string, ScriptName> = {
      TOKENIZER: "tokenizer_train.py",
      PACKING: "hf_to_packed.py",
      PRETRAIN: "train_pretrain.py",
      SFT: "train_sft.py",
      GRPO: "train_grpo.py",
      DPO: "train_dpo.py",
    };
    return map[type.toUpperCase()] || "train_pretrain.py";
  }

  private emitStatus(pipelineId: string, status: string): void {
    this.jobManager.emit("pipeline:status", {
      pipelineId,
      type: "status" as const,
      data: { status },
      timestamp: new Date(),
    });
  }
}

// Singleton
let orchestratorInstance: PipelineOrchestrator | null = null;

export function getPipelineOrchestrator(): PipelineOrchestrator {
  if (!orchestratorInstance) {
    orchestratorInstance = new PipelineOrchestrator();
  }
  return orchestratorInstance;
}
