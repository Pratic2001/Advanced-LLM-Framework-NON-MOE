/**
 * Standardized API error handling utilities.
 * Provides typed error classes and a consistent response helper.
 */

import { NextResponse } from "next/server";

export class ApiError extends Error {
  statusCode: number;
  details?: unknown;

  constructor(statusCode: number, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

export function unauthorized(): ApiError {
  return new ApiError(401, "Unauthorized");
}

export function notFound(resource = "Resource"): ApiError {
  return new ApiError(404, `${resource} not found`);
}

export function badRequest(message: string, details?: unknown): ApiError {
  return new ApiError(400, message, details);
}

export function validationError(
  fieldErrors: Record<string, string[]>
): ApiError {
  return new ApiError(400, "Validation failed", { fields: fieldErrors });
}

export function internalError(): ApiError {
  return new ApiError(500, "Internal server error");
}

/**
 * Convert any thrown error into a consistent NextResponse.
 * Handles ApiError, ZodError, and generic Error.
 */
export function handleApiError(err: unknown): NextResponse {
  if (err instanceof ApiError) {
    return NextResponse.json(
      {
        error: err.message,
        ...(err.details ? { details: err.details } : {}),
      },
      { status: err.statusCode }
    );
  }

  // Fallback for unexpected errors
  console.error("[ApiError] Unhandled:", err);
  return NextResponse.json(
    { error: "Internal server error" },
    { status: 500 }
  );
}
