import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const jobs = pgTable("jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  engine: text("engine").notNull().default("ace-step"),
  runpodJobId: text("runpod_job_id"),
  status: text("status").notNull().default("PENDING"),
  prompt: text("prompt").notNull(),
  lyrics: text("lyrics"),
  duration: integer("duration").notNull().default(30),
  style: text("style"),
  instrument: text("instrument"),
  tempo: integer("tempo"),
  inputParams: jsonb("input_params"),
  progress: integer("progress").default(0),
  outputUrl: text("output_url"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

export const insertJobSchema = createInsertSchema(jobs).omit({
  id: true,
  runpodJobId: true,
  status: true,
  progress: true,
  outputUrl: true,
  errorMessage: true,
  createdAt: true,
  completedAt: true,
});

export const createJobSchema = z.object({
  engine: z.string().default("ace-step"),
  prompt: z.string().min(1, "Prompt is required"),
  lyrics: z.string().optional(),
  duration: z.number().min(1).max(600).default(30),
  style: z.string().optional(),
  instrument: z.string().optional(),
  tags: z.string().optional(),
  negative_tags: z.string().optional(),
  title: z.string().optional(),
  bpm: z.number().min(30).max(300).optional(),
  key_scale: z.string().optional(),
  time_signature: z.string().optional(),
  vocal_language: z.string().optional(),
  inference_steps: z.number().min(1).max(200).optional(),
  guidance_scale: z.number().min(0).max(30).optional(),
  thinking: z.boolean().optional(),
  shift: z.number().min(1).max(5).optional(),
  infer_method: z.enum(["ode", "sde"]).optional(),
  use_adg: z.boolean().optional(),
  batch_size: z.number().min(1).max(8).optional(),
  seed: z.number().int().optional(),
  audio_format: z.enum(["mp3", "wav", "flac"]).optional(),
  model: z.string().optional(),
  temperature: z.number().min(0.1).max(2.0).optional(),
  cfg_scale: z.number().min(0).max(10).optional(),
  topk: z.number().int().min(1).max(500).optional(),
});

export type InsertJob = z.infer<typeof insertJobSchema>;
export type Job = typeof jobs.$inferSelect;
export type CreateJobInput = z.infer<typeof createJobSchema>;

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export const comparisons = pgTable("comparisons", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  status: text("status").notNull().default("PENDING"),
  engine: text("engine").notNull().default("heartmula"),
  prompt: text("prompt").notNull(),
  lyrics: text("lyrics"),
  tags: text("tags"),
  negative_tags: text("negative_tags"),
  title: text("title"),
  duration: integer("duration").notNull().default(30),
  inputParams: jsonb("input_params"),
  ourJobId: text("our_job_id"),
  sunoJobId: text("suno_job_id"),
  ourStatus: text("our_status").default("PENDING"),
  sunoStatus: text("suno_status").default("PENDING"),
  ourAudioUrl: text("our_audio_url"),
  sunoAudioUrl: text("suno_audio_url"),
  sunoModel: text("suno_model").default("V5"),
  ourPpJobId: text("our_pp_job_id"),
  ourPpStatus: text("our_pp_status"),
  ourPpAudioUrl: text("our_pp_audio_url"),
  geminiAnalysis: jsonb("gemini_analysis"),
  errorMessage: text("error_message"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

export const insertComparisonSchema = createInsertSchema(comparisons).omit({
  id: true,
  status: true,
  ourJobId: true,
  sunoJobId: true,
  ourStatus: true,
  sunoStatus: true,
  ourAudioUrl: true,
  sunoAudioUrl: true,
  ourPpJobId: true,
  ourPpStatus: true,
  ourPpAudioUrl: true,
  geminiAnalysis: true,
  errorMessage: true,
  createdAt: true,
  completedAt: true,
});

export const createComparisonSchema = z.object({
  engine: z.string().default("heartmula"),
  prompt: z.string().min(1, "Prompt is required"),
  lyrics: z.string().optional(),
  tags: z.string().optional(),
  negative_tags: z.string().optional(),
  title: z.string().optional(),
  duration: z.number().min(10).max(300).default(30),
  sunoModel: z.enum(["V5", "V4_5ALL", "V4_5PLUS", "V4_5", "V4"]).default("V5"),
  enablePP: z.boolean().default(false),
  inputParams: z.record(z.any()).optional(),
});

export type Comparison = typeof comparisons.$inferSelect;
export type InsertComparison = z.infer<typeof insertComparisonSchema>;
