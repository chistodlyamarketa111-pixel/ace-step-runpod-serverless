import { type Job, type InsertJob, type User, type InsertUser, type Comparison, type InsertComparison, jobs, users, comparisons } from "@shared/schema";
import { db } from "./db";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  createJob(job: Partial<InsertJob> & { prompt: string }): Promise<Job>;
  getJob(id: string): Promise<Job | undefined>;
  getAllJobs(): Promise<Job[]>;
  updateJob(id: string, updates: Partial<Job>): Promise<Job | undefined>;
  createComparison(data: Partial<InsertComparison> & { prompt: string }): Promise<Comparison>;
  getComparison(id: string): Promise<Comparison | undefined>;
  getAllComparisons(): Promise<Comparison[]>;
  updateComparison(id: string, updates: Partial<Comparison>): Promise<Comparison | undefined>;
}

export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const [user] = await db.insert(users).values(insertUser).returning();
    return user;
  }

  async createJob(jobData: Partial<InsertJob> & { prompt: string }): Promise<Job> {
    const [job] = await db.insert(jobs).values(jobData).returning();
    return job;
  }

  async getJob(id: string): Promise<Job | undefined> {
    const [job] = await db.select().from(jobs).where(eq(jobs.id, id));
    return job;
  }

  async getAllJobs(): Promise<Job[]> {
    return db.select().from(jobs).orderBy(desc(jobs.createdAt));
  }

  async updateJob(id: string, updates: Partial<Job>): Promise<Job | undefined> {
    const [job] = await db.update(jobs).set(updates).where(eq(jobs.id, id)).returning();
    return job;
  }

  async createComparison(data: Partial<InsertComparison> & { prompt: string }): Promise<Comparison> {
    const [comparison] = await db.insert(comparisons).values(data).returning();
    return comparison;
  }

  async getComparison(id: string): Promise<Comparison | undefined> {
    const [comparison] = await db.select().from(comparisons).where(eq(comparisons.id, id));
    return comparison;
  }

  async getAllComparisons(): Promise<Comparison[]> {
    return db.select().from(comparisons).orderBy(desc(comparisons.createdAt));
  }

  async updateComparison(id: string, updates: Partial<Comparison>): Promise<Comparison | undefined> {
    const [comparison] = await db.update(comparisons).set(updates).where(eq(comparisons.id, id)).returning();
    return comparison;
  }
}

export const storage = new DatabaseStorage();
