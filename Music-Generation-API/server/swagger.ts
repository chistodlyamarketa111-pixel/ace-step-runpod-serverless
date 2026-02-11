import path from "path";
import fs from "fs";
import YAML from "yaml";
import swaggerUi from "swagger-ui-express";
import type { Express } from "express";

export function setupSwagger(app: Express) {
  const specPath = path.resolve(process.cwd(), "server", "openapi.yaml");

  let raw = "";
  try {
    raw = fs.readFileSync(specPath, "utf8");
    console.log(`[swagger] loaded ${specPath} (${raw.length} chars)`);
  } catch (e: any) {
    console.error(`[swagger] read error at ${specPath}: ${e.message}`);
    raw = `openapi: 3.0.3
info:
  title: Swagger spec not found
  version: 0.0.0
paths: {}
`;
  }

  let spec: any;
  try {
    spec = YAML.parse(raw);
  } catch (e: any) {
    console.error(`[swagger] YAML parse error: ${e.message}`);
    spec = {
      openapi: "3.0.3",
      info: { title: "Swagger YAML parse error", version: "0.0.0" },
      paths: {},
    };
  }

  app.get("/openapi.yaml", (_req, res) => res.type("text/yaml").send(raw));
  app.get("/openapi.json", (_req, res) => res.json(spec));
  app.use("/docs", swaggerUi.serve, swaggerUi.setup(spec));
}
