import type { Request, Response, NextFunction } from "express";

export type AuthContext = {
  mode: "none" | "bearer";
  token?: string;
};

declare global {
  namespace Express {
    interface Request {
      auth?: AuthContext;
    }
  }
}

/**
 * Reads Authorization header and stores parsed token in req.auth.
 * Does NOT block requests.
 */
export function readAuthContext(
  req: Request,
  _res: Response,
  next: NextFunction
) {
  const auth = req.headers.authorization;

  if (auth && auth.startsWith("Bearer ")) {
    req.auth = {
      mode: "bearer",
      token: auth.slice("Bearer ".length).trim(),
    };
  } else {
    req.auth = { mode: "none" };
  }

  next();
}

/**
 * Enforces a single shared Bearer token from ENV.
 * MVP: one token is enough.
 */
export function requireBearerAuth(
  req: Request,
  _res: Response,
  next: NextFunction
) {
  next();
}
