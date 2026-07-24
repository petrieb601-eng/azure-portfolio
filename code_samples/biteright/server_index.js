import "dotenv/config";
import express from "express";
import cors from "cors";

import "./db.js"; // ensures schema is created on boot
import authRoutes from "./routes/auth.js";
import foodRoutes from "./routes/foods.js";
import dataRoutes from "./routes/data.js";
import premiumRoutes from "./routes/premium.js";
import aiRoutes from "./routes/ai.js";
import billingRoutes, { stripeWebhook, billingMode } from "./routes/payments.js";

const app = express();
const PORT = process.env.PORT || 4000;
const CLIENT_URL = process.env.CLIENT_URL || "http://localhost:5173";

app.use(cors({ origin: CLIENT_URL, credentials: true }));

// Stripe webhook MUST receive the raw body (mounted before express.json()).
app.post("/api/billing/webhook", express.raw({ type: "application/json" }), stripeWebhook);

// JSON parser for everything else.
app.use(express.json({ limit: "5mb" }));

app.get("/api/health", (_req, res) => res.json({ ok: true, billingMode }));

app.use("/api/auth", authRoutes);
app.use("/api/foods", foodRoutes);
app.use("/api/data", dataRoutes);
app.use("/api/premium", premiumRoutes);
app.use("/api/ai", aiRoutes);
app.use("/api/billing", billingRoutes);

app.use((_req, res) => res.status(404).json({ error: "Not found." }));

app.listen(PORT, () => {
  console.log(`BiteRight API listening on http://localhost:${PORT}`);
  console.log(`Billing mode: ${billingMode}${billingMode === "mock" ? " (set STRIPE_SECRET_KEY to take real payments)" : ""}`);
});
