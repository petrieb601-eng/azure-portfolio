import { Router } from "express";
import Stripe from "stripe";
import { requireAuth } from "../middleware.js";
import { setPremium, setStripeCustomer, userByStripeCustomer, userById, publicUser } from "../db.js";

const KEY = process.env.STRIPE_SECRET_KEY || "";
const PRICE_ID = process.env.STRIPE_PRICE_ID || "";
const WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET || "";
const CLIENT_URL = process.env.CLIENT_URL || "http://localhost:5173";

// Stripe is only initialized when a secret key is provided.
export const stripe = KEY ? new Stripe(KEY) : null;
export const billingMode = stripe ? "stripe" : "mock";

const router = Router();

// POST /api/billing/checkout — start a Premium subscription.
// In MOCK mode (no Stripe key) we simply grant Premium so you can build end-to-end.
router.post("/checkout", requireAuth, async (req, res) => {
  if (!stripe) {
    setPremium.run(1, req.user.id);
    return res.json({ mock: true, user: publicUser(userById.get(req.user.id)) });
  }
  if (!PRICE_ID) return res.status(500).json({ error: "STRIPE_PRICE_ID is not configured on the server." });

  try {
    let customerId = req.user.stripe_customer_id;
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: req.user.email,
        name: req.user.name || undefined,
        metadata: { userId: String(req.user.id) },
      });
      customerId = customer.id;
      setStripeCustomer.run(customerId, req.user.id);
    }
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      customer: customerId,
      line_items: [{ price: PRICE_ID, quantity: 1 }],
      success_url: `${CLIENT_URL}/?upgraded=1`,
      cancel_url: `${CLIENT_URL}/?canceled=1`,
      metadata: { userId: String(req.user.id) },
    });
    res.json({ url: session.url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// POST /api/billing/portal — manage/cancel an existing subscription.
router.post("/portal", requireAuth, async (req, res) => {
  if (!stripe) return res.json({ mock: true });
  if (!req.user.stripe_customer_id) return res.status(400).json({ error: "No billing account yet." });
  try {
    const session = await stripe.billingPortal.sessions.create({
      customer: req.user.stripe_customer_id,
      return_url: CLIENT_URL,
    });
    res.json({ url: session.url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Raw-body webhook handler (mounted separately in index.js).
// Keeps the user's premium flag in sync with Stripe subscription events.
export function stripeWebhook(req, res) {
  if (!stripe || !WEBHOOK_SECRET) return res.status(400).send("Webhooks not configured.");
  let event;
  try {
    event = stripe.webhooks.constructEvent(req.body, req.headers["stripe-signature"], WEBHOOK_SECRET);
  } catch (e) {
    return res.status(400).send(`Webhook signature verification failed: ${e.message}`);
  }

  const findUser = (customerId, metaUserId) =>
    (customerId && userByStripeCustomer.get(customerId)) ||
    (metaUserId && userById.get(Number(metaUserId)));

  switch (event.type) {
    case "checkout.session.completed": {
      const s = event.data.object;
      const user = findUser(s.customer, s.metadata?.userId);
      if (user) {
        if (s.customer && !user.stripe_customer_id) setStripeCustomer.run(s.customer, user.id);
        setPremium.run(1, user.id);
      }
      break;
    }
    case "customer.subscription.updated":
    case "customer.subscription.created": {
      const sub = event.data.object;
      const user = findUser(sub.customer);
      if (user) setPremium.run(["active", "trialing", "past_due"].includes(sub.status) ? 1 : 0, user.id);
      break;
    }
    case "customer.subscription.deleted": {
      const sub = event.data.object;
      const user = findUser(sub.customer);
      if (user) setPremium.run(0, user.id);
      break;
    }
    default:
      break;
  }
  res.json({ received: true });
}

export default router;
