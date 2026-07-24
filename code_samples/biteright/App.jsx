import React, { useState, useEffect, useRef, useCallback } from "react";
import { Heart } from "lucide-react";
import { api } from "./api";
import AuthScreen from "./AuthScreen";
import {
  C, FONT, TEXT_SIZES, MEAL_SLOTS, TIPS,
  dateKey, defaultData, mergeData, computeStreak,
  HomePage, MealsPage, FitnessPage, WellnessPage, MorePage,
  UpgradePage, AnalyticsPage, PlansPage, BottomNav,
} from "./components";
import { ProgramsPage, MealPlanPage } from "./premium-features";

const LOCAL_KEY = "biteright_data";
function loadLocal() {
  try { return JSON.parse(localStorage.getItem(LOCAL_KEY) || "null"); } catch { return null; }
}

export default function App() {
  const [data, setData] = useState(null);
  const [account, setAccount] = useState(null);
  const [serverUp, setServerUp] = useState(false);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState("home");
  const [busy, setBusy] = useState(false);
  const saveTimer = useRef(null);

  // Detect whether the API is reachable (enables catalog search even when signed out).
  useEffect(() => { api.health().then(() => setServerUp(true)).catch(() => setServerUp(false)); }, []);

  // Initial load: prefer a signed-in account's synced data, else local cache.
  useEffect(() => {
    (async () => {
      const token = api.getToken();
      if (token) {
        try {
          const { user } = await api.me();
          setAccount(user);
          const { data: remote } = await api.getData();
          const base = remote && Object.keys(remote).length ? remote : loadLocal();
          const merged = mergeData(base);
          merged.profile.isPremium = user.isPremium;
          setData(merged);
          setLoading(false);
          return;
        } catch {
          api.clearToken();
        }
      }
      setData(mergeData(loadLocal()));
      setLoading(false);
    })();
  }, []);

  // Handle return from Stripe Checkout.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("upgraded")) {
      api.me().then(({ user }) => {
        setAccount(user);
        setData((d) => (d ? { ...d, profile: { ...d.profile, isPremium: user.isPremium } } : d));
      }).catch(() => {});
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []);

  // Apply text-size scaling to the root font-size.
  useEffect(() => {
    if (!data) return;
    const t = TEXT_SIZES.find((x) => x.id === data.profile.textSize) || TEXT_SIZES[0];
    document.documentElement.style.fontSize = t.root;
  }, [data?.profile.textSize]);

  // Persist: always cache locally; sync to the server when signed in (debounced).
  useEffect(() => {
    if (!data) return;
    try { localStorage.setItem(LOCAL_KEY, JSON.stringify(data)); } catch {}
    if (account) {
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => { api.putData(data).catch(() => {}); }, 600);
    }
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [data, account]);

  const go = useCallback((v) => { setView(v); window.scrollTo?.({ top: 0 }); }, []);

  const onReset = useCallback(() => { setData(defaultData()); setView("home"); }, []);

  // After signing in: merge — if the server has data, adopt it; otherwise push local up.
  const onAuthed = useCallback(async (user) => {
    setAccount(user);
    try {
      const { data: remote } = await api.getData();
      if (remote && Object.keys(remote).length) {
        const merged = mergeData(remote);
        merged.profile.isPremium = user.isPremium;
        setData(merged);
      } else {
        setData((d) => {
          const next = d ? { ...d, profile: { ...d.profile, isPremium: user.isPremium } } : mergeData(null);
          api.putData(next).catch(() => {});
          return next;
        });
      }
    } catch {}
    setView("home");
  }, []);

  const onSignOut = useCallback(() => {
    api.clearToken();
    setAccount(null);
    setData((d) => (d ? { ...d, profile: { ...d.profile, isPremium: false } } : d));
    setView("home");
  }, []);

  const onSubscribe = useCallback(async () => {
    setBusy(true);
    try {
      const r = await api.createCheckout();
      if (r.url) { window.location.href = r.url; return; }     // real Stripe Checkout
      if (r.mock) {                                            // mock billing mode
        setAccount((a) => (a ? { ...a, isPremium: true } : a));
        setData((d) => ({ ...d, profile: { ...d.profile, isPremium: true } }));
        setView("home");
      }
    } catch (e) {
      alert(e.message || "Could not start checkout.");
    } finally {
      setBusy(false);
    }
  }, []);

  const onManageBilling = useCallback(async () => {
    try { const r = await api.openPortal(); if (r.url) window.location.href = r.url; }
    catch (e) { alert(e.message || "Billing portal unavailable."); }
  }, []);

  const onExport = useCallback(() => { api.exportData().catch((e) => alert(e.message || "Export failed.")); }, []);

  if (loading || !data) {
    return (
      <div style={{ backgroundColor: C.bg, fontFamily: FONT }} className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div style={{ backgroundColor: C.brandSoft }} className="mx-auto mb-3 inline-flex rounded-3xl p-4"><Heart size={32} style={{ color: C.brand }} /></div>
          <p style={{ color: C.ink }} className="text-xl font-bold">BiteRight</p>
          <p style={{ color: C.inkSoft }} className="mt-1">Getting things ready…</p>
        </div>
      </div>
    );
  }

  // Derived dashboard values
  const today = dateKey();
  const dayMeals = data.meals[today] || {};
  let kcalToday = 0, itemsToday = 0;
  MEAL_SLOTS.forEach((s) => (dayMeals[s.id] || []).forEach((it) => { kcalToday += it.cal; itemsToday++; }));
  const waterToday = dayMeals.water || 0;
  const moodToday = data.wellness[today]?.mood;
  const workedOutToday = (data.completedExercises[today] || []).length > 0;
  const streak = computeStreak(data);
  const tip = TIPS[new Date().getDate() % TIPS.length];

  const foodSearch = serverUp ? api.searchFoods : null;

  return (
    <div style={{ backgroundColor: C.bg, fontFamily: FONT }} className="min-h-screen w-full">
      <div className="mx-auto min-h-screen w-full max-w-md px-4 pb-32 pt-6">
        {view === "home" && (
          <HomePage data={data} go={go} todaysMeals={itemsToday} kcalToday={kcalToday} waterToday={waterToday} moodToday={moodToday} workedOutToday={workedOutToday} streak={streak} tip={tip} />
        )}
        {view === "meals" && <MealsPage data={data} setData={setData} go={go} searchFoods={foodSearch} onAnalyzePhoto={(file) => api.analyzePhoto(file)} />}
        {view === "fitness" && <FitnessPage data={data} setData={setData} go={go} />}
        {view === "wellness" && <WellnessPage data={data} setData={setData} go={go} />}
        {view === "more" && (
          <MorePage
            data={data} setData={setData} go={go} onReset={onReset}
            account={account}
            onSignIn={() => go("auth")}
            onSignOut={onSignOut}
            onManageBilling={account ? onManageBilling : undefined}
            onExport={account ? onExport : undefined}
          />
        )}
        {view === "upgrade" && (
          <UpgradePage
            data={data} setData={setData} back={() => go("more")}
            account={account} onSubscribe={onSubscribe} onNeedAccount={() => go("auth")} busy={busy}
          />
        )}
        {view === "analytics" && <AnalyticsPage data={data} back={() => go("more")} />}
        {view === "plans" && <PlansPage data={data} setData={setData} back={() => go("more")} />}
        {view === "programs" && <ProgramsPage data={data} setData={setData} back={() => go("more")} />}
        {view === "mealplan" && <MealPlanPage data={data} setData={setData} back={() => go("more")} />}
        {view === "auth" && <AuthScreen onAuthed={onAuthed} onCancel={() => go("more")} />}
      </div>
      {view !== "auth" && <BottomNav view={view} go={go} />}
    </div>
  );
}
