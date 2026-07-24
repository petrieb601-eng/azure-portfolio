import sys, os, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np, random, glob, json, csv
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASELINE_DIR   = os.path.expanduser("~/WRIVA_Challenge/baseline")
WEIGHTS        = os.path.expanduser("~/WRIVA_Challenge/example_models_flex_geo_dinov3_posloss/best.pt")
sys.path.insert(0, BASELINE_DIR)
from models.flex_geo_match_dinov3_posloss import FlexGeoApprox

DATA      = os.path.expanduser("~/WRIVA_Challenge/data/WRIVA-CVGL-DEV")
OUT       = os.path.expanduser("~/WRIVA_Challenge/attacks/wriva_attack")
CACHE_DIR = os.path.expanduser("~/WRIVA_Challenge/attacks/wriva_chip_cache")
CKPT      = os.path.join(OUT, "checkpoint.json")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

S1_ITERS       = 200
S1_TOP_K       = 1
S2_ITERS       = 300
S2_TOP_K       = 5
EPSILON        = 16/255
ALPHA          = EPSILON/4
EOT_N          = 5
RESTARTS       = 3
CHIPS_PER_TIF  = 20
MOMENTUM_DECAY = 1.0
RETARGET_EVERY = 25
N_PER_SITE     = 200
IMG_SIZE       = 224   # DINOv3 expects 224, not 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
NORM_MIN = (torch.zeros(1,3,1,1).to(device) - MEAN) / STD
NORM_MAX = (torch.ones(1,3,1,1).to(device)  - MEAN) / STD
EPS_NORM = EPSILON / STD
ALP_NORM = ALPHA   / STD

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

print("Loading WRIVA model...")
model = FlexGeoApprox(
    embed_dim=1024,
    enable_pos=True,
    pos_mode="grid",
    pos_grid=2,
    pos_loss_type="heatmap",
    pos_reg_beta=0.2,
    pos_head_variant="sat_token_heatmap",
    pos_head_hidden_dim=1024,
    pos_head_depth=2,
    separate_pos_neck=True,
    pretrained=False,
    backbone_model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
    enable_ial=False,
    ial_num_classes=4,
    sff_scale=2.0,
    share_backbone=True,
).to(device)

ckpt_data = torch.load(WEIGHTS, map_location=device)
state_dict = ckpt_data.get("model", ckpt_data)
# Strip DDP module. prefix if present
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  Missing keys:    {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
model.eval()
print("Model ready.\n")

# ── DUMMY TENSORS for single-branch embedding extraction ──────────────────────
# The forward requires both ground and sat together.
# We use fixed dummy tensors for the branch we don't care about.
DUMMY_SAT    = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
DUMMY_GROUND = torch.zeros(1, 1, 3, IMG_SIZE, IMG_SIZE).to(device)
DUMMY_MASK   = torch.ones(1, 1).to(device)

def load_image(path):
    return preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def load_sat_chip(tif_path, chip=160):
    img = Image.open(tif_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img, mask=img.split()[3]); img = bg
    elif img.mode != "RGB": img = img.convert("RGB")
    w,h = img.size
    c = random.randint(0, max(0, w-chip))
    r = random.randint(0, max(0, h-chip))
    return preprocess(img.crop((c,r,c+chip,r+chip))).unsqueeze(0).to(device)

def get_ground_emb(img_tensor):
    """img_tensor: [1, 3, H, W] — wraps to [1,1,3,H,W] for forward."""
    ground_imgs = img_tensor.unsqueeze(1)    # [1,1,3,H,W]
    with torch.no_grad():
        out = model(ground_imgs, DUMMY_MASK, DUMMY_SAT)
    return F.normalize(out["g_set"].squeeze(0), dim=-1)

def get_ground_emb_grad(adv):
    """Same but keeps grad. adv: [1,3,H,W]."""
    ground_imgs = adv.unsqueeze(1)
    out = model(ground_imgs, DUMMY_MASK, DUMMY_SAT)
    return F.normalize(out["g_set"].squeeze(0), dim=-1)

def get_sat_emb(sat_tensor):
    """sat_tensor: [1, 3, H, W]."""
    with torch.no_grad():
        out = model(DUMMY_GROUND, DUMMY_MASK, sat_tensor)
    return F.normalize(out["s"].squeeze(0), dim=-1)

def get_ground_emb_np(img_tensor):
    return get_ground_emb(img_tensor).cpu().numpy()

def eot_tfm(img):
    return T.Compose([
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.GaussianBlur(3, sigma=(0.1,1.0)),
    ])(img.squeeze(0)).unsqueeze(0)

def psnr(orig, adv):
    o = (orig*STD+MEAN).clamp(0,1)
    a = (adv *STD+MEAN).clamp(0,1)
    mse = torch.mean((o-a)**2).item()
    return round(10*np.log10(1.0/mse),2) if mse>0 else 999.0

def ssim_val(orig, adv):
    o = (orig*STD+MEAN).clamp(0,1).squeeze(0).cpu().numpy()
    a = (adv *STD+MEAN).clamp(0,1).squeeze(0).cpu().numpy()
    mu_o,mu_a = o.mean(),a.mean()
    so,sa = o.std(),a.std()
    soa = ((o-mu_o)*(a-mu_a)).mean()
    c1,c2 = 0.01**2,0.03**2
    return float(((2*mu_o*mu_a+c1)*(2*soa+c2))/
                 ((mu_o**2+mu_a**2+c1)*(so**2+sa**2+c2)))

# ── CHIP CACHE ─────────────────────────────────────────────────────────────────
def build_chip_cache(tif_paths, cache_name):
    emb_path = os.path.join(CACHE_DIR, f"{cache_name}_embs.npy")
    tif_path = os.path.join(CACHE_DIR, f"{cache_name}_tifs.json")
    if os.path.exists(emb_path):
        embs = np.load(emb_path)
        with open(tif_path) as f: tifs = json.load(f)
        print(f"  Loaded: {cache_name} ({len(embs)} chips)")
        return embs, tifs
    print(f"  Building: {cache_name} ({len(tif_paths)} TIFs x {CHIPS_PER_TIF} chips)...")
    all_embs, all_tifs = [], []
    for tif in tif_paths:
        for _ in range(CHIPS_PER_TIF):
            try:
                chip = load_sat_chip(tif)
                all_embs.append(get_sat_emb(chip).cpu().numpy())
                all_tifs.append(tif)
            except Exception as e:
                print(f"    Warning: {e}")
    embs = np.array(all_embs)
    np.save(emb_path, embs)
    with open(tif_path,"w") as f: json.dump(all_tifs, f)
    print(f"  Cached {len(embs)} chips")
    return embs, all_tifs

def get_target_emb(query_emb_np, cache_embs, k=1):
    sims    = cache_embs @ query_emb_np
    top_k   = np.argsort(sims)[-k:]
    avg_emb = cache_embs[top_k].mean(axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    return F.normalize(torch.tensor(avg_emb).float().to(device), dim=0)

# ── MOMENTUM PGD ──────────────────────────────────────────────────────────────
def run_pgd(start_adv, source, c_emb_t, init_w_emb_t,
            cache_embs, iters, top_k, momentum=None):
    adv = start_adv.clone().detach()
    if momentum is None:
        momentum = torch.zeros_like(adv.data)
    adv.requires_grad = True
    w_emb_t  = init_w_emb_t
    best_adv = adv.detach().clone()
    best_mg  = -999.0
    best_fw  = best_fc = 0.0

    for step in range(iters):
        if step > 0 and step % RETARGET_EVERY == 0:
            curr_np  = get_ground_emb_np(adv.detach())
            w_emb_t  = get_target_emb(curr_np, cache_embs, k=top_k).detach()

        total_grad = torch.zeros_like(adv.data)
        for _ in range(EOT_N):
            adv_t = eot_tfm(adv)
            g_emb = get_ground_emb_grad(adv_t)
            loss  = -F.cosine_similarity(g_emb.unsqueeze(0),
                                          w_emb_t.unsqueeze(0), dim=-1).squeeze() \
                  + 0.1*F.cosine_similarity(g_emb.unsqueeze(0),
                                             c_emb_t.unsqueeze(0), dim=-1).squeeze()
            loss.backward()
            if adv.grad is not None:
                total_grad += adv.grad.clone()
                adv.grad.zero_()

        with torch.no_grad():
            grad_norm = total_grad / (total_grad.abs().mean() + 1e-8)
            momentum  = MOMENTUM_DECAY * momentum + grad_norm
            adv.data -= ALP_NORM * momentum.sign()
            adv.data  = torch.max(torch.min(adv.data, source+EPS_NORM), source-EPS_NORM)
            adv.data  = torch.max(torch.min(adv.data, NORM_MAX), NORM_MIN)

        with torch.no_grad():
            adv_emb = get_ground_emb(adv.detach())
            fw = float(F.cosine_similarity(adv_emb.unsqueeze(0),
                                            w_emb_t.unsqueeze(0), dim=-1))
            fc = float(F.cosine_similarity(adv_emb.unsqueeze(0),
                                            c_emb_t.unsqueeze(0), dim=-1))
            mg = fw - fc
        if mg > best_mg:
            best_mg  = mg
            best_adv = adv.detach().clone()
            best_fw, best_fc = fw, fc

    return best_adv, best_fw, best_fc, best_mg, momentum.detach()

def hierarchical_attack(source, c_emb_t, source_emb_np,
                        cross_cache, same_cache, iters, top_k,
                        warm_momentum=None):
    init_cross = get_target_emb(source_emb_np, cross_cache, k=top_k)
    iw_cross   = float(F.cosine_similarity(c_emb_t.unsqueeze(0),
                                            init_cross.unsqueeze(0), dim=-1))
    starts = []
    if warm_momentum is not None:
        starts.append((source.clone().detach(), warm_momentum))
    for _ in range(RESTARTS if warm_momentum is None else RESTARTS-1):
        s = source.clone().detach()
        s = s + torch.empty_like(s).uniform_(-1,1)*EPS_NORM
        s = torch.max(torch.min(s, NORM_MAX), NORM_MIN).detach()
        starts.append((s, None))

    best_adv = source
    best_mg  = iw_cross - 1.0
    best_mom = None
    best_fw_p1 = best_fc_p1 = 0.0

    for start_adv, start_mom in starts:
        adv, fw, fc, mg, mom = run_pgd(
            start_adv, source, c_emb_t, init_cross,
            cross_cache, iters, top_k, momentum=start_mom)
        if mg > best_mg:
            best_mg, best_adv, best_mom = mg, adv, mom
            best_fw_p1, best_fc_p1 = fw, fc

    if best_mg > 0:
        return best_adv, best_fw_p1, best_fc_p1, best_mg, "cross_site", best_mom, iw_cross

    curr_emb  = get_ground_emb_np(best_adv.detach())
    init_same = get_target_emb(curr_emb, same_cache, k=top_k)
    iw_same   = float(F.cosine_similarity(c_emb_t.unsqueeze(0),
                                           init_same.unsqueeze(0), dim=-1))
    adv, fw, fc, mg, mom = run_pgd(
        best_adv.detach(), source, c_emb_t, init_same,
        same_cache, iters, top_k, momentum=best_mom)
    phase = "same_site" if mg > 0 else "not_flipped"
    iw    = iw_same if mg > 0 else iw_cross
    return adv, fw, fc, mg, phase, mom, iw

# ── DATASETS ──────────────────────────────────────────────────────────────────
DEV001_G = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-001/ground/*.jpg"))
DEV002_G = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-002/ground/*.jpg"))
DEV003_G = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-003/ground/*.jpg"))
DEV001_S = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-001/maxar/*.tif"))
DEV002_S = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-002/maxar/*.tif"))
DEV003_S = sorted(glob.glob(f"{DATA}/WRIVA-CVGL-DEV-003/maxar/*.tif"))

print("Loading chip caches...")
cache001_embs,_ = build_chip_cache(DEV001_S, "wriva_dev001_sat")
cache002_embs,_ = build_chip_cache(DEV002_S, "wriva_dev002_sat")
cache003_embs,_ = build_chip_cache(DEV003_S, "wriva_dev003_sat")

cross_cache = {"001": cache003_embs, "002": cache003_embs, "003": cache001_embs}
same_cache  = {"001": cache001_embs, "002": cache002_embs, "003": cache003_embs}

random.seed(42)
sample001 = random.sample(DEV001_G, min(N_PER_SITE, len(DEV001_G)))
sample002 = random.sample(DEV002_G, min(N_PER_SITE, len(DEV002_G)))
sample003 = random.sample(DEV003_G, min(N_PER_SITE, len(DEV003_G)))

jobs = (
    [(src,"001") for src in sample001] +
    [(src,"002") for src in sample002] +
    [(src,"003") for src in sample003]
)
total = len(jobs)
print(f"\nTotal: {total} | S1:{S1_ITERS} iters 1-chip | S2:{S2_ITERS} iters top-{S2_TOP_K}\n")

# ── CHECKPOINT ────────────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CKPT):
        with open(CKPT) as f: return json.load(f)
    return {"s1_done":{}, "s1_results":{}, "s1_momentum":{},
            "s2_done":{}, "s2_results":{}}

def save_checkpoint(ckpt):
    with open(CKPT,"w") as f: json.dump(ckpt, f)

ckpt = load_checkpoint()

csv_path   = os.path.join(OUT, "results.csv")
fieldnames = ["src","site","stage","phase","init_wrong",
              "final_correct","final_wrong","final_margin",
              "margin_shift","flipped","psnr","ssim"]

if not os.path.exists(csv_path):
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# ── STAGE 1 ───────────────────────────────────────────────────────────────────
s1_todo = [(src,site) for src,site in jobs
           if f"{src}:{site}" not in ckpt["s1_done"]]

if s1_todo:
    print(f"{'='*60}")
    print(f"STAGE 1 — {S1_ITERS} iters | 1 chip | {total} images")
    print(f"{'='*60}")
    start_time = datetime.now()
    n_done = len(ckpt["s1_done"])

    for idx,(src,site) in enumerate(s1_todo):
        key    = f"{src}:{site}"
        source = load_image(src)
        c_emb_t    = get_ground_emb(source).detach()
        src_emb_np = c_emb_t.cpu().numpy()

        adv, fw, fc, mg, phase, mom, iw = hierarchical_attack(
            source, c_emb_t, src_emb_np,
            cross_cache[site], same_cache[site],
            S1_ITERS, S1_TOP_K)

        r = {
            "src":           os.path.basename(src), "site": site,
            "stage":         1, "phase": phase,
            "init_wrong":    round(iw,4),
            "final_correct": round(fc,4),
            "final_wrong":   round(fw,4),
            "final_margin":  round(mg,4),
            "margin_shift":  round(mg-(iw-1.0),4),
            "flipped":       mg>0,
            "psnr":          psnr(source, adv),
            "ssim":          ssim_val(source, adv),
        }

        ckpt["s1_results"][key]  = r
        ckpt["s1_done"][key]     = True
        ckpt["s1_momentum"][key] = mom.cpu().flatten().tolist()

        with open(csv_path,"a",newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {k: r.get(k,"") for k in fieldnames})

        n_done += 1
        elapsed   = max(1,(datetime.now()-start_time).seconds)
        remaining = int((elapsed/n_done)*(len(s1_todo)-idx-1))
        n_flip    = sum(1 for v in ckpt["s1_results"].values() if v["flipped"])

        print(f"[S1 {n_done:03d}/{total}] site:{site}  "
              f"phase:{phase:12s}  shift:{r['margin_shift']:+.4f}  "
              f"{'FLIP' if r['flipped'] else '    '}  "
              f"[{n_flip}/{n_done} flipped={100*n_flip/n_done:.1f}%  "
              f"~{remaining//60}m left]")

        if n_done % 25 == 0:
            save_checkpoint(ckpt)
            print(f"  [checkpoint saved]")

    save_checkpoint(ckpt)
    s1_flip = sum(1 for v in ckpt["s1_results"].values() if v["flipped"])
    print(f"\nStage 1: {s1_flip}/{total} flipped ({100*s1_flip/total:.1f}%)")
    print(f"Stage 2 will process {total-s1_flip} failures\n")

# ── STAGE 2 ───────────────────────────────────────────────────────────────────
s1_failures = [(src,site) for src,site in jobs
               if not ckpt["s1_results"].get(f"{src}:{site}",{}).get("flipped",True)]
s2_todo     = [(src,site) for src,site in s1_failures
               if f"{src}:{site}" not in ckpt["s2_done"]]

if s2_todo:
    print(f"{'='*60}")
    print(f"STAGE 2 — {S2_ITERS} iters | top-{S2_TOP_K} | {len(s2_todo)} images")
    print(f"{'='*60}")
    start_time = datetime.now()
    n_done = len(ckpt["s2_done"])

    for idx,(src,site) in enumerate(s2_todo):
        key    = f"{src}:{site}"
        source = load_image(src)
        c_emb_t    = get_ground_emb(source).detach()
        src_emb_np = c_emb_t.cpu().numpy()

        warm_mom = None
        if key in ckpt["s1_momentum"]:
            flat = torch.tensor(ckpt["s1_momentum"][key]).float().to(device)
            n_elem = 1*3*IMG_SIZE*IMG_SIZE
            if len(flat) == n_elem:
                warm_mom = flat.view(1,3,IMG_SIZE,IMG_SIZE)

        adv, fw, fc, mg, phase, _, iw = hierarchical_attack(
            source, c_emb_t, src_emb_np,
            cross_cache[site], same_cache[site],
            S2_ITERS, S2_TOP_K, warm_momentum=warm_mom)

        r = {
            "src":           os.path.basename(src), "site": site,
            "stage":         2, "phase": phase,
            "init_wrong":    round(iw,4),
            "final_correct": round(fc,4),
            "final_wrong":   round(fw,4),
            "final_margin":  round(mg,4),
            "margin_shift":  round(mg-(iw-1.0),4),
            "flipped":       mg>0,
            "psnr":          psnr(source, adv),
            "ssim":          ssim_val(source, adv),
        }

        ckpt["s2_results"][key] = r
        ckpt["s2_done"][key]    = True

        with open(csv_path,"a",newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {k: r.get(k,"") for k in fieldnames})

        n_done += 1
        elapsed   = max(1,(datetime.now()-start_time).seconds)
        remaining = int((elapsed/n_done)*(len(s2_todo)-idx-1))
        n_flip    = sum(1 for v in ckpt["s2_results"].values() if v["flipped"])

        print(f"[S2 {n_done:03d}/{len(s2_todo)}] site:{site}  "
              f"phase:{phase:12s}  shift:{r['margin_shift']:+.4f}  "
              f"{'FLIP' if r['flipped'] else '    '}  "
              f"[recovered:{n_flip}/{n_done}={100*n_flip/n_done:.1f}%  "
              f"~{remaining//60}m left]")

        if n_done % 25 == 0:
            save_checkpoint(ckpt)
            print(f"  [checkpoint saved]")

    save_checkpoint(ckpt)

# ── SUMMARY ───────────────────────────────────────────────────────────────────
s1_results = ckpt["s1_results"]
s2_results = ckpt["s2_results"]
s1_flip    = sum(1 for v in s1_results.values() if v["flipped"])
s2_flip    = sum(1 for v in s2_results.values() if v["flipped"])
combined   = s1_flip + s2_flip

print(f"\n{'='*60}")
print(f"WRIVA MODEL — FINAL RESULTS")
print(f"{'='*60}")
print(f"Stage 1: {s1_flip}/{total} ({100*s1_flip/total:.1f}%)")
if s2_results:
    print(f"Stage 2: {s2_flip}/{len(s2_results)} failures recovered")
print(f"Combined: {combined}/{total} ({100*combined/total:.1f}%)")

print(f"\nPer site:")
for site in ["001","002","003"]:
    s1_s = {k:v for k,v in s1_results.items() if v["site"]==site}
    s2_s = {k:v for k,v in s2_results.items() if v["site"]==site}
    n_s1f = sum(1 for v in s1_s.values() if v["flipped"])
    n_s2f = sum(1 for v in s2_s.values() if v["flipped"])
    nn    = len(s1_s)
    print(f"  DEV-{site}: {n_s1f+n_s2f}/{nn} ({100*(n_s1f+n_s2f)/nn:.1f}%)  "
          f"[s1:{n_s1f}  s2:{n_s2f}]")

print(f"\nCompare to Sample4Geo V11: 78.2% overall")
print(f"Results written to: {csv_path}")
