"""Compare Pegasus->Kimi and Nova->Kimi analyses of one video.   uv run python scripts/compare_pipelines.py uaQa-9xmfhs"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VID = sys.argv[1] if len(sys.argv) > 1 else "uaQa-9xmfhs"
NOVA_FILE = sys.argv[2] if len(sys.argv) > 2 else "B"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def toks(s):
    return {t for t in norm(s).split() if len(t) > 2}


def nums(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def best_match(claim, others):
    a = toks(claim)
    best, bj = None, 0.0
    for o in others:
        b = toks(o)
        j = len(a & b) / max(1, min(len(a), len(b)))  # containment of the smaller side
        if j > bj:
            best, bj = o, j
    return best, bj


def main():
    peg = json.loads((ROOT / "data/poc20" / f"{VID}.record.json").read_text())
    s1 = json.loads((ROOT / "data/poc20" / f"{VID}.stage1.json").read_text())
    nova = json.loads((ROOT / "data/kimi_test" / f"{VID}.{NOVA_FILE}.json").read_text())
    fr = json.loads((ROOT / "data/frames20" / f"{VID}.frames.json").read_text())
    caps = json.loads((ROOT / "data/poc20" / f"{VID}.captions.json").read_text())
    P, N = peg["document"], nova
    cap_txt = " ".join(c["text"] for c in caps)
    peg_txt = " ".join(s["on_screen_text"] + " " + s["title"] + " " + s["summary"] for s in s1["segments"])
    nova_txt = " ".join(f["on_screen_text"] + " " + f["slide_title"] + " " + f["description"] for f in fr["frames"])
    src_nums = nums(cap_txt + " " + peg_txt + " " + nova_txt)

    def stats(name, doc, own_txt):
        cl = doc["claims"]
        q = [norm(c.get("quote", "")) for c in cl]
        in_caps = sum(1 for x in q if x and x in norm(cap_txt))
        in_own = sum(1 for x in q if x and x in norm(cap_txt + " " + own_txt))
        cn = set().union(*[nums(c["claim"]) for c in cl]) if cl else set()
        print(f"\n{name}: claims={len(cl)} types={ {t: sum(c['claim_type'] == t for c in cl) for t in ('fact', 'quantitative', 'forecast', 'opinion', 'recommendation')} }")
        print(f"  quotes verified: captions {in_caps}/{len(cl)} | captions+own slide text {in_own}/{len(cl)} | empty quote {sum(1 for x in q if not x)}")
        print(f"  numbers in claims: {len(cn)}; not present in ANY source (captions/Pegasus/Nova): {sorted(cn - src_nums)}")
        print(f"  entities {len(doc['entities'])} | speakers {[s['name'] for s in doc['speakers']]} | topics {len(doc['topics'])} | source_docs {len(doc['source_documents'])} | flags {len(doc['quality_flags'])}")
        return cn

    pn = stats("PEGASUS -> Kimi", P, peg_txt)
    nn = stats("NOVA    -> Kimi", N, nova_txt)

    pc, nc = [c["claim"] for c in P["claims"]], [c["claim"] for c in N["claims"]]
    thr = 0.6
    p_in_n = [c for c in pc if best_match(c, nc)[1] >= thr]
    n_in_p = [c for c in nc if best_match(c, pc)[1] >= thr]
    print(f"\nclaim overlap (token containment >= {thr}): {len(p_in_n)}/{len(pc)} Pegasus claims have a Nova match; {len(n_in_p)}/{len(nc)} Nova claims have a Pegasus match")
    print("\nPegasus-only claims (sample):")
    for c in [c for c in pc if c not in p_in_n][:8]:
        print("  -", c[:150])
    print("\nNova-only claims (sample):")
    for c in [c for c in nc if c not in n_in_p][:8]:
        print("  -", c[:150])
    print(f"\nnumbers only in Pegasus claims: {sorted(pn - nn)}\nnumbers only in Nova claims:    {sorted(nn - pn)}")
    pe, ne = {e['name'].lower() for e in P['entities']}, {e['name'].lower() for e in N['entities']}
    print(f"\nentities: Pegasus {len(pe)} vs Nova {len(ne)}; Pegasus-only: {sorted(pe - ne)[:14]}")
    print("\nsummary lengths (words): Pegasus", len(P['summary'].split()), "| Nova", len(N['summary'].split()))
    print("\ncost: Pegasus stage1 $%.2f + stage2 $%.2f = $%.2f | Nova stage1 $%.3f + stage2 ~$0.36 (cache-aware)" % (
        s1['meta']['usd'], peg['analysis']['stage2']['usd_est'], s1['meta']['usd'] + peg['analysis']['stage2']['usd_est'], fr['meta']['usd']))


if __name__ == "__main__":
    main()
