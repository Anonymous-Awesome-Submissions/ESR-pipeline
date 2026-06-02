"""Wikipedia revision split builder for cross-domain validation.

Pulls revision histories for a curated list of ~250 Wikipedia pages where current
facts have changed across diverse domains (CEO changes, sports manager changes,
head-of-state, software latest version, etc.). Each revision is an event; the
question is the current effective value of attribute X for entity Y, with the
phenomenon label derived from the revision-history pattern.

Output: data/processed/streams_wiki_v4.jsonl  (the `wiki_v4` split used in the paper)
"""
from __future__ import annotations

import os, json, logging, time, urllib.request
from pathlib import Path
from urllib.parse import urlencode

PROJECT_ROOT = Path(os.environ.get("ESR_ROOT", Path(__file__).resolve().parents[1]))
PROC_DIR = PROJECT_ROOT / "data" / "processed"
# PROC_DIR.mkdir is deferred to main() — see the rationale in github_api_pull.py.

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("wiki")

# Expanded curated page list — diverse domains where current facts evolve
SEED_PAGES = [
    # --- Tech CEOs (companies that have had leadership changes) ---
    ("Microsoft", "current CEO"), ("Apple_Inc.", "current CEO"), ("Twitter", "current CEO"),
    ("Reddit", "current CEO"), ("OpenAI", "current CEO"), ("Alphabet_Inc.", "current CEO"),
    ("Meta_Platforms", "current CEO"), ("Amazon_(company)", "current CEO"),
    ("Tesla,_Inc.", "current CEO"), ("Netflix", "current CEO"), ("Spotify", "current CEO"),
    ("Uber", "current CEO"), ("Airbnb", "current CEO"), ("Lyft", "current CEO"),
    ("Disney", "current CEO"), ("Boeing", "current CEO"), ("Goldman_Sachs", "current CEO"),
    ("JPMorgan_Chase", "current CEO"), ("Morgan_Stanley", "current CEO"),
    ("Bank_of_America", "current CEO"), ("Citigroup", "current CEO"),
    ("Wells_Fargo", "current CEO"), ("BlackRock", "current CEO"),
    ("Berkshire_Hathaway", "current CEO"), ("Walmart", "current CEO"),
    ("Coca-Cola", "current CEO"), ("PepsiCo", "current CEO"), ("Pfizer", "current CEO"),
    ("Johnson_&_Johnson", "current CEO"), ("Procter_&_Gamble", "current CEO"),
    ("ExxonMobil", "current CEO"), ("Chevron_Corporation", "current CEO"),
    ("General_Electric", "current CEO"), ("Ford_Motor_Company", "current CEO"),
    ("General_Motors", "current CEO"), ("Toyota", "current CEO"),
    ("Volkswagen_Group", "current CEO"), ("BMW", "current CEO"),
    ("Mercedes-Benz_Group", "current CEO"), ("Stellantis", "current CEO"),
    ("Samsung_Electronics", "current CEO"), ("Sony_Group_Corporation", "current CEO"),
    ("LG_Corporation", "current CEO"), ("IBM", "current CEO"), ("Oracle_Corporation", "current CEO"),
    ("Intel", "current CEO"), ("AMD", "current CEO"), ("Nvidia", "current CEO"),
    ("Qualcomm", "current CEO"), ("Adobe_Inc.", "current CEO"), ("Salesforce", "current CEO"),
    ("Snap_Inc.", "current CEO"), ("Pinterest", "current CEO"), ("Square,_Inc.", "current CEO"),
    ("PayPal", "current CEO"), ("Stripe,_Inc.", "current CEO"),
    # --- Software latest versions ---
    ("Python_(programming_language)", "current stable version"),
    ("Linux_kernel", "current stable version"), ("Visual_Studio_Code", "current stable version"),
    ("Mozilla_Firefox", "current stable version"), ("Google_Chrome", "current stable version"),
    ("Microsoft_Edge", "current stable version"), ("Safari_(web_browser)", "current stable version"),
    ("Node.js", "current stable version"), ("PostgreSQL", "current stable version"),
    ("MySQL", "current stable version"), ("MariaDB", "current stable version"),
    ("MongoDB", "current stable version"), ("Redis", "current stable version"),
    ("Kubernetes", "current stable version"), ("Docker_(software)", "current stable version"),
    ("PyTorch", "current stable version"), ("TensorFlow", "current stable version"),
    ("React_(software)", "current stable version"), ("Vue.js", "current stable version"),
    ("Angular_(web_framework)", "current stable version"),
    ("Django_(web_framework)", "current stable version"), ("Flask_(web_framework)", "current stable version"),
    ("Ruby_on_Rails", "current stable version"), ("Spring_Framework", "current stable version"),
    ("Java_(programming_language)", "current stable version"), ("Go_(programming_language)", "current stable version"),
    ("Rust_(programming_language)", "current stable version"), ("Swift_(programming_language)", "current stable version"),
    ("Kotlin_(programming_language)", "current stable version"),
    ("TypeScript", "current stable version"), ("Ubuntu", "current stable version"),
    ("Debian", "current stable version"), ("Fedora_Linux", "current stable version"),
    ("Arch_Linux", "current stable version"), ("MacOS", "current stable version"),
    ("Windows_11", "current stable version"), ("IOS", "current stable version"),
    ("Android_(operating_system)", "current stable version"),
    # --- Heads of state ---
    ("List_of_Prime_Ministers_of_the_United_Kingdom", "current PM"),
    ("List_of_presidents_of_the_United_States", "current president"),
    ("List_of_Chancellors_of_Germany", "current Chancellor"),
    ("List_of_presidents_of_France", "current president"),
    ("List_of_prime_ministers_of_Japan", "current PM"),
    ("List_of_prime_ministers_of_Canada", "current PM"),
    ("List_of_prime_ministers_of_India", "current PM"),
    ("List_of_prime_ministers_of_Australia", "current PM"),
    ("List_of_presidents_of_Italy", "current president"),
    ("List_of_prime_ministers_of_Italy", "current PM"),
    ("List_of_prime_ministers_of_Spain", "current PM"),
    ("List_of_prime_ministers_of_the_Netherlands", "current PM"),
    ("List_of_presidents_of_Brazil", "current president"),
    ("List_of_presidents_of_Mexico", "current president"),
    ("List_of_presidents_of_Argentina", "current president"),
    ("List_of_presidents_of_South_Korea", "current president"),
    ("List_of_presidents_of_Indonesia", "current president"),
    ("List_of_presidents_of_Turkey", "current president"),
    ("List_of_presidents_of_Russia", "current president"),
    ("List_of_presidents_of_China", "current president"),
    ("Pope_Francis", "current pope"),
    ("Secretary-General_of_the_United_Nations", "current SG"),
    ("President_of_the_European_Commission", "current president"),
    ("List_of_Secretaries-General_of_NATO", "current SG"),
    # --- Sports league managers/coaches ---
    ("Manchester_United_F.C.", "current manager"), ("Real_Madrid_CF", "current manager"),
    ("FC_Barcelona", "current manager"), ("Liverpool_F.C.", "current manager"),
    ("Bayern_Munich", "current manager"), ("Paris_Saint-Germain_F.C.", "current manager"),
    ("Arsenal_F.C.", "current manager"), ("Manchester_City_F.C.", "current manager"),
    ("Chelsea_F.C.", "current manager"), ("Tottenham_Hotspur_F.C.", "current manager"),
    ("Juventus_F.C.", "current manager"), ("Inter_Milan", "current manager"),
    ("AC_Milan", "current manager"), ("Atletico_Madrid", "current manager"),
    ("Borussia_Dortmund", "current manager"), ("Ajax_(football_club)", "current manager"),
    ("Boston_Celtics", "current head coach"), ("Los_Angeles_Lakers", "current head coach"),
    ("Golden_State_Warriors", "current head coach"), ("Miami_Heat", "current head coach"),
    ("Philadelphia_76ers", "current head coach"), ("Chicago_Bulls", "current head coach"),
    ("New_York_Yankees", "current manager"), ("Los_Angeles_Dodgers", "current manager"),
    ("Boston_Red_Sox", "current manager"), ("New_England_Patriots", "current head coach"),
    ("Dallas_Cowboys", "current head coach"), ("Green_Bay_Packers", "current head coach"),
    ("Toronto_Maple_Leafs", "current head coach"), ("Montreal_Canadiens", "current head coach"),
    ("New_Zealand_national_rugby_union_team", "current head coach"),
    # --- Universities ---
    ("Massachusetts_Institute_of_Technology", "current president"),
    ("Harvard_University", "current president"), ("Stanford_University", "current president"),
    ("Yale_University", "current president"), ("Princeton_University", "current president"),
    ("Columbia_University", "current president"), ("Cornell_University", "current president"),
    ("University_of_Pennsylvania", "current president"),
    ("University_of_California,_Berkeley", "current chancellor"),
    ("California_Institute_of_Technology", "current president"),
    ("University_of_Cambridge", "current vice-chancellor"),
    ("University_of_Oxford", "current vice-chancellor"),
    ("Imperial_College_London", "current president"),
    ("ETH_Zurich", "current president"), ("National_University_of_Singapore", "current president"),
    ("University_of_Tokyo", "current president"), ("Tsinghua_University", "current president"),
    ("Peking_University", "current president"),
    # --- City populations ---
    ("New_York_City", "current population"), ("Tokyo", "current population"),
    ("London", "current population"), ("Paris", "current population"),
    ("Beijing", "current population"), ("Shanghai", "current population"),
    ("Mumbai", "current population"), ("Delhi", "current population"),
    ("Cairo", "current population"), ("Lagos", "current population"),
    ("Mexico_City", "current population"), ("Sao_Paulo", "current population"),
    ("Buenos_Aires", "current population"), ("Moscow", "current population"),
    ("Istanbul", "current population"), ("Seoul", "current population"),
    ("Bangkok", "current population"), ("Jakarta", "current population"),
    ("Singapore", "current population"), ("Sydney", "current population"),
    ("Toronto", "current population"), ("Berlin", "current population"),
    ("Madrid", "current population"), ("Rome", "current population"),
    ("Los_Angeles", "current population"), ("Chicago", "current population"),
    # --- Bands / artists with member changes ---
    ("Coldplay", "current members"), ("Pink_Floyd", "current members"),
    ("Metallica", "current members"), ("Radiohead", "current members"),
    ("U2", "current members"), ("Foo_Fighters", "current members"),
    ("Red_Hot_Chili_Peppers", "current members"), ("Imagine_Dragons", "current members"),
    ("Maroon_5", "current members"), ("Linkin_Park", "current members"),
    ("Aerosmith", "current members"), ("Bon_Jovi", "current members"),
    ("Iron_Maiden", "current members"), ("Black_Sabbath", "current members"),
    # --- Movies / shows ---
    ("List_of_Marvel_Cinematic_Universe_films", "latest released film"),
    ("Game_of_Thrones", "current showrunner"), ("Stranger_Things", "current status"),
    ("Star_Wars", "latest film"), ("The_Crown_(TV_series)", "current showrunner"),
    ("Better_Call_Saul", "current status"), ("Westworld_(TV_series)", "current status"),
    # --- Tech / AI ---
    ("ChatGPT", "current model"), ("DeepMind", "current parent company"),
    ("GPT-4", "current capabilities"), ("Claude_(language_model)", "current version"),
    ("Gemini_(language_model)", "current version"), ("Mistral_AI", "current model"),
    ("Cohere", "current model"),
    # --- Controversies / definitions ---
    ("Pluto", "planet status"), ("Brexit", "current status"),
    ("COVID-19", "current status"),
    ("List_of_countries_in_the_European_Union", "current member states"),
    # --- Companies in flux (ownership changes) ---
    ("Twitter", "current owner"), ("WhatsApp", "current owner"),
    ("LinkedIn", "current owner"), ("YouTube", "current owner"),
    ("Activision_Blizzard", "current owner"), ("ZeniMax_Media", "current owner"),
    ("Bungie", "current owner"), ("Mojang_Studios", "current owner"),
    # --- Currencies / rates ---
    ("Bitcoin", "current notable claims"), ("Ethereum", "current notable claims"),
    # --- Stadiums / sports venues ---
    ("Wembley_Stadium", "current capacity"), ("Camp_Nou", "current capacity"),
    ("Old_Trafford", "current capacity"), ("Madison_Square_Garden", "current capacity"),
]

# Deduplicate
seen = set()
DEDUP_PAGES = []
for p, a in SEED_PAGES:
    if p not in seen:
        seen.add(p); DEDUP_PAGES.append((p, a))
print(f"Unique pages to fetch: {len(DEDUP_PAGES)}")


def fetch_revisions(page: str, limit: int = 30, max_retries: int = 5) -> list[dict]:
    params = {
        "action": "query", "prop": "revisions", "titles": page,
        "rvprop": "ids|timestamp|user|comment|content", "rvslots": "main",
        "rvlimit": str(min(limit, 50)), "format": "json", "formatversion": "2",
    }
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    delay = 2.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ESR-Bench/0.2 (research)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            pages = (data.get("query") or {}).get("pages") or []
            if not pages: return []
            return pages[0].get("revisions") or []
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log.info(f"  429 on {page}, retry in {delay:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(delay); delay *= 2
            else:
                raise
    return []


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "streams_wiki_v4.jsonl"
    out = []
    for i, (page, attr_label) in enumerate(DEDUP_PAGES):
        try:
            revs = fetch_revisions(page, limit=20)
            log.info(f"[{i+1}/{len(DEDUP_PAGES)}] {page}: {len(revs)} revisions")
            if len(revs) < 3:
                continue
            revs = list(reversed(revs))
            events = []
            for r in revs:
                content = (r.get("slots") or {}).get("main", {}).get("content") or ""
                content = content[:2400]
                events.append({
                    "event_id": f"wiki_{page}_{r.get('revid')}",
                    "kind": "revision",
                    "timestamp": r.get("timestamp"),
                    "text": (r.get("comment") or "") + "\n\n" + content,
                    "actor": r.get("user"),
                    "role": "editor",
                    "meta": {"revid": r.get("revid"), "comment": r.get("comment")},
                })
            stream = {
                "stream_id": f"wiki_{page}",
                "repo": "wikipedia", "issue_number": page, "title": page.replace("_", " "),
                "current_state": "active", "current_labels": [],
                "created_at": events[0]["timestamp"], "closed_at": None,
                "n_comments": len(events), "events": events,
                "wiki_attribute_hint": attr_label,
            }
            out.append(stream)
        except Exception as e:
            log.warning(f"  fetch failed for {page}: {e}")
            continue
        time.sleep(1.5)

    with open(out_path, "w") as f:
        for s in out: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info(f"wrote {out_path} ({len(out)} streams)")

if __name__ == "__main__":
    main()
