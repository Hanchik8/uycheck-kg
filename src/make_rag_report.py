import json
import os

ROOT = r"C:\Users\user\.gemini\antigravity\scratch\uycheck-kg"
with open(os.path.join(ROOT, "tests", "rag_eval_20_results.json"), encoding="utf-8") as f:
    data = json.load(f)

lines = [
    "# UyCheck KG — 20-Question RAG Evaluation Report",
    "",
    f"**Score:** **{data['passed']} / {data['total']} ({data['accuracy_pct']}%)**  ",
    "**Corpus:** 17 official Kyrgyz Republic documents (GIK KR, Law No. 153, Housing Code No. 117, GU Cadastre, DDU, KR Tax Code)  ",
    "**Languages tested:** Russian (14), English (4), Kyrgyz (2) — including 2 out-of-scope refusal tests.",
    "",
    "| # | Lang | Scope | Result | Question | Retrieved Sources |",
    "|---|------|-------|--------|----------|-------------------|",
]
for r in data["results"]:
    scope = "In-doc" if r["in_scope"] else "Out-of-doc"
    res = "PASS" if r["passed"] else "FAIL"
    srcs = ", ".join(r["sources"]) if r["sources"] else "Refused (I don't know)"
    q = r["question"].replace("|", "/")
    lines.append(f"| {r['id']} | {r['language'].upper()} | {scope} | **{res}** | {q} | {srcs} |")

out_path = os.path.join(ROOT, "RAG_TEST_RESULTS.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Saved RAG_TEST_RESULTS.md:", os.path.getsize(out_path), "bytes")
