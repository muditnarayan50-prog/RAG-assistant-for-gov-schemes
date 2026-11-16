"""10-question check: 5 schemes that ARE in the database, 5 that are NOT.
Run from the project folder:
    python run_test.py
Writes test_results.md (full answers, read these) and test_results.csv.
The automatic PASS/CHECK is only a first filter. Read every answer yourself.
"""
import csv, sys, time
from answer import AnswerGenerator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# (question, expected: list of facts that must appear, scheme-name fragment that must appear in the sources)
IN_DB = [
    ("What are the minimum and maximum age to join Atal Pension Yojana?",
     ["18", "40"], "atal pension"),
    ("What are the loan categories under Pradhan Mantri Mudra Yojana?",
     ["shishu", "kishore", "tarun"], "mudra"),
    ("How much scholarship per year does the National Means-cum-Merit Scholarship Scheme give?",
     ["12,000"], "means-cum-merit"),
    ("What is the yearly premium under Pradhan Mantri Jeevan Jyoti Bima Yojana?",
     ["436"], "jeevan jyoti"),
    ("Is a minimum balance required in a Basic Savings Bank Deposit Account under Pradhan Mantri Jan Dhan Yojana?",
     ["minimum balance"], "jan dhan"),
]
NOT_IN_DB = [
    "What is the eligibility for Sukanya Samriddhi Yojana?",
    "What loan amount does the Stand-Up India scheme offer?",
    "What is the health insurance cover under Ayushman Bharat PM-JAY?",
    "How much subsidy is given for rooftop solar under PM Surya Ghar Muft Bijli Yojana?",
    "How much monthly assistance does the Gruha Lakshmi scheme give to women?",
]

def main():
    g = AnswerGenerator()
    rows, md = [], ["# Test results: 5 in database, 5 not in database\n"]
    jobs = [("IN", q, exp, name) for q, exp, name in IN_DB] + [("NOT", q, None, None) for q in NOT_IN_DB]
    for kind, q, exp, name in jobs:
        t = time.perf_counter()
        r = g.generate(q)
        secs = round(time.perf_counter() - t, 1)
        text = g.format_answer(r)
        low = text.lower()
        if kind == "IN":
            facts_ok = all(f.lower() in low for f in exp)
            name_ok = name in low
            verdict = "PASS" if (not r.abstained and facts_ok and name_ok) else "CHECK"
            detail = f"abstained={r.abstained} facts_found={facts_ok} scheme_in_sources={name_ok}"
        else:
            verdict = "PASS" if r.abstained else "CHECK"
            detail = f"abstained={r.abstained} (should be True)"
        rows.append([kind, q, verdict, round(float(r.best_score), 3), secs, detail])
        md.append(f"## [{kind}] {q}\n**{verdict}** | score {r.best_score:.3f} | {secs}s | {detail}\n\n{text}\n")
        print(f"[{kind}] {verdict:5s} score {r.best_score:.3f} {secs:5.1f}s  {q[:70]}", flush=True)
    with open("test_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["type", "question", "verdict", "best_score", "seconds", "detail"]); w.writerows(rows)
    open("test_results.md", "w", encoding="utf-8").write("\n".join(md))
    p_in = sum(1 for x in rows if x[0] == "IN" and x[2] == "PASS")
    p_out = sum(1 for x in rows if x[0] == "NOT" and x[2] == "PASS")
    print(f"\nIn database: {p_in}/5 pass | Not in database: {p_out}/5 abstained")
    print("Read test_results.md before trusting these numbers.")

if __name__ == "__main__":
    main()
