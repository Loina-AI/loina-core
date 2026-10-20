#!/usr/bin/env python3
"""Prepare JSONL chat cases for loina-quality and report objective checks."""
import argparse
from html import escape
import json
from pathlib import Path
import re


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = []
    seen = set()
    for index, case in enumerate(records(args.suite)):
        name = str(case.get("id", case.get("question_id", index)))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", name) or name in seen:
            raise ValueError(f"Invalid case ID: {name}")
        seen.add(name)
        directory = root / name
        directory.mkdir(exist_ok=True)
        for old in directory.glob("*.txt"):
            old.unlink()
        turns = case.get("turns", [case.get("prompt", "")])
        if not turns or not all(isinstance(t, str) and t.strip() for t in turns):
            raise ValueError(f"Missing prompts: {name}")
        for turn, prompt in enumerate(turns, 1):
            (directory / f"prompt-{turn}.txt").write_text(prompt, encoding="utf-8")
        for file, text in case.get("evidence", {}).items():
            if not re.fullmatch(r"(?:search|page)(?:-[1-4])?\.txt", file):
                raise ValueError(f"Invalid evidence filename: {file}")
            (directory / file).write_text(text, encoding="utf-8")
        target = f"{args.remote.rstrip('/')}/{name}" if args.remote else str(directory)
        tokens = case.get("max_tokens", 1024)
        if not isinstance(tokens, int) or not 1 <= tokens <= 4096:
            raise ValueError(f"Invalid token limit: {name}")
        plan.append(f"{name}\t{int(case.get('tools', args.tools))}\t{tokens}\t{target}\n")
    (root / "plan.tsv").write_text("".join(plan), encoding="utf-8")
    print(f"Prepared {len(plan)} conversations in {root}")


def score(args):
    cases = {str(c.get("id", c.get("question_id", i))): c for i, c in enumerate(records(args.suite))}
    totals = {}
    seen = set()
    for result in records(args.results):
        key = (result["id"], result["turn"])
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        case = cases[result["id"]]
        text = result["answer"] or ""
        checks = {"completed": result["status"] == 0 and bool(text.strip()),
                  "no_control_tags": not re.search(r"<think>|<\|tool_call|<function", text)}
        if case.get("tools"):
            checks["used_web"] = bool(result["calls"])
        rules = case.get("checks", {})
        for word in rules.get("include", []):
            checks[f"include:{word}"] = word.casefold() in text.casefold()
        for word in rules.get("exclude", []):
            checks[f"exclude:{word}"] = word.casefold() not in text.casefold()
        if "min_words" in rules:
            checks["min_words"] = len(text.split()) >= rules["min_words"]
        if "exact" in rules:
            checks["exact"] = text.strip() == rules["exact"]
        if rules.get("no_links"):
            checks["no_links"] = not re.search(r"https?://|\]\(", text)
        languages = case.get("turn_languages")
        language = languages[result["turn"] - 1] if languages else case.get("language", "unknown")
        total = totals.setdefault(language, {"turns": 0, "passed": 0, "checks": 0, "failures": []})
        total["turns"] += 1
        total["passed"] += sum(checks.values())
        total["checks"] += len(checks)
        total["failures"] += [f"{result['id']}:{result['turn']}:{key}" for key, ok in checks.items() if not ok]
    expected = {(name, turn) for name, case in cases.items()
                for turn in range(1, len(case.get("turns", [case.get("prompt")])) + 1)}
    print(json.dumps({"objective_checks_only": True, "missing_turns": sorted(expected - seen),
                      "languages": totals}, ensure_ascii=False, indent=2))


def compare(args):
    runs = [{(r["id"], r["turn"]): r for r in records(path)} for path in [args.before, args.after]]
    sections = []
    for key in dict.fromkeys([*runs[0], *runs[1]]):
        panels = []
        for label, run in zip(["Before", "After"], runs):
            r = run.get(key)
            if r is None:
                panels.append(f"<section><h3>{label}</h3><p>Missing result</p></section>")
                continue
            metrics = r.get("metrics", [])
            backend = metrics[-1].get("backend", "unknown") if metrics else "unknown"
            evidence = "\n\n".join(c["url"] + "\n" + c["result"] for c in r.get("calls", []))
            panels.append(f'<section><h3>{label}</h3><small>{r["seconds"]:.1f}s · '
                          f'{len(r["answer"].split())} words · backend {backend}</small>'
                          f'<pre>{escape(r["answer"])}</pre><details><summary>Tool evidence</summary>'
                          f'<pre>{escape(evidence or "No tools used")}</pre></details></section>')
        prompt = next(run[key]["prompt"] for run in runs if key in run)
        title = escape(f"{key[0]} · turn {key[1]}")
        sections.append(f'<details class="case" {"open" if key[0] == "news-it" else ""}>'
                        f'<summary>{title}</summary><p>{escape(prompt)}</p><div class="answers">'
                        + "".join(panels) + '</div></details>')
    html = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Chat quality comparison</title><style>
body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;background:#f5f5f2;color:#232828}
.case{margin:20px 0;padding:18px;background:white;border:1px solid #ddd;border-radius:10px}
summary{cursor:pointer;font-weight:650}.answers{display:grid;grid-template-columns:1fr 1fr;gap:28px}
pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}small{color:#555}h3{margin-bottom:0}
@media(max-width:700px){.answers{grid-template-columns:1fr}}</style>
<h1>Chat quality comparison</h1><p>Recorded answers and their tool evidence. Replays use fixed recorded or fictional material, not live news.
Review correctness, coverage, freshness, language and usefulness; word counts are not quality scores.</p>
'''
    Path(args.output).write_text(html + "".join(sections), encoding="utf-8")


parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest="command", required=True)
p = commands.add_parser("prepare")
p.add_argument("suite")
p.add_argument("output")
p.add_argument("--remote", help="Case directory on the phone")
p.add_argument("--tools", action="store_true", help="Enable tools unless a case overrides it, matching the app profile")
p.set_defaults(run=prepare)
p = commands.add_parser("score")
p.add_argument("suite")
p.add_argument("results")
p.set_defaults(run=score)
p = commands.add_parser("compare")
p.add_argument("before")
p.add_argument("after")
p.add_argument("output")
p.set_defaults(run=compare)
args = parser.parse_args()
args.run(args)
