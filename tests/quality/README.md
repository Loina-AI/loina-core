# Chat quality evaluation

The shared prompt uses a small set of behaviors drawn from
[OpenAI's research prompting guidance](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)
and [Claude's published prompts](https://platform.claude.com/docs/en/release-notes/system-prompts/claude-sonnet-4-6):
answer the whole question, synthesize evidence, adapt detail and formatting,
check freshness, and identify uncertainty. Public captures of
[ChatGPT](https://github.com/asgeirtj/system_prompts_leaks/blob/main/OpenAI/gpt-5.2-thinking.md)
and [Claude](https://github.com/asgeirtj/system_prompts_leaks/blob/main/Anthropic/claude-sonnet-4.6.md)
were also inspected as unverified examples. Their tool schemas, identities,
product-specific features and long policies are not part of Loina's prompt.

Use [MT-Bench-X](https://github.com/Modalities/MT-Bench-X) for open-ended,
two-turn conversation quality in English, Italian, French, German and Spanish.
All five are among the default LFM model's
[16 supported languages](https://huggingface.co/LiquidAI/LFM3-VL-2B).
Use the original benchmark's judge protocol or blinded human review for quality
scores. A small slice is a regression check, not an official benchmark result.
Its Apache-2.0 data uses `question_id`, `category`, `turns` and `reference` fields;
`tools/quality.py prepare` accepts these JSONL records directly. Add `language`
to report each language separately. Keep upstream revision, question IDs and
references with the answers. Do not translate the benchmark with the model being tested.
MT-Bench-X defaults to tools disabled. Use `prepare --tools` for the app's
tool-enabled prompt profile; keep that profile fixed when comparing prompts.

`web.jsonl` is a separate, original regression suite. Its news and museum
evidence is **fictional** and deliberately fixed to 2026-10-20. Never present
these fixtures as live news. Five news requests test synthesis across topics,
stale stories, a prompt injection, future versus current events and unconfirmed
outcomes. Two requests test unavailable evidence. Sixteen arithmetic/format
turns cover the model's declared languages in one conversation; these check basic instruction
following, not fluency or general knowledge in every language. References are
still shown in the app's Sources area, so the suite checks for unwanted links.

`dates.jsonl` isolates publication dates, older events and future announcements
in Italian, English and Spanish using fictional evidence. Review whether the
past protest stays in the past, the announced strike stays in the future, and
the energy price change is distinguished from household bills. Mechanical
keyword checks alone cannot verify these distinctions. The system prompt gets
the device date, weekday and UTC offset on every model render.

`conversation.jsonl` adds multi-turn coverage, implicit current-information
requests, short follow-ups, quoted English, explicit language changes and failed
searches. Its web evidence is fictional and dated 2026-10-21; record the device
date when replaying it. Review language after English tool results, distinct
topics in an overview, requested item counts, and preservation of qualifications
in follow-ups. These checks apply to any installed chat model. Tool permission
does not mean every follow-up should browse: `checks.web` sets an expected call
and `turn_checks` overrides rules for individual turns.

Build with the same runtime as the app:

```sh
make BUILD=build-eval CMAKE_FLAGS=-DLOINA_EVAL=ON
python3 tools/quality.py prepare tests/quality/web.jsonl /tmp/loina-quality-cases
./build-eval/loina-quality .loina ./build-eval/runtimes/llama/bin ./templates \
  lfm3-vl-2b-qad-q4_0-b5bfc258b7d7cd35 /tmp/loina-quality-cases/plan.tsv > answers.jsonl
python3 tools/quality.py score tests/quality/web.jsonl answers.jsonl
python3 tools/quality.py compare before.jsonl after.jsonl comparison.html
python3 tools/quality.py judge suite.jsonl before.jsonl after.jsonl judge.json
```

The model must be installed or staged in its package's `.partial` directory;
the runner uses normal installation to verify and commit it. `LIBRARIES` points
to the GGML backend directory (ignored for statically registered backends).
Pass another catalog text package ID to evaluate another supported model.
The runner pins the package, CPU backend, full 32K context, four threads,
greedy sampling, seed 42 and thinking off. An optional final argument selects
`Vulkan` or `OpenCL`; verify that backend on the target device before an A/B run.
A case can set `max_tokens`, default
1024. Each conversation gets an independent session; multiple `turns` share it.
Use a separate storage directory for evaluation because sessions are saved.

The native runner also builds in Android's CMake tree with `-DLOINA_EVAL=ON`.
Push it and the prepared cases to a connected phone, use `prepare --remote`
to write device paths into the plan, and run it as the app user with the APK's
native library directory in `LD_LIBRARY_PATH`. Keep runs sequential. Compare
two template directories containing the old and new `system.jinja`; missing
template files use the same bundled versions. Record their hashes, model hash,
build revision, settings and device with the output. All web requests return
the case's `search.txt` or `page.txt`; numbered files override individual calls.
No live networking occurs in this runner.

The scorer reports only transparent mechanical checks and missing turns.
Keyword matches and answer length cannot establish factuality, language quality
or a supported interpretation. Review the saved answers and tool evidence for:

- Correctness: no invented events, numbers, enacted laws or guaranteed effects.
- Freshness: archived reports stay out of today's news; announced events keep their future date.
- Coverage: independent significant topics, deduplicated stories, and useful context.
- Grounding: each substantive claim follows from evidence; limitations are clear.
- Language and presentation: natural requested language, direct opening, readable detail.
- Instruction following: requested format, length and follow-up constraints.

For subjective comparison, hide variant names and swap answer order before
review. Report wins, ties and losses with reasons instead of manufacturing an
overall benchmark score. `judge` prepares complete conversation histories,
references, evidence and a five-dimension rubric with shuffled A/B labels.
Keep its `.key.json` file out of the judge's input. Save the judge's identity,
per-pair scores and reasons before unblinding. The tool does not call an API or
grade its own answers. If the same agent develops and judges a change, disclose
that limitation; a label-blinded development review is not an independent or
official MT-Bench-X evaluation. Run the Android `webAnswers` instrumentation separately
for live search and `browserExtraction` for deterministic WebView extraction.
Live news changes between runs and must not be used as the only A/B evidence.
