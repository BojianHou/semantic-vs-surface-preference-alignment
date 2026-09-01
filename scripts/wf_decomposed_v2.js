export const meta = {
  name: 'decomposed-v2-resume',
  description: 'Write/repair the v2 counterfactual shards that are still outstanding (resumable from disk state)',
  phases: [
    { title: 'Write', detail: 'writer agents for shards with no output yet' },
    { title: 'Audit', detail: 'auditor agents: repair token length, semantics, syntax control' },
  ],
}

// args: { root: "data/decomposed_v2", missing: [...], dirty: [...] }
const ROOT = '/data/users/bojianhou/projects/preference_alignment'
const ENV = 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'
const DIR = args.root
const MISSING = args.missing || []
const DIRTY = args.dirty || []
const pad = (i) => String(i).padStart(3, '0')

const SPEC = `
## The study you are building data for

We measure whether a preference-aligned model judges summaries by MEANING or by SURFACE FORM:
  g1 = score(chosen) - score(rejected_prime)                -> pure SEMANTICS
  g2 = score(rejected_prime) - score(rejected_double_prime) -> pure SYNTAX
  g3 = score(rejected_double_prime) - score(rejected)       -> pure LENGTH
For those to isolate what they claim, the counterfactuals must be surgically controlled.

## What every variant must satisfy

1. MEANING = that of \`rejected\`. Same claims, same stance, same information.
2. NEVER drift toward \`chosen\`'s meaning or information. Most important rule.
3. Fluent, natural English, same register as the originals (a Reddit TL;DR summary).
4. Begins with EXACTLY ONE leading space, like the input strings.
5. EXACTLY \`chosen_tokens\` (call it N) tokens under the Qwen2.5 tokenizer. Not "about" N.
6. Not byte-identical to \`rejected\` or to each other.

## The four variants per example

1. \`rejected_double_prime\` - rejected's meaning at N tokens, KEEPING REJECTED'S OWN SYNTACTIC
   STRUCTURE (same clause order, voice, sentence count as \`rejected\` as far as possible).
2. \`rejected_prime\` - rejected's meaning at N tokens, MIRRORING CHOSEN'S SYNTAX: same number
   of sentences, same clause order, same voice, similar connectives as \`chosen\`. Content
   stays rejected's.
3. \`rejected_double_prime_v2\` - independent second version of (1).
4. \`rejected_prime_v2\` - independent second version of (2).

## THE FAILURE MODE YOU MUST AVOID

In the previous dataset \`rejected_prime\` and \`rejected_double_prime\` differed by 46% of their
characters - two unrelated paraphrases, so the "syntax" gap really measured vocabulary. Your (1)
and (2) must differ ONLY IN SYNTAX: reuse the same content words and REARRANGE them. "Same
sentence, restructured", not "two ways of saying it". Same for the v2 pair.

## Token counting (use the real tokenizer, do not guess)

Batch your checks - do NOT call this once per string:

  cd ${ROOT} && echo '["  text one","  text two", ...]' | ${ENV} .venv/bin/python scripts/count_tokens.py 2>/dev/null

prints a JSON list of counts in order. Draft all four variants for all examples, check in ONE
batched call, revise only the misses, re-check. Note the count is taken on the string WITH its
single leading space - include it. Adjust with content-neutral edits (contractions, "and"->",",
articles, tightening) that change neither meaning nor the syntax constraint.
`

const WRITE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['shard', 'n_examples', 'n_token_exact', 'notes'],
  properties: {
    shard: { type: 'integer' }, n_examples: { type: 'integer' },
    n_token_exact: { type: 'integer' }, notes: { type: 'string' },
  },
}
const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['shard', 'n_checked', 'n_repaired', 'n_still_failing', 'failure_summary'],
  properties: {
    shard: { type: 'integer' }, n_checked: { type: 'integer' }, n_repaired: { type: 'integer' },
    n_still_failing: { type: 'integer' }, failure_summary: { type: 'string' },
  },
}

const writerPrompt = (i) => `You are a careful linguistic data author building a controlled counterfactual dataset.

Input shard:  ${ROOT}/${DIR}/shards/shard_${pad(i)}.json
Output file:  ${ROOT}/${DIR}/out/out_${pad(i)}.json

Read the input shard: a JSON list with id, prompt (a Reddit post), chosen, rejected,
chosen_tokens (= N), rejected_tokens.

${SPEC}

## Output

Write the output file as a JSON list, one object per input example, EVERY example in the shard,
with exactly these fields:
  id, rejected_double_prime, rejected_prime, rejected_double_prime_v2, rejected_prime_v2

No commentary in the JSON. Verify it parses before finishing.`

const auditorPrompt = (i) => `You are an independent AUDITOR for a controlled counterfactual dataset. A previous agent wrote
these rewrites; catch and FIX its mistakes. Be skeptical - assume it cut corners on the hard
constraints.

Input shard (ground truth): ${ROOT}/${DIR}/shards/shard_${pad(i)}.json
File to audit and repair:   ${ROOT}/${DIR}/out/out_${pad(i)}.json

${SPEC}

## Audit checklist, per example

A. TOKEN LENGTH: all four variants exactly N = chosen_tokens, measured WITH the leading space.
   Verify with a batched call to scripts/count_tokens.py. Most commonly violated.
B. MEANING PRESERVED: each variant still says what \`rejected\` says.
C. NO SEMANTIC LEAKAGE toward \`chosen\`. Check \`rejected_prime\` hardest - mimicking chosen's
   syntax tempts the writer into copying chosen's content. Rewrite if it leaked.
D. SYNTAX CONTROL: \`rejected_prime\` mirrors \`chosen\`'s structure; \`rejected_double_prime\`
   keeps \`rejected\`'s. They must differ in STRUCTURE while sharing VOCABULARY. If they read as
   two unrelated paraphrases, rewrite so they are minimal structural variants of each other.
E. DISTINCTNESS: nothing byte-identical to \`rejected\` or to another variant.
F. LEADING SPACE: exactly one, on every variant.

If the file is missing or unparseable, WRITE IT FROM SCRATCH per the spec above.
Repair in place, OVERWRITE the file with corrected JSON (same schema), re-verify token counts
after editing. Report honestly - do not claim success you did not achieve.`

// Shards with no usable output: write, then audit. Pipelined so each audits as soon as it is written.
const fresh = MISSING.length ? await pipeline(
  MISSING,
  (i) => agent(writerPrompt(i), { label: `write:${pad(i)}`, phase: 'Write', schema: WRITE_SCHEMA }),
  (prev, i) => agent(auditorPrompt(i), { label: `audit:${pad(i)}`, phase: 'Audit', schema: AUDIT_SCHEMA }),
) : []

// Shards already written but failing QC: repair only, no rewrite from scratch.
const repaired = DIRTY.length ? await parallel(
  DIRTY.map((i) => () => agent(auditorPrompt(i), { label: `repair:${pad(i)}`, phase: 'Audit', schema: AUDIT_SCHEMA }))
) : []

const all = [...fresh, ...repaired].filter(Boolean)
const tot = (f) => all.reduce((s, r) => s + (r[f] || 0), 0)
log(`${DIR}: ${all.length} shard-jobs done; repaired ${tot('n_repaired')}, still failing ${tot('n_still_failing')}`)
return {
  dir: DIR,
  written: fresh.filter(Boolean).length,
  repaired: repaired.filter(Boolean).length,
  still_failing: tot('n_still_failing'),
}
