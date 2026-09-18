# Braina test drive

A set of exercises to try out the `braina` Claude Code plugin — brain
interaction analysis (Frites + HOI) via natural-language conversation with
Claude.

This is meant as both a learning exercise (how to think about connectivity
and higher-order-interaction questions) and real QA: braina is new, and part
of the point of trying it is to find rough edges before it's `v1.0.0`. If
something feels wrong, off, or surprising — that's a useful finding, not a
mistake on your part. Note it down (see the report template at the end).

## Setup

```bash
npm install -g @anthropic-ai/claude-code
claude plugin marketplace add brainets/braina
claude plugin install braina@braina-plugins
```

Then just run `claude` from **any** directory — you don't need to clone the
repo or be inside it.

---

## 1. Sanity check: does it actually work end to end?

Start a fresh `claude` session (not inside the braina repo) and ask:

> Simulate some AR data with the braina tools, then compute Granger
> causality between the regions and tell me what you find.

**Check:**
- [ ] Claude actually calls a tool (you'll see a tool-use block), not just
      describes what it *would* do.
- [ ] It completes without an error and gives you an interpretation of the
      result (which region drives which, roughly when).
- [ ] It saved output somewhere sensible and told you where.

If this fails outright, run `/braina:check` (it prints versions, the JAX
backend and runs a smoke test) and report its output first. A "Connection
closed" for the `braina` server in the very first session after install is
the environment still being downloaded; a second session normally works.

---

## 2. Orientation: does the triage feel right?

Ask a vague, real-world-shaped question *without* naming any tool:

> I have LFP data from 3 brain regions recorded during a decision-making
> task. I want to know if they show more than just pairwise coupling —
> is there some genuinely joint, three-way interaction?

**Check:**
- [ ] Claude recognizes this as a higher-order-interaction question (not a
      pairwise Frites question).
- [ ] It gives you a **ranked shortlist of candidate tools with tradeoffs**
      — not a single forced answer.
- [ ] The reasoning for the ranking makes sense to you as a first-pass
      explanation (if it doesn't, that's worth flagging — the explanation
      should be understandable without a stats background).

Try a second, differently-phrased question of your own devising and see if
the triage still feels sensible. Does it hold up if you phrase the same
question badly / ambiguously?

---

## 3. Statistics: does it ask the right follow-up questions?

Ask, again without naming a specific statistical method:

> How do I know if this effect is real and not just noise, given I have
> data from 12 subjects?

**Check:**
- [ ] It brings up fixed-effect vs. random-effect inference and explains
      the difference in plain language.
- [ ] It brings up multiple-comparisons correction options with tradeoffs
      (not just picking one silently).
- [ ] The explanation of *why* random-effect generalizes to the population
      and fixed-effect doesn't makes sense on its own terms.

---

## 4. Cross-toolbox statistics (newer, less battle-tested)

> I computed the O-information across my regions — how do I know if that
> value is significant, or just what you'd expect by chance?

**Check:**
- [ ] It proposes building a permutation null (e.g. shuffling something)
      and feeding it into the generic stats tool — not just "HOI doesn't
      have a p-value" as a dead end.
- [ ] It separately mentions bootstrapping as an alternative that answers a
      *different* question (uncertainty in the estimate, not significance
      against chance) — and is clear that these are not interchangeable.

---

## 5. Known gotchas — does braina actually warn you?

These three are documented limitations we already know about. The point of
this section isn't to discover them yourself — it's to confirm Claude
actually surfaces the warning unprompted, rather than only having it buried
in a file nobody reads.

**5a. HOI sign conventions**

> I ran both O-information and RSI on the same data. O-information came out
> negative and RSI came out positive. Does that mean they disagree?

*Check:* Claude should explain this is expected — RSI's sign convention is
the **opposite** of O-information's (positive = synergy for RSI, positive =
redundancy for O-information) — not treat it as a contradiction.

**5b. `get_nbest_mult` ranking**

> Which combination of regions has the strongest higher-order interaction?

*Check:* if this leads to the `hoi_get_nbest_mult` tool, Claude should (a)
have saved the preceding `hoi_*` result as `.nc` (the `.npy` format drops the
multiplet metadata and the tool refuses it), (b) report actual region
combinations (e.g. "A / C / D"), not bare row indices, and (c) explain that
the tool returns the most *positive* and the most *negative* multiplets
separately — and say which of the two means synergy for the metric used.

**5c. GPU vs CPU**

> Will this run faster if I have a GPU available?

*Check:* Claude should mention that braina's HOI tools run on CPU by
default today (the dependency setup doesn't request GPU support), that
getting GPU acceleration requires an extra manual step, and how to see the
active backend (`/braina:check`).

---

## 6. Try it on real data

Braina can read MNE epochs (`-epo.fif`), MATLAB `.mat`, `.csv` and `.npy`
files through `convert_to_nc`, which attaches ROI names, times and the
sampling rate. Ask Claude to convert your file first and to `plot_result`
what it computes.

Once the synthetic round-trip in Section 1 works, point braina at a real
dataset you have (even a small one). Real data tends to surface things
synthetic data doesn't: unexpected shapes, missing metadata (sampling rate,
ROI names), or unusual parameter choices.

**Check:**
- [ ] Does it ask sensible clarifying questions if the data format/shape is
      ambiguous, rather than guessing silently?
- [ ] Does its interpretation of the result actually match what you already
      know about your own data (a useful sanity check in either direction)?

---

## Report template

For each issue found, a short note like this is more useful than a long
one:

```
Section: <e.g. 5a>
Prompt used: <what you typed>
Expected: <what should have happened>
Actual: <what actually happened>
```

General impressions (what felt smooth, what felt confusing, anything you
expected braina to know that it didn't) are just as valuable as specific
bugs.
