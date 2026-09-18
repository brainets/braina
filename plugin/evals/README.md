# Braina eval suite

Behavioural checks for the plugin, one directory per case (`prompt.md` +
`graders/`). They mirror the manual test drive in
`usecases/plugin_testing/TESTING.md`.

```bash
claude plugin eval plugin                      # all cases
claude plugin eval plugin --case gpu-question  # one case
claude plugin eval plugin --json results.json --threshold 0.8
```

`results/` is written by each run and is git-ignored.
