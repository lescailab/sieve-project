# Carbon Footprint

Training and explaining a deep learning model on an exome cohort costs energy,
and that cost is rarely reported. SIEVE measures it directly, so a paper or a
report can state the compute footprint of its analysis as a measured quantity
rather than an estimate reconstructed after the fact.

Measurement uses [CodeCarbon](https://codecarbon.io), which samples CPU, GPU and
RAM energy draw during a run and converts it to CO2-equivalent emissions using
the carbon intensity of the local electricity grid.

!!! warning "Experimental, and not in the released package"

    This feature currently lives on the `co2footprint` branch and is **not**
    part of an installed release. If you `pip install sieve` or install the
    conda package, `sieve-co2-report` will not be on your `PATH` and no
    measurements will be written.

    To use it, work from a checkout of that branch:

    ```bash
    git clone https://github.com/lescailab/sieve-project.git
    cd sieve-project
    git checkout co2footprint
    pip install -e .
    ```

    The interface described below may change before it is merged.

## Installing CodeCarbon

CodeCarbon is installed from PyPI. Its current releases are not maintained as
conda packages, so install it with `pip` even inside a conda environment:

```bash
python -m pip install "codecarbon>=3.2.8,<4"
```

For a conda environment created for SIEVE:

```bash
conda run -n sieve python -m pip install "codecarbon>=3.2.8,<4"
```

## How measurement works

Two stages are instrumented, because they are the two that dominate compute:

| Stage | Command | Recorded as |
|-------|---------|-------------|
| Training | `sieve-train` | `sieve-train` |
| Explainability | `sieve-explain` | `sieve-explain` |

Each run appends one row to a CSV below the output directory the command was
already given, so measurements sit alongside the results they belong to:

```
experiments/my_model/co2footprint/emissions.csv
results/explainability/co2footprint/emissions.csv
```

Rows are appended rather than overwritten, so repeated runs into the same
output directory accumulate rather than replacing one another.

### Measurement is fail-open

Tracking never changes what a command returns or raises. If CodeCarbon is not
installed, cannot start, or fails to write, the command prints a warning to
standard error and carries on with the analysis:

```
WARNING: CO2 footprint tracking is unavailable; continuing without measurement
```

This is deliberate: a telemetry problem must never cost you a training run.
The practical consequence is that a missing measurement is a warning, not an
error, so check for the CSV if you expect one.

Uninstalling CodeCarbon is therefore also how you turn measurement off. There
is no flag to disable it.

### Tracking mode

Measurement runs in CodeCarbon's `machine` mode, which attributes the whole
machine's energy draw to the run. On a dedicated node or an exclusive GPU
allocation this is what you want. On a shared host it will also capture whatever
else is running, so treat those numbers as an upper bound.

## Generating a report

`sieve-co2-report` compiles the per-stage CSVs into one consolidated report:

```bash
sieve-co2-report \
    --input experiments/my_model \
    --input results/explainability \
    --output-dir results/co2_footprint
```

`--input` is repeatable and accepts either a CodeCarbon CSV directly or a
directory, which is scanned recursively for `co2footprint/emissions.csv`. That
means you can usually point it at a project directory and let it find
everything:

```bash
sieve-co2-report --input results/my_cohort --output-dir results/co2_footprint
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path | required | CodeCarbon CSV, or a directory scanned recursively for `co2footprint/emissions.csv`. Repeat for multiple sources. |
| `--output-dir` | path | required | Directory for the two output files |

### Outputs

| File | Contents |
|------|----------|
| `co2_footprint_runs.csv` | One normalised row per measured run, with energy split by CPU, GPU and RAM, plus the hardware and grid-region metadata |
| `co2_footprint_report.md` | Human-readable report |

The Markdown report contains four sections: an overview with total runtime,
energy and emissions; totals broken down by stage; a table of individual runs;
and a summary of the hardware and grid regions the measurements came from.

### Which rows are used

The reporter is deliberately strict, so a stray CodeCarbon file from another
project cannot contaminate a SIEVE report. A row is skipped, with a warning
naming the file and line, when it:

- has a `project_name` that does not begin with `sieve-`
- is missing `run_id` or `timestamp`
- repeats a `run_id` already seen, which keeps a file that was copied twice
  from double-counting
- has a missing or unparseable duration, energy or emissions value

If no valid rows survive, the command reports an error and exits non-zero
rather than writing an empty report.

## Interpreting the numbers

These figures are useful, and they are not a full life-cycle assessment. Four
caveats travel with every report, and are repeated in the report itself:

- The values estimate **operational** CPU, GPU and RAM energy and the emissions
  implied by them. They exclude the manufacturing footprint of the hardware and
  the overhead of the facility housing it, so they understate total impact.
- Machine tracking mode can include unrelated activity on a shared host.
  Dedicated allocations give cleaner estimates.
- CPU and Apple Silicon measurements fall back to estimates when RAPL or
  `powermetrics` access is unavailable, which is common inside containers and
  on managed clusters. Where that happens, the CPU component is modelled rather
  than measured.
- Component energy values absent from a source row are recorded as zero in the
  normalised output, so a zero in the GPU column may mean "no GPU" or "not
  reported", not "no GPU energy used".

Emissions depend on the carbon intensity of the grid supplying the machine, so
the same computation run in two countries will report different emissions for
the same energy. When comparing runs, compare energy in kWh; when reporting
impact, report both.
