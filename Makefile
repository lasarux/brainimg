## brainimg — Makefile
##
## Convenience layer over the canonical commands in AGENTS.md.
## Run `make help` to see all targets. Most targets accept overrides, e.g.:
##   make decode MODEL=sd15-turbo FILE=outputs/mandril.brainimg OUT=mandril.png
##   make encode IMG=samples/real.jpg SAMPLE=puppy SEED=42
##   make test-one T=test_color.py::test_match_brightness_moves_toward_target

.PHONY: help setup lint test test-one encode decode grids grid compare sample paper paper-watch clean clean-paper check all

all: help

## help: Show this help (default target)
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make [target] [VAR=value ...]\n\nTargets:\n"} \
	/^## [a-zA-Z_-]+:.*/ { sub(/^## /,""); printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

## setup: Create the Python 3.12 venv and install dependencies
setup:
	uv venv -p 3.12
	@echo "Run: source .venv/bin/activate && uv pip install -r requirements.txt"

## lint: Run ruff (line-length 100, rules E/F/W/I)
lint:
	ruff check .

## test: Run the ML-free test suite (pytest, seconds)
test:
	python -m pytest

## test-one: Run a single test (T=<file>::<test> or T=<file>)
test-one:
	python -m pytest tests/$(T)

## encode: Encode an image -> outputs/<SAMPLE>.brainimg (IMG=, SAMPLE=, SEED=)
encode: outputs
	python src/encoder.py $(IMG) -o outputs/$(SAMPLE).brainimg $(if $(SEED),--seed $(SEED))

## decode: Decode a .brainimg -> outputs/<OUT> (FILE=, OUT=, DEVICE=, MODEL=, QUANTIZE=)
decode: outputs
	python src/decoder.py $(FILE) -o outputs/$(OUT) --device $(or $(DEVICE),cpu) \
		$(if $(MODEL),--model $(MODEL)) $(if $(QUANTIZE),--quantize)

## grids: Regenerate all 4 SIPI sample grids (outputs/ + docs/grids/)
grids:
	python scripts/run_all_grids.py

## grid: Build one sample's grid (SAMPLE=mandril|peppers|cameraman|airplane)
grid:
	python scripts/make_backend_grid.py $(SAMPLE) --size 512

## compare: Side-by-side original vs recon (SAMPLE=, OUT=)
compare:
	python scripts/make_comparison.py $(SAMPLE) $(OUT)

## sample: Generate a synthetic test image (scripts/make_sample.py)
sample:
	python scripts/make_sample.py

## paper: Compile docs/paper/PAPER.typ -> PAPER.pdf (typst)
paper:
	typst compile --root . docs/paper/PAPER.typ

## paper-watch: Recompile the paper on every save (live)
paper-watch:
	typst watch --root . docs/paper/PAPER.typ

## clean: Remove generated outputs and caches (keeps tracked grids + paper)
clean:
	rm -rf outputs/ .pytest_cache/ .ruff_cache/ .mypy_cache/

## clean-paper: Remove only the rendered PDF
clean-paper:
	rm -f docs/paper/PAPER.pdf

## check: Pre-push gate — lint + test
check: lint test

outputs:
	mkdir -p outputs