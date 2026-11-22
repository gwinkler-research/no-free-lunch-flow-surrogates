# Makefile of the reproduction package.
#
#   make check     rebuild both result matrices from the per-seed result files and
#                  re-emit the value macros; all three must equal the committed files
#                  byte for byte (CPU, seconds)
#
# Training and inference need the raw simulation data and a GPU; see README.md.

PY ?= python
CHECK := build/check

export PYTHONPATH := $(CURDIR)
export CUDA_VISIBLE_DEVICES :=

.PHONY: help check

help:
	@echo "Targets: check (see README.md)"

check:
	@mkdir -p $(CHECK)
	@$(PY) scripts/build_paper_matrix.py --only cmp --out-dir $(CHECK) > $(CHECK)/cmp_matrix.log
	@$(PY) scripts/build_paper_matrix.py --only kvs --out-dir $(CHECK) > $(CHECK)/kvs_matrix.log
	@$(PY) values/emit_values.py --out $(CHECK)/paper_values.tex > $(CHECK)/emit_values.log
	@ok=1; \
	for f in reports/PAPER_MATRIX_FIX/cmp_matrix.csv reports/PAPER_MATRIX_FIX/cmp_matrix.json \
	         reports/PAPER_MATRIX_CLEAN/kvs_matrix.csv reports/PAPER_MATRIX_CLEAN/kvs_matrix.json \
	         values/paper_values.tex; do \
	  if cmp -s $(CHECK)/$$(basename $$f) $$f; then echo "[ok]   $$f"; \
	  else echo "[DIFF] $$f"; ok=0; fi; \
	done; \
	if [ $$ok = 1 ]; then echo "check passed: every printed number is reproduced"; \
	else echo "check FAILED"; exit 1; fi
