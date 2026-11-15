TEAM ?= NEXT
PY ?= python

.PHONY: test fake submit

# Convenience only (make is not on Windows by default). The commands themselves are in README.md.

# Fake data only -- safe for anyone, including Claude.
test:
	$(PY) -m pytest -q

# A synthetic match in the real file formats, for debugging without the real data.
# test_data/ is git-ignored.
fake:
	$(PY) tools/make_fake_match.py --out test_data/fake_match --seed 0

# REAL DATA: run by a human, never by Claude. Regenerates the submission byte-for-byte.
submit:
	$(PY) -m src.pipeline --team-name $(TEAM) --no-cache
	$(PY) evaluation/validate_submission.py evaluation/results/$(TEAM).csv
	$(PY) tools/sanity_check.py evaluation/results/$(TEAM).csv
