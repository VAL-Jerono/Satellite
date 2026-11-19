# ──────────────────────────────────────────────────────────────────────────────
# Kenya Land Readiness Atlas — Local Dev Makefile
# Usage:
#   make api          → start FastAPI on :8000 (foreground)
#   make frontend     → start Streamlit on :8501 (foreground)
#   make drift        → run seasonal drift simulation report
#   make check        → verify models + county files are present
#   make libomp       → re-create libomp symlink (run if LightGBM crashes on Mac restart)
#   make county-data  → regenerate county JSONs + reference parquet (~15s)
#   make git-status   → show git status
# ──────────────────────────────────────────────────────────────────────────────

PYTHON := /Library/Developer/CommandLineTools/usr/bin/python3
STREAMLIT := /Users/leonida/Library/Python/3.9/bin/streamlit
UVICORN := /Users/leonida/Library/Python/3.9/bin/uvicorn
MPLCONFIGDIR := /tmp/mpl

.PHONY: api frontend drift check libomp county-data git-status

api:
	@echo "──────────────────────────────────────────"
	@echo "  🚀 Starting FastAPI on http://localhost:8000"
	@echo "  API docs: http://localhost:8000/docs"
	@echo "  Ctrl+C to stop"
	@echo "──────────────────────────────────────────"
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(UVICORN) api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--reload-dir api

frontend:
	@echo "──────────────────────────────────────────"
	@echo "  🗺️  Starting Streamlit on http://localhost:8501"
	@echo "  Ctrl+C to stop"
	@echo "──────────────────────────────────────────"
	API_URL=http://localhost:8000 \
	MPLCONFIGDIR=$(MPLCONFIGDIR) \
	$(STREAMLIT) run frontend/app.py \
		--server.port 8501 \
		--server.headless true

drift:
	@echo "── Running seasonal drift simulation ──────"
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) -m monitoring.drift --simulate
	@echo "── Report written to monitoring/reports/ ──"

check:
	@echo "── Checking models ───────────────────────"
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/verify_models.py
	@echo "── Checking county files ─────────────────"
	@ls -1 data/county_summaries/*.json 2>/dev/null | wc -l | xargs -I{} echo "{} county JSON files present"
	@ls data/splits/reference_features.parquet 2>/dev/null && echo "reference_features.parquet OK" || echo "MISSING: reference_features.parquet"
	@ls data/atlas/zone_confidence.json 2>/dev/null && echo "zone_confidence.json OK" || echo "MISSING: zone_confidence.json"

libomp:
	@echo "── Re-linking libomp for LightGBM on Mac ─"
	sudo ln -sf /Users/leonida/Library/Python/3.9/lib/python/site-packages/torch/lib/libiomp5.dylib \
		/usr/local/lib/libomp.dylib
	@echo "── Done. LightGBM should now load OK ─────"

county-data:
	@echo "── Building county summaries + reference parquet (~15s) ──"
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) scripts/build_county_summaries.py

git-status:
	git log --oneline -5
	git status --short
