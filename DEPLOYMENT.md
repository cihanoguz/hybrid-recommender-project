## Deployment Notes

### Render.com

**Setup:**
- Runtime: Docker
- Build: Auto-detects Dockerfile
- Port: Auto (uses `$PORT` from Render)

**Data file (643MB):**
- Downloaded during Docker build (baked into image)
- No runtime download needed
- Default URL: GitHub Release v1.0.0 (hardcoded in Dockerfile)
- To change: Update `ARG DATA_URL` in Dockerfile

**Environment variables (optional):**
- `PICKLE_PATH`: `data/prepare_data_demo.pkl` (default)
- `LOG_LEVEL`: `INFO` or `ERROR`
- `PYTHONUNBUFFERED`: `1`

**Performance:**
- Build time: ~5-10 min (includes 643MB download)
- Startup: Fast (data already in image)
- RAM: Starter plan (2GB) sufficient for demo data

**Notes:**
- Free tier: 750 hrs/month
- Auto-deploy on push to main
- Logs available in dashboard

