## Render.com — Developer Notes

Docker-based deployment on Render.com.

### Prerequisites
- GitHub repo connected to Render
- Render account (free tier works)
- Data file handling (see below)

### Render Setup
1. **New Web Service** → Connect GitHub repo
2. **Settings:**
   - Name: `hybrid-recommender`
   - Region: `Frankfurt` (or closest)
   - Branch: `main`
   - Root Directory: `/` (root)
   - Runtime: `Docker`
   - Dockerfile Path: `Dockerfile`
   - Docker Context: `.`
   - Plan: `Starter` (2GB RAM, enough for demo data)

3. **Environment Variables:**
   - `PORT`: Auto-set by Render (don't set manually)
   - `PICKLE_PATH`: `data/prepare_data_demo.pkl` (default, optional)
   - `DATA_URL`: GitHub Release URL or external storage (if needed)
   - `LOG_LEVEL`: `INFO` or `ERROR` (default: `INFO`)
   - `PYTHONUNBUFFERED`: `1` (recommended)

4. **Advanced:**
   - Health Check Path: `/`
   - Auto-Deploy: `Yes` (on push to main)

### Data File (643MB)
Options:
1. **Git LFS** (recommended):
   ```bash
   git lfs track "data/prepare_data_demo.pkl"
   git add .gitattributes data/prepare_data_demo.pkl
   git commit -m "Add data via Git LFS"
   git push origin main
   ```
   Render will download via Git LFS.

2. **GitHub Release + Download:**
   - Upload `prepare_data_demo.pkl` to GitHub Releases
   - Set `DATA_URL` env var: `https://github.com/cihanoguz/hybrid-recommender-project/releases/download/v1.0.0/prepare_data_demo.pkl`
   - App auto-downloads on first run

3. **External Storage:**
   - S3, Google Drive, etc.
   - Set `DATA_URL` to public download URL

### Using render.yaml (Optional)
If using `render.yaml`:
- Render auto-detects it
- Settings in YAML override dashboard settings
- Useful for infrastructure-as-code

### Build & Deploy
- Render builds Docker image automatically
- Build time: ~5-10 minutes (NumPy/SciPy compilation)
- First deployment may take longer

### Troubleshooting
- **Build fails**: Check Dockerfile, ensure all files copied
- **Data not found**: Verify `DATA_URL` set correctly, check logs
- **Memory limit**: Upgrade to `Standard` plan (4GB RAM)
- **Port binding error**: Dockerfile uses `$PORT` from Render
- **Health check fails**: Increase `start-period` in Dockerfile

### Cost
- Free tier: 750 hours/month (enough for testing)
- Starter: $7/month (2GB RAM)
- Standard: $25/month (4GB RAM, for full dataset)

### Notes
- Dockerfile auto-detects Render's `PORT`
- Persistent disk not needed (data in image or downloaded)
- Auto-deploy on push to main branch
- Logs available in Render dashboard

