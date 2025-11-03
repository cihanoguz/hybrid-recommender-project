## Data Dosyasını Public URL'ye Yükleme

3 seçenek var (en kolaydan başlayarak):

---

## Yöntem 1: GitHub Releases (Önerilen)

### Adım 1: Release Oluştur
1. `https://github.com/cihanoguz/hybrid-recommender-project` → **Releases** → **Create a new release**
2. **Tag**: `v1.0.0`, **Title**: `v1.0.0 - Data Release`
3. `data/prepare_data_demo.pkl` dosyasını sürükle-bırak
4. **Publish release**

### Adım 2: URL
```
https://github.com/cihanoguz/hybrid-recommender-project/releases/download/v1.0.0/prepare_data_demo.pkl
```

### Adım 3: Render.com'a Ekle
- **Environment** → **Add Variable**
- Key: `DATA_URL`
- Value: Yukarıdaki URL'yi yapıştır
- **Save** → **Manual Deploy**

---

## Yöntem 2: Google Drive (Alternatif)

### Adım 1: Google Drive'a Yükle
1. Google Drive'a git → `prepare_data_demo.pkl` yükle
2. Dosyaya sağ tık → **Share** → **Anyone with the link** → **Viewer**
3. Link'i kopyala (şu formatta): `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`

### Adım 2: Direct Download URL'ine Çevir
Link şu formata dönüştürülmeli:
```
https://drive.google.com/uc?export=download&id=FILE_ID
```

**FILE_ID'yi nasıl bul:**
- Normal link: `https://drive.google.com/file/d/ABC123xyz/view`
- FILE_ID = `ABC123xyz`
- Direct download: `https://drive.google.com/uc?export=download&id=ABC123xyz`

### Adım 3: Render.com'a Ekle
- `DATA_URL` = direct download URL'i

**Not:** Büyük dosyalar için Google Drive link'i bazen redirect yapar, download script'i güncellememiz gerekebilir.

---

## Yöntem 3: Dropbox (Alternatif)

1. Dropbox'a yükle
2. **Share** → **Create link** → **Anyone with the link**
3. Link'in sonuna `?dl=1` ekle:
   - `https://www.dropbox.com/s/xxx/file.pkl?dl=1`
4. Render.com'a `DATA_URL` olarak ekle

---

## Öneri
**GitHub Releases** en güvenilir ve kolay. 643MB dosya için uygun (limit 2GB).

Alternatif olarak **Google Drive** veya **Dropbox** da kullanılabilir, ama büyük dosyalarda redirect sorunları olabilir.

