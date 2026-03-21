# Texture Analysis Pipeline - Hibrit Dokümantasyon (v3 & v5)

Bu dokümantasyon, cilt dokusu pürüzlülüğünü (texture) 0-100 arasında değerlendiren iki bağımsız modelin (**v3** ve **v5**) eğitim süreçlerini ve bu iki modelin birleştirilerek test ortamında (`local_tester.py`) oluşturduğu **Hibrit Karar Mekanizması**'nı açıklamaktadır.

---

## 1. Genel Mimari ve Amaç

Cilt dokusunu (sivilce, yara izi, geniş gözenekler, ince çizgiler) nesnel olarak puanlamak hedeflenmektedir. Bu amaçla tek bir her şeye yeten model yerine, farklı güçlü yanları olan iki model eğitilmiş ve hibrit bir sistem kurulmuştur:

1.  **v3 Modeli (İyi Cilt Uzmanı):** Sadece yüksek kaliteli genel yüz verileriyle (FFHQ) eğitilmiş, nispeten dar bir puan aralığında tahminde bulunan güvenli model. (Referans: `Texture_Analysis_Colab_v3.ipynb`)
2.  **v5 Modeli (Sorunlu Cilt Uzmanı):** FFHQ verilerine ek olarak binlerce spesifik sivilce (Acne) ve yara izi (Scar) verisi eklenerek eğitlen, kusurları hemen cezalandıran model. (Referans: `Texture_Analysis_Colab_v5.ipynb`)

Test aşamasında (inference) bu iki modelin skorları normalize edilerek belirli eşik ve ağırlıklandırma kurallarıyla birleştirilir.

---

## 2. v3 Modeli: Balanced Training (İyi Cilt Odaklı)

v3 modeli, yüksek çözünürlüklü ve genel yüz fotoğraflarından oluşan FFHQ veri setine odaklanır.

### Özellikler:
*   **Veri Seti:** ~10,000 FFHQ-256 görseli.
*   **Odak Noktası:** Cildin genel yapısını anlamak, pürüzsüz cilde "pürüzsüz" demek.
*   **Labellama Stratejisi:** İnsan müdahalesi olmadan 8 farklı matematiksel özellik (Fourier Entropy, GLCM, Ra, Rq vb.) üzerinden formül ile otomatik puanlama.
*   **Dengeleme (Balancing):** `Oversampling` yapılarak az bulunan skor aralıkları çoğaltılmış ve dengeli bir dağılım oluşturulmuştur.
*   **Model Özellikleri:** EfficientNetB0 Tabanlı, Huber Loss. Hedef MAE (Mean Absolute Error) `5 - 8`.
*   **Karakteristiği:** Uç puanlamalardan (çok iyi veya çok kötü) kaçınır, puanlama aralığı **20 - 70** arasında yoğunlaşır. Pasiftir.

---

## 3. v5 Modeli: Semi-Supervised ve İnsan Kalibrasyonu (Sorunlu Cilt Odaklı)

v5 modeli, v3'ün eksik kaldığı spesifik sorunlu ciltleri (şiddetli sivilce, akne, skar) doğru cezalandırabilmesi üzerine tasarlanmıştır. Yarı-denetimli bir mimaridiren (Semi-Supervised Learning).

### Özellikler:
*   **Veri Seti:** 
    *   `FFHQ` (Normal Yüzler) ~5,000 görsel
    *   `Acne` (Sivilceli Yüzler) ~1,000 görsel
    *   `Scar` (Yara İzli Yüzler) ~500 görsel
*   **Labellama Stratejisi (Human Calibration):** 
    1. Sistemden rastgele 300 görsel (her kategoriden 100) seçildi.
    2. İnsan tarafından manuel olarak 0-100 arası (0:Çok kötü, 100: Mükemmel) skorlandı.
    3. Manuel etiketler kullanılarak matematiksel 12 fütur ile eşleştirip bir **XGBoost Regressor** (Kalibrasyon Modeli) eğitildi.
*   **Label Propagation:** XGBoost modeli manuel puanlanan 300 görsel üzerinden öğrendiğini geriye kalan 6,000+ görsele uygulayarak tüm veri setini etiketledi.
*   **Ana Model Eğitimi:** XGBoost kararlarıyla labellanan 6000+ görsel, EfficientNetB0 yapısı ile tekrar eğitildi.
*   **Karakteristiği:** Sivilcelerde puanı anında 5-10 bandına çeken, oldukça **agresif** ve kusur avcısıdır. Puanlama aralığı ağırlıklı olarak **5 - 40** arasındadır.

---

## 4. Matematiksel Özellik Çıkarımı (Feature Extraction)

Eğitim setlerini labellarken (V3'te doğrudan, V5'te kalibrasyonu desteklemek için) fiziksel pürüzlülüğü ölçen sinyal işleme araçları kullanılmıştır:

1.  **Fourier Entropy & Spatial Entropy**: Piksellerin karmaşıklığı.
2.  **DWT Energy (Wavelet)**: İnce detaylardaki enerji seviyesi.
3.  **Ra, Rq (ISO 4287)**: Fiziksel yüzey pürüzlülük standartları.
4.  **GLCM (Contrast, Homogeneity vb.)**: Komşu pikseller arası renk geçişlerindeki sertlik.
5.  **MLC, Laplacian Variance, Gradient Magnitude**: Kenar/Köşe tespitleri. 

---

## 5. Inference: v3 ve v5 Hibrit Çıkarım (local_tester.py)

Son kullanıcı tarafında çalışan sistem, canlı kamerada ya da resim üzerinde, v3 ve v5 modellerini *birlikte* çalıştırır. Elde edilen iki farklı kararı matematiksel "Ağırlıklı Karar Mekanizması" ile uzlaştırarak nihai bir cilt skoru verir.

### Aşama 1: Normalizasyon

Model karakterleri sebebiyle üretilen raw (ham) skorlar aynı uzayda değildir. Bu nedenle belirlenen dağılım karakteristiklerine göre ikisi de 0-100 skalasına standardize edilir.

*   `v3_norm = (v3_raw - 20) / (70 - 20) * 100` (Yumuşak artış için aralık `20-70` olarak genişletilmiştir).
*   `v5_norm = (v5_raw - 5) / (40 - 5) * 100`

### Aşama 2: Ağırlıklı Karar Kuralları

Karar aşamasında (Hybrid Score Calculation) bir `Eşik (Threshold) = 50.0` kabul edilir. 50 üstü iyi, altı problemli cilt demektir. Algoritma iki skorun `norm` karşılığına bakar.

1.  **İstisna Koruması (v5 Şişkinliğini Önleme):** 
    Eğer `v5_raw < 28` (Hatalıdır, Kötüdür) **VE** `v3_raw >= 55` (Harikadır, Temizdir) ise, sistem çelişkiye düşer. Bu çelişkide v5'in norm puanının matematiksel tavan yapıp skoru suni saptırmasını önlemek için *`v5_norm = min(v5_norm, 45.0)`* sınırından geçememesi emredilir.
2.  **Mükemmeliyet (Kural 1 - %90 v3):** 
    Eğer `v3_norm >= 80` ise, cilt tartışmasız pürüzsüzdür. Sistem v5'in agresifliğine kulak asmaz, v3'ün kararına %90 ağırlık verir.
3.  **Ortak İyilik (Kural 2 - %80 v3):** 
    İki model de eşiği geçtiyse (`v3_norm >= 50` ve `v5_norm >= 50`), cilt güzeldir. İyi cilt uzmanı olan v3'e %80 güvenilir.
4.  **Ortak Kötülük (Kural 3 - %80 v5):** 
    İki model de eşiği geçemediyse (`v3_norm < 50` ve `v5_norm < 50`), cilt sorunludur. Sorunlu cilt uzmanı v5'e %80 güvenilir.
5.  **Karamsar Çelişki (Kural 4 - %80 v5):** 
    v3 cildi beğendi ama v5 beğenmediyse (`v5_norm < 50`). v5 agresif model olduğu için %80 ihtimalle bir noktada mikroskobik skar veya sivilce bulmuştur. Hastalığı atlamamak adına karamsar uzman v5'e %80 güvenilir.
6.  **İyimser Çelişki (Kural 5 - %50 Kararsızlık):** 
    v5 "çok iyi" diyor, ama normal şartlarda pasif olan v3 "hayır kötü" (`v3_norm < 50`) diyorsa, ortama kararsızlık hakim olur. Bu durumda iki modelin güvenilirliği yarı-yarıya `%50` alınarak ortalamaya gidilir. 

---

## 6. Sonuç ve Dağılımlar Çıktısı

| Skor Aralığı | UI Yorumu | Renk Kodu |
| :--- | :--- | :--- |
| **70 - 100** | Pürüzsüz cilt | Yeşil |
| **50 - 69** | Normal doku | Açık Yeşil |
| **35 - 49** | Orta düzey | Turuncu |
| **0 - 34** | Belirgin doku | Kırmızı |

Yukarıdaki akış sayesinde tek başına sadece normal yüzlerde çalışan v3 modeli ile sadece bozuk cildi tespit eden v5 modeli "Komite (Ensemble)" tarzı bir çalışma ile dengelenmiş olur.
