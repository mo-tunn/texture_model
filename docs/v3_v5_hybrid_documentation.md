# Texture Analysis: v3 ve v5 Hibrit Model Dokümantasyonu

Bu belge, cilt dokusu analizinde kullanılan **v3** ve **v5** yapay zeka modellerinin özelliklerini, farklı eğitim stratejilerini ve `local_tester.py` üzerinde iki modelin nasıl bir araya getirildiğini açıklar.

---

## 1. Modellerin Karakteristik Özellikleri ve Eğitim Analizi

### v3 Modeli (Balanced Training)
*   **Odak Noktası:** İyi yüzlerde (pürüzsüz ve normal ciltler) oldukça başarılı.
*   **Eğitim Verisi:** Sadece FFHQ veri seti (yüksek kaliteli genel yüz fotoğrafları) temel alınarak ve **Oversampling** yapılarak dengelenmiş (Balanced) bir yapı kurulmuştur.
*   **Karakteristiği:** Cildin genel pürüzsüzlüğünü iyi anlar ancak nispeten daha "pasif" tepki verir. Çok uçuk hatalı puanlar vermez, dar bir puan aralığında kalma eğilimindedir.
*   **Tipik Ham Skor Aralığı:** `~20 - 70` puan.

### v5 Modeli (Semi-Supervised Learning with Human Calibration)
*   **Odak Noktası:** Sorunlu ciltleri (akne, leke, derin skar vb.) tespit etmekte ve kusurları yakalamakta uzmandır.
*   **Eğitim Verisi:** FFHQ veri setinin yanı sıra **Acne** ve **Scar** veri setleri kullanılarak eğitilmiştir. Ayrıca, manuel labellama (Human Calibration) ve XGBoost kalibrasyon modeli ile yarı-denetimli (semi-supervised) bir yaklaşımla geliştirilmiştir.
*   **Karakteristiği:** Detaylara takılan, kusurları hemen cezalandıran ve puanı aşağı çeken "agresif" bir modeldir.
*   **Tipik Ham Skor Aralığı:** `~5 - 40` puan.

---

## 2. Normalizasyon Stratejisi

Modellerin ham (raw) skor çıktıları birbirinden çok farklı aralıklardadır. Bu nedenle modelleri doğrudan kıyaslamak veya ortalamalarını almak mümkün değildir. Hibrit sistemde ilk aşama **Normalizasyon** işlemidir:

*   **v3 Normalizasyonu (20-70 Aralığı):** 
    `v3_norm = (v3_raw - 20) / (70 - 20) * 100`
    *Not: v3'ün yüksek puanlarda aşırı zıplamasını (şişmesini) engellemek için aralık esnek tutulmuş (20-70), böylece artış daha yumuşak hale getirilmiştir.*
*   **v5 Normalizasyonu (5-40 Aralığı):** 
    `v5_norm = (v5_raw - 5) / (40 - 5) * 100`

---

## 3. Hibrit Karar Mantığı ve Ağırlıklandırma Kriterleri

İki modelin normalleştirilmiş skorları elde edildikten sonra sistem, modellerin zafiyet ve güçlerine göre hangi modele daha çok "güveneceğine" (ağırlık vereceğine) karar verir. 

Kullanılan **Eşik (Threshold)**: `50` puan. (50 üzeri iyiye, 50 altı kötüye işaret eder).
 `_hybrid_score` içerisindeki kurallar sırasıyla şöyledir:

### A. İstisna Kuralı (v5 Puan Şişmesini Engelleme)
*   **Kural:** Eğer `v5 raw < 28` ve `v3 raw >= 55` ise `v5_norm = min(v5_norm, 45.0)`
*   **Nedeni:** Sistem, v3 modelinin cildi çok pürüzsüz bulduğu (>=55.0) ama v5'in hala "Kötü" ya da "Ortalama" dediği (<28.0) durumlarda, v5 normalizasyonunun matematiksel olarak çok yüksek çıkıp 50 bandını geçmesini ve ağırlık dengesini bozmasını engeller.

### B. Ağırlıklandırma (Kime Güvenilecek?)

| Kural | Durum | Karar (Ağırlık) | Açıklama |
| :--- | :--- | :--- | :--- |
| **1. Kural** | `v3 norm >= 80` | **%90 v3** (`0.9`) - %10 v5 | v3 modeli cilde o kadar yüksek puan veriyor ki bu kişi kesinlikle çok çok pürüzsüz bir cilde sahip. v5'in agresifliğine takılmamak için neredeyse tamamen v3 sonucuna güvenilir. |
| **2. Kural** | `v3 norm >= 50` ve `v5 norm >= 50` | **%80 v3** (`0.8`) - %20 v5 | İki model de cildin iyi olduğu konusunda hemfikir, bu yüzden iyi ciltlerde uzman olan v3 modeline daha çok güvenilir. |
| **3. Kural** | `v3 norm < 50` ve `v5 norm < 50` | **%80 v5** (`0.8`) - %20 v3 | İki model de cildin kötü olduğu konusunda hemfikir, bu yüzden sorunlu ciltlerde uzman olan v5 modeline daha çok güvenilir. |
| **4. Kural** | Modeller Anlaşamıyor ve `v5 norm < 50` | **%80 v5** (`0.8`) - %20 v3 | v3 cilde fena değil dese bile, v5 modeli bir problem (leke, sivilce) tespit edip puanı düşürmüşse, çok yüksek ihtimalle gözle görülür bir cilt kusuru vardır. Bu sebeple v5 haklı kabul edilir. |
| **5. Kural** | Modeller Anlaşamıyor ve `v3 norm < 50` | **%50 v3** (`0.5`) - %50 v5 | v5 modeli bu kez "cilt temiz" derken, normalde toleranslı olan v3 bir sorun bulup puanını 50'nin altına indirmiş: Kararsız durum. Güvenli limanda kalmak için iki modelin %50-%50 ortalaması alınır. |

> [!TIP]
> Bu hibrit karar ağacı sayesinde, sadece FFHQ ile eğitilmiş bir model ile (v3) Scar/Acne dahil edilerek yarı-denetimli yapılandırılmış model (v5) tek bir vücut gibi hareket eder. Pürüzsüz yüzler için v3'ün toleransı şımartılırken, sorunlu yüzler için v5'in detaycılığı sistem otokontrolü görevi görür.
