"""
Texture Analysis - Local Tester v6 (Hybrid)
v3 + v5 hibrit model ile yerel bilgisayarda test yapma

v3: İyi yüzlerde güçlü (FFHQ trained)
v5: Sorunlu ciltlerde güçlü (FFHQ + Acne + Scar trained)

Kullanım:
    python local_tester.py                    # Webcam ile test
    python local_tester.py image.jpg          # Tek görsel test
    python local_tester.py --folder ./images  # Klasördeki tüm görseller
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
import urllib.request
from glob import glob

# Face landmark indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]


class TextureTester:
    def __init__(self, model_path="texture_regressor.keras", hybrid_mode=True):
        self.model_v3 = None
        self.model_v5 = None
        self.model = None  # Fallback için
        self.detector = None
        self.model_path = model_path
        self.hybrid_mode = hybrid_mode
        
        # Model yükle
        self._load_model()
        
    def _load_model(self):
        """Model dosyalarını yükle (hibrit mod için v3 ve v5)"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # v3 model yolları
        v3_paths = [
            os.path.join(script_dir, "v3", "texture_regressor.keras"),
            os.path.join(script_dir, "v3", "best_model.keras"),
            "v3/texture_regressor.keras",
            "v3/best_model.keras",
        ]
        
        # v5 model yolları
        v5_paths = [
            os.path.join(script_dir, "v5", "texture_regressor.keras"),
            os.path.join(script_dir, "v5", "best_model.keras"),
            "v5/texture_regressor.keras",
            "v5/best_model.keras",
        ]
        
        # v4 fallback yolları
        v4_paths = [
            os.path.join(script_dir, "v4", "best_model.keras"),
            os.path.join(script_dir, "v4", "texture_regressor.keras"),
            "v4/best_model.keras",
        ]
        
        if self.hybrid_mode:
            # Hibrit mod: v3 ve v5'i yükle
            print("=== HİBRİT MOD ===")
            
            # v3 yükle
            for path in v3_paths:
                if os.path.exists(path):
                    print(f"v3 model yükleniyor: {path}")
                    try:
                        self.model_v3 = tf.keras.models.load_model(path, compile=False)
                        print("v3 model yüklendi!")
                    except Exception as e:
                        print(f"v3 yükleme hatası: {e}")
                        self.model_v3 = self._build_and_load_weights(path)
                    break
            
            # v5 yükle
            for path in v5_paths:
                if os.path.exists(path):
                    print(f"v5 model yükleniyor: {path}")
                    try:
                        self.model_v5 = tf.keras.models.load_model(path, compile=False)
                        print("v5 model yüklendi!")
                    except Exception as e:
                        print(f"v5 yükleme hatası: {e}")
                        self.model_v5 = self._build_and_load_weights(path)
                    break
            
            # En az bir model yüklenmeli
            if self.model_v3 is None and self.model_v5 is None:
                print("UYARI: Hibrit mod için model bulunamadı, tek model moduna geçiliyor...")
                self.hybrid_mode = False
            elif self.model_v3 is None:
                print("UYARI: v3 model bulunamadı, sadece v5 kullanılacak")
            elif self.model_v5 is None:
                print("UYARI: v5 model bulunamadı, sadece v3 kullanılacak")
            else:
                print("Her iki model de yüklendi!")
                return
        
        # Tek model modu (fallback)
        if not self.hybrid_mode or (self.model_v3 is None and self.model_v5 is None):
            all_paths = v5_paths + v4_paths + v3_paths + [self.model_path, "texture_regressor.keras"]
            
            for path in all_paths:
                if os.path.exists(path):
                    print(f"Model yükleniyor: {path}")
                    try:
                        self.model = tf.keras.models.load_model(path, compile=False)
                        print("Model yüklendi!")
                        return
                    except Exception as e:
                        print(f"Yükleme hatası: {e}")
                        self.model = self._build_and_load_weights(path)
                        if self.model:
                            return
            
            print("HATA: Hiçbir model bulunamadı!")
            sys.exit(1)
    
    def _build_and_load_weights(self, keras_path):
        """Model oluştur ve weights'leri .keras dosyasından yükle"""
        try:
            import zipfile
            import tempfile
            
            model = self._build_model()
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(keras_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                weights_path = os.path.join(tmp_dir, "model.weights.h5")
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print("Weights yüklendi!")
                    return model
        except Exception as e:
            print(f"Weights yükleme hatası: {e}")
        return None
    
    def _build_model(self):
        """Model mimarisini oluştur"""
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        for layer in base.layers[:-30]:
            layer.trainable = False
        
        model = tf.keras.Sequential([
            base,
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _init_detector(self):
        """MediaPipe Face Landmarker'ı başlat"""
        if self.detector is not None:
            return
            
        model_path = 'face_landmarker.task'
        if not os.path.exists(model_path):
            print("MediaPipe model indiriliyor...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("MediaPipe model indirildi!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    def _get_coords(self, landmarks, indices, w, h):
        """Landmark koordinatlarını al"""
        return np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
    
    def create_texture_map(self, image):
        """Görüntüden texture map oluştur"""
        self._init_detector()
        
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return None, None, "Görüntü okunamadı"
        else:
            img = image.copy()
        
        # Resize
        img = cv2.resize(img, (512, 512))
        h, w = 512, 512
        
        # Face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None, None, "Yüz tespit edilemedi"
        
        landmarks = result.face_landmarks[0]
        
        # Mask oluştur
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self._get_coords(landmarks, FACE_OVAL, w, h)], 255)
        
        for region in [LIPS, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW]:
            cv2.fillPoly(mask, [self._get_coords(landmarks, region, w, h)], 0)
        
        # Sakal/tüy eliminasyonu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        hair_zones = cv2.GaussianBlur(edges, (25, 25), 0)
        _, beard_mask = cv2.threshold(hair_zones, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(beard_mask))
        
        if np.sum(mask == 255) < 20000:
            return None, None, "Yetersiz cilt alanı"
        
        # Texture map oluştur
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = cv2.split(lab)[0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        blurred = cv2.GaussianBlur(enhanced_l, (37, 37), 0)
        texture_map = cv2.add(cv2.subtract(enhanced_l, blurred), 128)
        
        ai_ready = np.where(mask == 255, texture_map, 128).astype(np.uint8)
        
        return ai_ready, img, None
    
    def predict(self, texture_map):
        """Texture map'ten skor tahmin et (hibrit mod destekli)"""
        # Preprocessing (eğitimle aynı)
        img = np.stack([texture_map, texture_map, texture_map], axis=-1)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        
        if self.hybrid_mode and self.model_v3 is not None and self.model_v5 is not None:
            # Hibrit tahmin
            raw_v3 = float(self.model_v3.predict(img, verbose=0)[0][0])
            raw_v5 = float(self.model_v5.predict(img, verbose=0)[0][0])
            
            raw_v3 = np.clip(raw_v3, 0, 100)
            raw_v5 = np.clip(raw_v5, 0, 100)
            
            # Hibrit skor hesaplama (normalize edilmiş)
            score, norm_v3, norm_v5 = self._hybrid_score(raw_v3, raw_v5)
            
            # Return: hibrit, normalize v3, normalize v5, raw v3, raw v5
            return score, norm_v3, norm_v5, raw_v3, raw_v5
        
        elif self.model_v3 is not None:
            pred = self.model_v3.predict(img, verbose=0)[0][0]
        elif self.model_v5 is not None:
            pred = self.model_v5.predict(img, verbose=0)[0][0]
        else:
            pred = self.model.predict(img, verbose=0)[0][0]
        
        score = float(np.clip(pred, 0, 100))
        return score
    
    def _normalize_score(self, score, model_type):
        """
        Model çıktısını 0-100 aralığına normalize et
        
        Her modelin tipik çıktı aralığı farklı:
        - v3: 25-55 arası (pasif, dar aralık)
        - v5: 5-40 arası (agresif, düşük skorlar)
        """
        if model_type == 'v3':
            # v3: 20-70 → 0-100 (Daha yumuşak bir artış için genişletildi)
            min_val, max_val = 20, 70
        elif model_type == 'v5':
            # v5: 5-40 → 0-100
            min_val, max_val = 5, 40
        else:
            return score
        
        # Linear normalization
        normalized = (score - min_val) / (max_val - min_val) * 100
        return float(np.clip(normalized, 0, 100))
    
    def _hybrid_score(self, score_v3, score_v5):
        """
        Hibrit skor hesaplama (normalize edilmiş skorlarla)
        
        Strateji / Kurallar:
        - İkisi de yüksekse → iyi cilt (v3'e güven)
        - İkisi de düşükse → kötü cilt (v5'e güven)
        - Farklıysa ve v5 düşük diyorsa → kötü cilt (v5'e güven)
        - Diğer farklı senaryolarda → ortasını al
        """
        # Önce normalize et
        norm_v3 = self._normalize_score(score_v3, 'v3')
        norm_v5 = self._normalize_score(score_v5, 'v5')
        
        # İstisna kural: v5 raw 28 altındaysa ve v3 raw 55 üzerindeyse v5'in norm değeri çok yükselmesin.
        if score_v5 < 28 and score_v3 >= 55:
            norm_v5 = min(norm_v5, 45.0)  # v5'in norm puanını maksimum 45'te tut
            
        # Eşik değeri (50 puan sınır olarak kabul edilebilir)
        threshold = 50.0
        
        # Ağırlık belirleme
        if norm_v3 >= 80:
            # v3 çok yüksek skor (80+) verdiyse, yüksek ihtimalle iyi bir cilttir, v3'e ağırlık ver
            weight_v3 = 0.9  # v3'ün ağırlığı %90
        elif norm_v3 >= threshold and norm_v5 >= threshold:
            # İkisi de yüksek → iyi cilt (v3'e güven)
            weight_v3 = 0.8  # v3'ün ağırlığı %80, v5 %20
        elif norm_v3 < threshold and norm_v5 < threshold:
            # İkisi de düşük → kötü cilt (v5'e güven)
            weight_v3 = 0.2  # v3'ün ağırlığı %20, v5 %80
        else:
            # Farklıysa (Biri yüksek biri düşük)
            if norm_v5 < threshold:
                # v5 düşük kararını vermiş (ama v3 yüksek demiş ve 80'i de geçememiş)
                weight_v3 = 0.2  # v5'e %80 oranında güven
            else:
                # v3 düşük, v5 yüksek diyor
                weight_v3 = 0.5
        
        weight_v5 = 1.0 - weight_v3
        hybrid = (norm_v3 * weight_v3) + (norm_v5 * weight_v5)
        
        return float(np.clip(hybrid, 0, 100)), norm_v3, norm_v5
    
    def get_interpretation(self, score):
        """Skor yorumu"""
        if score >= 70:
            return "Smooth skin", (0, 255, 0)  # Yeşil
        elif score >= 50:
            return "Normal texture", (0, 200, 100)  # Açık yeşil
        elif score >= 35:
            return "Moderate texture", (0, 165, 255)  # Turuncu
        else:
            return "Prominent texture", (0, 0, 255)  # Kırmızı
    
    def test_image(self, image_path, show=True):
        """Tek görüntü test et"""
        print(f"\nTest ediliyor: {image_path}")
        
        texture_map, original, error = self.create_texture_map(image_path)
        
        if error:
            print(f"HATA: {error}")
            return None
        
        result = self.predict(texture_map)
        
        # Hibrit mod kontrolü
        if isinstance(result, tuple) and len(result) == 5:
            score, norm_v3, norm_v5, raw_v3, raw_v5 = result
            interpretation, color = self.get_interpretation(score)
            
            print(f"{'='*55}")
            print(f"  v3 raw: {raw_v3:.1f} → normalized: {norm_v3:.1f}/100")
            print(f"  v5 raw: {raw_v5:.1f} → normalized: {norm_v5:.1f}/100")
            print(f"{'='*55}")
            print(f"  HİBRİT SKOR:  {score:.1f}/100")
            print(f"  Yorum: {interpretation}")
            print(f"{'='*55}")
        elif isinstance(result, tuple):
            score = result[0]
            interpretation, color = self.get_interpretation(score)
            print(f"{'='*40}")
            print(f"TEXTURE SCORE: {score:.1f}/100")
            print(f"Yorum: {interpretation}")
            print(f"{'='*40}")
        else:
            score = result
            interpretation, color = self.get_interpretation(score)
            
            print(f"{'='*40}")
            print(f"TEXTURE SCORE: {score:.1f}/100")
            print(f"Yorum: {interpretation}")
            print(f"{'='*40}")
        
        if show:
            # Doku varyasyonlarını (128'den olan absdiff sapmaları) hesapla
            deviation = cv2.absdiff(texture_map, 128)
            
            # Doku kusurlarını x8 kat büyüterek çooook daha belirgin hale getiriyoruz
            highlight = cv2.convertScaleAbs(deviation, alpha=8.0)
            
            # "Tüm dokuyu sarması" için doku alanının tamamına hafif kırmızı bir zemin ekliyoruz
            # Bu sayede sadece kusurlu yerler değil, modelin baktığı tüm alan belirginleşiyor
            mask_area = (deviation > 0).astype(np.uint8)
            mask_area = cv2.dilate(mask_area, np.ones((5, 5), np.uint8), iterations=3)
            base_red = mask_area * 40  # Hafif kırmızı zemin tonu
            
            # Önce zemin kırmızısı ile kusurların kırmızısını birleştirelim
            total_red_add = cv2.add(highlight, base_red)
            
            # Analiz edilen görselin üstüne yapıştırmak / kopyalamak için Overlay oluştur
            overlaid_image = original.copy()
            
            # Kırmızı kanalını (2) güçlüce artırıyoruz
            overlaid_image[:, :, 2] = cv2.add(overlaid_image[:, :, 2], total_red_add)
            
            # Kırmızılığın tam kan kırmızısı gibi belli olması için diğer renkleri (Mavi ve Yeşil) ciddi oranda kısıyoruz
            reduce_val = cv2.convertScaleAbs(highlight, alpha=0.8)
            overlaid_image[:, :, 1] = cv2.subtract(overlaid_image[:, :, 1], reduce_val)
            overlaid_image[:, :, 0] = cv2.subtract(overlaid_image[:, :, 0], reduce_val)

            # Görselleştirme (Gözle test etmeyi kolaylaştırmak için boyut 400x400'e çıkarıldı)
            display = np.hstack([
                cv2.resize(original, (400, 400)),
                cv2.resize(overlaid_image, (400, 400))
            ])
            
            # Skor yazısı
            cv2.putText(display, f"Score: {score:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, interpretation, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if isinstance(result, tuple) and len(result) >= 3:
                norm_v3_disp = result[1] if len(result) >= 2 else 0
                norm_v5_disp = result[2] if len(result) >= 3 else 0
                cv2.putText(display, f"v3:{norm_v3_disp:.0f} v5:{norm_v5_disp:.0f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Texture Analysis", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return score
    
    def test_folder(self, folder_path):
        """Klasördeki tüm görselleri test et"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        images = []
        for ext in extensions:
            images.extend(glob(os.path.join(folder_path, ext)))
        
        if not images:
            print(f"HATA: {folder_path} klasöründe görsel bulunamadı")
            return
        
        print(f"\n{len(images)} görsel bulundu\n")
        
        results = []
        for img_path in images:
            texture_map, _, error = self.create_texture_map(img_path)
            if error:
                continue
            
            result = self.predict(texture_map)
            if isinstance(result, tuple) and len(result) == 5:
                score, norm_v3, norm_v5, raw_v3, raw_v5 = result
                results.append((os.path.basename(img_path), score, norm_v3, norm_v5))
            elif isinstance(result, tuple):
                results.append((os.path.basename(img_path), result[0], None, None))
            else:
                results.append((os.path.basename(img_path), result, None, None))
        
        # Sonuçları göster
        print("\n" + "="*75)
        print("SONUÇLAR (Hibrit Mod - Normalize Edilmiş)" if self.hybrid_mode else "SONUÇLAR")
        print("="*75)
        
        if self.hybrid_mode and results and results[0][2] is not None:
            print(f"{'Dosya':<22} {'v3(norm)':>10} {'v5(norm)':>10} {'Hibrit':>10} {'Yorum':>20}")
            print("-"*75)
            
            for item in sorted(results, key=lambda x: x[1], reverse=True):
                filename, score, norm_v3, norm_v5 = item
                interpretation, _ = self.get_interpretation(score)
                print(f"{filename:<22} {norm_v3:>10.1f} {norm_v5:>10.1f} {score:>10.1f} {interpretation:>20}")
        else:
            print(f"{'Dosya':<30} {'Skor':>10} {'Yorum':>15}")
            print("-"*55)
            
            for item in sorted(results, key=lambda x: x[1], reverse=True):
                filename, score = item[0], item[1]
                interpretation, _ = self.get_interpretation(score)
                print(f"{filename:<30} {score:>10.1f} {interpretation:>15}")
        
        if results:
            scores = [r[1] for r in results]
            print("-"*70)
            print(f"Ortalama: {np.mean(scores):.1f}")
            print(f"Min: {np.min(scores):.1f}, Max: {np.max(scores):.1f}")
    
    def test_webcam(self):
        """Webcam ile canlı test"""
        print("\nWebcam başlatılıyor...")
        print("Çıkmak için 'q' tuşuna basın")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("HATA: Webcam açılamadı")
            return
        
        frame_count = 0
        last_score = None
        last_score_v3 = None
        last_score_v5 = None
        last_interpretation = None
        last_color = (255, 255, 255)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display = frame.copy()
            
            # Her 15 frame'de bir analiz yap (performans için)
            if frame_count % 15 == 0:
                texture_map, _, error = self.create_texture_map(frame)
                
                if error is None:
                    result = self.predict(texture_map)
                    if isinstance(result, tuple) and len(result) == 5:
                        last_score, last_score_v3, last_score_v5, _, _ = result
                    elif isinstance(result, tuple):
                        last_score = result[0]
                        last_score_v3 = None
                        last_score_v5 = None
                    else:
                        last_score = result
                        last_score_v3 = None
                        last_score_v5 = None
                    last_interpretation, last_color = self.get_interpretation(last_score)
                else:
                    last_score = None
                    last_interpretation = error
                    last_color = (128, 128, 128)
            
            # Sonucu göster
            if last_score is not None:
                cv2.putText(display, f"Score: {last_score:.1f}/100", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2)
                cv2.putText(display, last_interpretation, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
                
                if last_score_v3 is not None and last_score_v5 is not None:
                    cv2.putText(display, f"v3:{last_score_v3:.0f} v5:{last_score_v5:.0f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            elif last_interpretation:
                cv2.putText(display, last_interpretation, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_color, 2)
            
            cv2.imshow("Texture Analysis - Press 'q' to quit", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    tester = TextureTester()
    
    if len(sys.argv) == 1:
        # Argüman yok - webcam
        tester.test_webcam()
    
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        
        if arg in ['-h', '--help']:
            print(__doc__)
        elif os.path.isdir(arg):
            tester.test_folder(arg)
        elif os.path.isfile(arg):
            tester.test_image(arg)
        else:
            print(f"HATA: '{arg}' bulunamadı")
    
    elif len(sys.argv) == 3 and sys.argv[1] == '--folder':
        tester.test_folder(sys.argv[2])
    
    else:
        print("Kullanım:")
        print("  python local_tester.py                    # Webcam")
        print("  python local_tester.py image.jpg          # Tek görsel")
        print("  python local_tester.py ./images           # Klasör")
        print("  python local_tester.py --folder ./images  # Klasör")


if __name__ == "__main__":
    main()
