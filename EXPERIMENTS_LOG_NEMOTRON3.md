# سجل التجارب والنتائج الشامل — المرحلة الأولى: `NVIDIA Nemotron-3 Diarization`

**المشروع:** From Speech to Investment Intelligence: A Unified Framework for Business Dialogue Understanding and Analysis (Shark Tank Egypt)  
**الفريق:** محمد بهاء (Mohamed Bahaa) & شهد ربيع (Shahd Rabie)  
**تاريخ التوثيق:** 2026-11-23  

---

## 1. الروابط والملفات المحفوظة (Backups & Cloud Links)

كل الأوزان المدربة، الجداول، الداتا، والأكواد تم تأمينها ورفعها على Hugging Face بالإضافة للملفات المحلية:

1. **مستودع الموديل المنشور (Model Repository):**
   * الرابط: `https://huggingface.co/Mbahaa08/SharkTank-ArabicEnglish-Diarizer`
   * يحتوي على:
     * `nemotron3_arzen_adapter.pt` (أوزان طبقة `TemporalResidualAdapter` المدربة + معاملات المعايرة `cfg_star`, `lam_star`).
     * `pipeline_code.py` / `sharktank_diarizer.py` (الكود الكامل للتشغيل).
     * `final_13files_comparison_summary.csv` و `ablation_summary_table.csv`.
2. **مستودع البيانات والنتائج (Private Dataset Repository):**
   * الرابط: `https://huggingface.co/datasets/Mbahaa08/sharktank-diarization-data`
   * المجلدات:
     * `arzen/recordings/`: الـ 38 ملف صوت كاملين (`16kHz Mono WAV`).
     * `arzen/test/` و `arzen/dev/` و `arzen/train/`: ملفات الـ Ground Truth (`.rttm`) والـ `manifest.json`.
     * `results_and_weights/`: نسخة كاملة من كل نتائج التجارب وملفات `.rttm` المتوقعة للـ 13 ملف وأوزان الـ Adapter.
3. **الأدوات المحلية على اللابتوب (`D:\Grad\sharktank-diarization`):**
   * `prepare_arzen_rttm.py`: سكربت تحويل بيانات `ArzEn` إلى صيغة `.rttm` القياسية.
   * `nemotron3_baseline_eval.ipynb`: نوتبوك التقييم الأساسي.
   * `rttm_studio.html`: أداة الويب التفاعلية لمراجعة وتعديل ملفات الـ `.rttm` مع الصوت أو الفيديو وعزل المتحدثين.
   * الفيديوهات المقصوصة للمراجعة:
     * `D:\Grad\videos_sharks\videos_sharks\Output\Videos\video_10_part_03.mp4`
     * `D:\Grad\videos_sharks\videos_sharks\Output\Videos\video_13_part_03.mp4`

---

## 2. ملخص الورقة البحثية المختبرة (`arXiv:2611.12246`)

* **عنوان الورقة:** *Nemotron 3 Diarization: Advancing the Accuracy–Efficiency Frontier in Streaming Speaker Diarization* (NVIDIA & BUT-FIT, 9 Nov 2026).
* **الموديل:** `nvidia/Nemotron-3-Diarization` (~99.2M parameters).
* **المعمارية:**
  * 31-layer Pure Transformer Encoder مع Rotary Positional Embeddings (RoPE) مبدوء من `NEST` (المتدرب على `Granary-v2` لـ 25 لغة أوروبية).
  * خطوة الـ Encoder هي `80ms` وفوقها طبقة **1D Subpixel Convolution Upsampler (8×)** لإخراج توقعات كل **`10ms`**.
  * يدعم حتى **8 متحدثين** (`S = 8` قنوات ثابتة) مرتبين حسب وقت أول ظهور (**Arrival-Time-Sort `ATS`**) مع ذاكرة **`AOSC` (Arrival-Order Speaker Cache = 264 frames)**.
  * مدرب بـ 4 دوال خسارة: `PIL (0.75)` + `ATS (0.25)` + `Activity Loss (0.5)` + `Phantom Loss (0.1)`.
* **فجوة التدريب المكتشفة من الورقة:**
  1. الموديل متدرب على `VoxConverse v0.3 (dev & test)`, `AMI`, `DIHARD III dev`, `AliMeeting`, `AISHELL-4`, `NOTSOFAR-1`, `DISPLACE`، لكنه **لم يرَ أي بيانات عربية أو عربي-إنجليزي (Code-Switching)** في التدريب (`ArzEn` هي 100% Unseen).
  2. أقصى طول مقطع اتدربت عليه ذاكرة `AOSC` في المرحلة الثانية من التدريب هو **`105 ثانية` فقط** (`Section 2.4`).

---

## 3. تفاصيل الـ 7 تجارب على العربي المصري-الإنجليزي (`ArzEn` Ablation Study)

تم تصميم 7 تجارب متسلسلة لمعالجة مشاكل الموديل على الكلام العربي المصري المختلط بالإنجليزي (`ArzEn`):

### تشخيص مشكلة الـ Baseline (`Exp 0`) على `ArzEn`:
* عند تشغيل الموديل الخام على `ArzEn`، وجدنا أن **`Confusion_% = 0.64%` فقط** (الموديل يميز أصوات المتحدثين بدقة 99.36% عندما يسمعهم)، لكن:
  1. **`Missed Speech = 19.06%`** (يمثل **87% من إجمالي الخطأ الكلي** `DER = 21.84%`)؛ بسبب أن الموديل متدرب بـ Forced Alignment فيقطع الجملة عند أي وقفة نَفَس صغيرة، وبسبب العتبة الصارمة (`0.50`) التي تُسقط أواخر الكلمات المصرية الهادئة.
  2. **اختراع متحدث رابع وهمي (`pred_speakers = 3.46` مقابل `2.92` حقيقي)** في 7 من أصل 13 ملف؛ بسبب تغير طبقة الصوت لحظياً عند التحول من العربي للإنجليزي (**Bilingual Code-Switching Pitch Shift**).

---

### وصف التجارب من `Exp 0` إلى `Exp 6`:

1. **`Exp 0: Baseline (Raw Nemotron-3)`**
   * تشغيل الموديل الخام بالعتبة الافتراضية (`0.50`) بدون أي معالجة مسبقة أو لاحقة.
2. **`Exp 1: + Audio Preprocessing (قبل دخول الموديل)`**
   * إزالة الـ DC offset.
   * فلتر ترددات الكلام البشري **`Zero-Phase 4th-Order Butterworth Band-Pass Filter (80 Hz – 7600 Hz)`** باستخدام `sosfiltfilt` (بإزاحة زمنية صفرية `0ms phase shift`).
   * تقوية الحروف الهادئة **`Noise-Gated Envelope AGC`** (نافذة `50ms`، ترفع مستوى الكلام الواطي فوق بوابة الضوضاء `2× noise_floor` حتى `2.5×–3.0×` بدون تكبير الصمت) متبوعة بضغط ديناميكي ناعم **`Soft-Knee tanh Compression`**.
   * **الأثر:** خفّض الـ `DER_0.0s` بمقدار **`-2.10%`** بمفرده.
3. **`Exp 2: + Math Decision Layer Only (بعد الموديل على الصوت الخام)`**
   * استبدال عتبة `0.50` الجامدة بـ 3 معادلات رياضية على مصفوفة احتمالات الـ `10ms`:
     * **أ) `Hysteresis Dual-Thresholding & Dilation`:** عتبة بداية $\theta_{\text{on}} = 0.40$ وعتبة استمرار $\theta_{\text{off}} = 0.18$ مع توسيع الحواف ($\delta_{\text{on}} = 0.10\text{s}, \delta_{\text{off}} = 0.15\text{s}$).
     * **ب) `1D Temporal Morphological Closing`:** سد فجوات النَفَس القصيرة داخل الجملة لنفس المتحدث ($\Delta_{\text{gap}} = 0.85\text{s}$).
     * **ج) `Phantom-Channel Energy Projection`:** إذا كان إجمالي طاقة قناة متحدث أقل من $E_{\min} = 15.0\text{s}$ في المقابلة، يتم إسقاطها وإعادة نسب مقاطعها للمتحدث الرئيسي الأعلى احتمالاً في نفس النافذة الزمنية (قضى تماماً على المتحدث الرابع الوهمي الناتج عن الـ Code-Switching!).
   * **الأثر:** خفّض الـ `DER_0.0s` من `24.86%` إلى **`10.26%`** والـ `JER` من `20.98%` إلى **`9.94%`**.
4. **`Exp 3: Combined (Audio Preprocessing + Math Layer)`**
   * دمج `Exp 1` (قبل الموديل) مع `Exp 2` (بعد الموديل).
   * **الأثر:** كسر حاجز الـ 10% لأول مرة (**`DER_0.0s = 9.98%`** و **`DER_0.25s = 8.37%`**).
5. **`Exp 4: Auto-Calibrated on DEV Split (Zero Data Leakage)`**
   * بدلاً من اختيار أرقام المعادلة يدوياً، تم استخدام خوارزمية **Coordinate Descent Optimization** على 3 ملفات من قسم **`ArzEn DEV` (Validation Set)** المنفصل تماماً عن الـ `TEST`.
   * المعاملات المتعلمة أوتوماتيكياً من الـ `DEV`:
     `theta_on = 0.45`, `theta_off = 0.25`, `pad_onset = 0.16`, `pad_offset = 0.24`, `max_breath_gap = 1.30`.
   * **الأثر عند اختبارها بشكل أعمى على الـ `TEST`:** نزل الـ `DER_0.0s` إلى **`9.21%`** والـ `DER_0.25s` إلى **`7.57%`**.
6. **`Exp 5: Frozen-Backbone Fine-Tuning (`TemporalResidualAdapter`)`**
   * تجميد الـ 31 طبقة الداخلية لـ `Nemotron-3` بالكامل (`99,226,504` بارامتر — `100% Frozen`) للحفاظ على دقة تمييز الأصوات.
   * بناء وتدريب طبقة عصبية خفيفة **`TemporalResidualAdapter` (`6,098` بارامتر فقط)** تستخدم **Multi-Scale Dilated 1D Convolutions** (`dilation = 1, 5, 25` لتغطية سياق زمني حتى `1.0s`) فوق مخرجات الـ `10ms`.
   * التدريب لمدة 40 Epoch باستخدام `AdamW (lr=3e-3)` ودالة خسارة **`Permutation-Invariant Hungarian Matching (PIL) BCE + Dice Loss`** على قسم الـ `DEV` (انخفض الـ Loss من `0.3033` إلى `0.0413`)، مع دمج Residual Blending ($\lambda^* = 0.40$) لمنع الـ Double-Padding.
   * **الأثر:** حقق نقطة التوازن المثالية (**Equal Error Rate Operating Point**: `Missed = 4.44% ≈ False Alarm = 4.40%`) ونزل بالـ `DER_0.0s` إلى **`9.14%`** والـ `JER` إلى **`9.60%`**.
7. **`Exp 6: + Acoustic-Energy Conditioning & Collision-Free Turn Handoff`**
   * تنعيم جاوسي زمني ($\sigma = 40\text{ms}$) على احتمالات الـ `10ms`.
   * ربط الاحتمالات بطاقة الصوت الفيزيائية الحقيقية (`10ms RMS Energy Envelope`) من ملف الـ WAV.
   * منع تصادم الحواف (**Collision-Free Padding & Gap Bridging**): منع توسيع نهاية المتحدث الأول أو قفل فجوة النَفَس إذا دخل متحدث آخر باحتمال قوي ($> 0.55$) في نفس اللحظة.
   * **الأثر:** أفضل نتيجة على الإطلاق (**`DER_0.0s = 9.09%`**، **`DER_0.25s = 7.54%`**، **`JER = 9.57%`** على العينتين، و **`8.29% DER_0.0s` / `7.01% DER_0.25s`** على الـ 13 ملف كاملة!).

---

### جدول الـ Ablation Study المجمّع (متوسط العينتين `C03-R04` & `C03-R08`)

| المرحلة (Experiment) | `ref_speakers` | `pred_speakers` | `DER_0.0s %` ↓ | `Missed %` ↓ | `FalseAlarm %` ↓ | `Confusion %` ↓ | `DER_0.25s %` ↓ | `JER %` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 0: Baseline (Raw Nemotron-3)** | 3.0 | 3.5 | 24.86% | 23.05% | 1.74% | 0.06% | 22.43% | 20.98% |
| **Exp 1: + Audio Preprocessing** | 3.0 | 3.5 | 22.76% | 20.73% | 1.97% | 0.06% | 20.41% | 19.34% |
| **Exp 2: + Math Layer Only** | 3.0 | 3.0 | 10.26% | 6.74% | 3.36% | 0.17% | 8.66% | 9.94% |
| **Exp 3: Preproc + Math** | 3.0 | 3.0 | 9.98% | 5.97% | 3.78% | 0.24% | 8.37% | 9.99% |
| **Exp 4: Auto-Calibrated (Learned on DEV)** | 3.0 | 3.0 | 9.21% | 3.60% | 5.21% | 0.40% | 7.57% | 9.76% |
| **Exp 5: Frozen-Backbone Fine-Tuned** | 3.0 | 3.0 | 9.14% | 4.44% | 4.40% | 0.31% | 7.56% | 9.60% |
| **Exp 6: + Acoustic-Energy & Collision-Free** | **3.0** | **3.0** | **9.09%** | **4.32%** | **4.42%** | **0.36%** | **7.54%** | **9.57%** |

---

## 4. النتائج النهائية الرسمية على الـ 13 ملف كاملة (`ArzEn Test` — +4 ساعات)

### ملخص المقارنة النهائية على الـ 13 ملف:

| النظام (على 13 مقابلة = 248 دقيقة) | `ref_speakers` | `pred_speakers` | `DER_0.0s %` ↓ | `Missed %` ↓ | `FalseAlarm %` ↓ | `Confusion %` ↓ | `DER_0.25s %` ↓ | `JER %` ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw NVIDIA Nemotron-3)** | 2.92 | 3.46 *(غلط في 7/13)* | 21.84% | 19.06% | 2.14% | 0.64% | 19.72% | 20.72% |
| **Ours (`Exp 6`: Preproc + FT Adapter + Math)** | **2.92** | **2.92 (13/13 صح = 100%)** | **8.29%** | **3.63%** | **3.67%** | **0.99%** | **7.01%** | **11.21%** |

### الجدول التفصيلي لكل ملف من الـ 13 ملف (قبل وبعد):

| # | الملف (`recording_id`) | المدة (دقيقة) | العدد الحقيقي | توقع Baseline | **توقعنا (Ours)** | `DER_0.0s` (Baseline) | **`DER_0.0s` (Ours)** | `DER_0.25s` (Baseline) | **`DER_0.25s` (Ours)** | `JER` (Baseline) | **`JER` (Ours)** |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | `C03-R04` | 17.3 | 3 | 3 | **3** | 25.85% | **10.31%** | 23.31% | **8.79%** | 22.77% | **12.77%** |
| 1 | `C03-R08` | 20.7 | 3 | 4 | **3** | 23.86% | **7.87%** | 21.55% | **6.28%** | 19.18% | **6.38%** |
| 2 | `C03-R15` | 18.5 | 3 | 4 | **3** | 17.18% | **7.41%** | 15.43% | **6.63%** | 21.22% | **14.86%** |
| 3 | `C03-R18` | 16.9 | 3 | 4 | **3** | 24.09% | **13.15%** | 21.37% | **11.50%** | 24.19% | **15.50%** |
| 4 | `C03-R19` | 19.2 | 3 | 4 | **3** | 18.45% | **5.68%** | 16.86% | **4.55%** | 16.84% | **8.09%** |
| 5 | `C03-R22` | 20.9 | 3 | 3 | **3** | 22.20% | **7.56%** | 20.28% | **6.42%** | 20.16% | **10.81%** |
| 6 | `C03-R24` | 22.7 | 3 | 4 | **3** | 22.80% | **9.61%** | 20.95% | **8.53%** | 18.12% | **9.55%** |
| 7 | `C03-R26` | 20.5 | 3 | 3 | **3** | 24.67% | **9.25%** | 23.09% | **8.51%** | 23.82% | **15.15%** |
| 8 | `C03-R30` | 16.4 | 3 | 4 | **3** | 24.14% | **10.93%** | 21.50% | **9.20%** | 26.49% | **16.55%** |
| 9 | `C03-R32` | 18.6 | 3 | 3 | **3** | 19.68% | **7.64%** | 17.47% | **6.45%** | 22.32% | **14.05%** |
| 10 | `C03-R33` | 16.7 | 2 | 2 | **2** | 19.76% | **4.87%** | 17.71% | **3.58%** | 19.94% | **6.33%** |
| 11 | `C03-R35` | 16.5 | 3 | 4 | **3** | 23.72% | **10.05%** | 21.26% | **8.02%** | 19.04% | **10.13%** |
| 12 | `C03-R36` | 23.3 | 3 | 3 | **3** | 17.52% | **3.46%** | 15.56% | **2.62%** | 15.33% | **5.59%** |

---

## 5. تجارب `Shark Tank Egypt` (`video_10_part_03.wav`) والتشخيص العلمي للانتقال للنوتبوك الثانية

تم اختبار `Nemotron-3` بعدة أوضاع على عرض `video_10_part_03.wav` (مدته 10.58 دقيقة، يحتوي على 8 أشخاص: 3 رواد أعمال بينهم أختان بصوت متشابه جداً + 5 Sharks منهم 3 تحدثوا قليلاً جداً وهم أيمن وبيشوي وطارق):

1. **وضع `ArzEn` الافتراضي:** أخرج 4 متحدثين فقط بسبب شرط `min_spk_energy_sec = 15.0` وتحيز طبقة `cross_spk` المدربة على 3 متحدثين في `ArzEn`.
2. **وضع `Positive-Bridging 7-Speaker Mode` و `Raw NVIDIA`:** أخرج 7 متحدثين (`Channel 7` كانت قيمتها `0.056` = صفر إطارات)، ولكن تبين بالفحص السمعي والبصري أن **`speaker_0` و `speaker_1` استحوذوا على ~520 ثانية (أكثر من 8.5 دقيقة من أصل 10.5 دقيقة!)**، أي أن الموديل الأساسي دمج معظم أسئلة الـ Sharks داخل قنوات رواد الأعمال وابتلع المداخلات القصيرة لأيمن وبيشوي وطارق!
3. **وضع تقطيع النوافذ مع `ECAPA-TDNN` (`Auto-K` / `Voiceprint Bank`):** التقط المداخلات المتأخرة في آخر دقيقتين، لكنه تسبب في تقسيم المتحدث الرئيسي إلى أكثر من اسم أو حساسية زائدة في المقاطع المتداخلة.

### الأسباب العلمية الـ 3 لفشل معمارية `Sortformer / AOSC` على عروض Shark Tank (مادة علمية للمناقشة):
1. **`Delayed Speaker Arrival` وتشبع ذاكرة الـ `105s AOSC`:**
   `Nemotron-3` مدرب على مقاطع طولها الأقصى **`105 ثانية`** بترتيب أول ظهور (`ATS`). في Shark Tank، أول 3 دقائق (`180 ثانية`) يتحدث فيها رواد الأعمال فقط أثناء الـ Pitch والـ 5 Sharks صامتون؛ فتتشبع ذاكرة `AOSC` (`264 frame`) بصوت رواد الأعمال (`speaker_0` و `speaker_1`) ويحدث **Prior Collapse** ينسب كلام الـ Sharks اللاحق إلى `speaker_0` و `speaker_1`.
2. **`Heavy-Tailed Speaker Distribution` ضد دالة `Phantom Loss`:**
   المستثمرون الذين يتحدثون جملاً قصيرة جداً في نهاية العرض (أيمن، بيشوي، طارق) تقمعهم دالة `Phantom Loss` في القنوات المتأخرة (`Channel 6 & 7`) باعتبارهم ضوضاء وهمية.
3. **`Sibling Vocal-Tract Overlap` + الموسيقى التصويرية (`Co-Channel Music`):**
   التشابه الصوتي الجيني بين الأختين يسبب تبديل القنوات (`Channel Permutation Swap`)، والموسيقى التصويرية المستمرة في الخلفية ترفع احتمالات البقاء فوق عتبة الإغلاق.

### خطة النوتبوك الثانية (Notebook 2):
تجربة ومقارنة معمارية **`Local-to-Global EEND + Vector Clustering`** — وتحديداً موديل **`DiariZen` (`WavLM + Conformer + ResNet34-LM + AHC` — الموجود في Slide 7 من العرض التقديمي للمشروع)** و/أو **`Pyannote 3.1`** على حلقات **Shark Tank Egypt**، لأنها تعالج النوافذ محلياً (8–16 ثانية بدون تشبع ذاكرة الـ 105 ثانية) وتجمع المتحدثين عالمياً ببصمة الصوت (`ResNet34-LM`) بدون تحيز لوقت الظهور.
