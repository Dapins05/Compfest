# STATISTICS.md - Landasan Statistik VisionQC

**Diferensiator utama modul AI.** Dokumen ini berisi seluruh rumus statistik yang dipakai,
lengkap dengan alasan pemakaian dan di langkah mana ia diterapkan.

> **Kenapa ini penting.** Peserta lain umumnya berhenti di: latih model  laporkan akurasi.
> Sistem QC industri tidak bisa berhenti di situ, karena pertanyaan sebenarnya adalah
> *"seberapa yakin kita boleh percaya pada keputusan ini?"* - dan itu pertanyaan statistik,
> bukan pertanyaan deep learning.

---

## Daftar Isi

| bagian | Topik | Dipakai di Step |
|---|---|---|
| 1 | [Notasi](#1-notasi) | - |
| 2 | [Validasi Dataset & Split](#2-validasi-dataset--split) | 2, 3 |
| 3 | [Estimasi Metrik dengan Ketidakpastian](#3-estimasi-metrik-dengan-ketidakpastian) | 4, 5 |
| 4 | [Uji Signifikansi Fine-Tuning](#4-uji-signifikansi-fine-tuning-) | 4, 5 |
| 5 | [Kalibrasi Kepercayaan Model](#5-kalibrasi-kepercayaan-model-) | 7 |
| 6 | [Conformal Prediction](#6-conformal-prediction-) | 7 |
| 7 | [Ambang Anomali via Extreme Value Theory](#7-ambang-anomali-via-extreme-value-theory-) | 6 |
| 8 | [Keputusan Sensitif Biaya](#8-keputusan-sensitif-biaya) | 7, 9 |
| 9 | [Statistical Process Control](#9-statistical-process-control-spc) | 10 (laporan) |
| 10 | [Kesepakatan dengan Inspektor Manusia](#10-kesepakatan-dengan-inspektor-manusia) | 10 |
| 11 | [Ringkasan Penerapan](#11-ringkasan-penerapan) | - |

---


## 1. Notasi

| Simbol | Arti |
|---|---|
| $n$ | ukuran sampel |
| $\hat{p}$ | proporsi hasil estimasi |
| $z_{\alpha/2}$ | kuantil normal baku ($z_{0,025} = 1{,}96$ untuk 95%) |
| $\alpha$ | tingkat signifikansi / galat yang diizinkan |
| $Y$ | label sebenarnya, $\hat{Y}$ label prediksi |
| $s(x,y)$ | skor ketidaksesuaian (*nonconformity score*) |
| $\text{TP, TN, FP, FN}$ | true/false positive/negative |

Kelas positif = **cacat** . Ini penting: seluruh metrik berorientasi pada kemampuan menangkap cacat.


---

## 2. Validasi Dataset & Split

### 2.1 Uji χ² keselarasan distribusi kelas antar split

Split train/val/test yang timpang membuat seluruh evaluasi tidak sah. Kita uji secara formal
apakah distribusi kelas pada tiap split konsisten dengan distribusi populasi.

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}, \qquad df = k - 1$$

- $O_i$ = frekuensi teramati kelas $i$ pada split
- $E_i$ = frekuensi harapan = $n_{\text{split}} \times p_i^{\text{populasi}}$

**Kriteria:**$p > 0{,}05$  gagal menolak $H_0$  distribusi split **selaras** (yang kita inginkan).

>  Catatan kejujuran: ini adalah kasus di mana kita *berharap* gagal menolak $H_0$. Uji ini
> berfungsi sebagai **pemeriksaan kewarasan** , bukan bukti kesamaan distribusi.

### 2.2 Ketimpangan kelas

**Rasio ketimpangan:**$$IR = \frac{n_{\max}}{n_{\min}}$$

**Entropi Shannon ternormalisasi:**$$H_{\text{norm}} = \frac{-\sum_{i=1}^{k} p_i \log p_i}{\log k} \in [0,1]$$

$H_{\text{norm}} = 1$ berarti sempurna seimbang. Target setelah augmentasi sintetik (Step 3):
$IR < 3$ dan $H_{\text{norm}} > 0{,}85$.

### 2.3 Ukuran sampel minimum test set

Berapa banyak sampel cacat yang dibutuhkan agar estimasi recall bermakna? Untuk margin galat $E$
pada tingkat kepercayaan $1-\alpha$:

$$n \geq \frac{z_{\alpha/2}^2 \, p(1-p)}{E^2}$$

**Contoh perhitungan.** Target recall $p \approx 0{,}90$, margin galat $E = 0{,}05$, kepercayaan 95%:

$$n \geq \frac{1{,}96^2 \times 0{,}90 \times 0{,}10}{0{,}05^2} = \frac{0{,}3457}{0{,}0025} \approx 139$$

 **butuh minimal ±139 sampel cacat di test set.** Kalau tidak terpenuhi, margin galatnya harus
dilaporkan apa adanya di proposal - bukan disembunyikan.

### 2.4 Validasi kualitas data sintetik

Data sintetik yang distribusinya terlalu jauh dari data asli justru merusak model. Kita ukur
memakai **jarak Wasserstein-1** antara histogram intensitas gambar asli dan sintetik:

$$W_1(P, Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| \, dx$$

dengan $F$ adalah fungsi distribusi kumulatif. Nilai kecil = sintetik menyerupai asli.
Pelengkap: **uji dua sampel Kolmogorov-Smirnov**$$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$

---

## 3. Estimasi Metrik dengan Ketidakpastian

> **Prinsip:**melaporkan "recall = 0,94" tanpa selang kepercayaan adalah menyembunyikan informasi.
> Recall 0,94 dari 50 sampel dan dari 5.000 sampel adalah dua klaim yang sangat berbeda.

### 3.1 Metrik dasar

$$\text{Recall} = \frac{TP}{TP+FN}, \qquad \text{Precision} = \frac{TP}{TP+FP}$$

$$F_\beta = (1+\beta^2)\cdot\frac{\text{Precision}\cdot\text{Recall}}{\beta^2\cdot\text{Precision} + \text{Recall}}$$

Kita memakai $\beta = 2$ (**F2-score** ), bukan F1. Alasannya: $\beta > 1$ memberi bobot lebih besar
pada recall, dan di QC pangan, cacat yang lolos ke konsumen jauh lebih mahal daripada produk bagus
yang salah ditolak.

**Matthews Correlation Coefficient** - metrik tunggal terbaik untuk data timpang:

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in [-1, 1]$$

MCC hanya tinggi bila keempat sel matriks konfusi bagus - tidak bisa "ditipu" oleh kelas mayoritas
seperti akurasi.

### 3.2 Selang kepercayaan Wilson (bukan Wald)

Selang Wald yang umum dipakai, $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$, **rusak** saat $n$ kecil
atau $\hat{p}$ mendekati 0/1 - batasnya bisa keluar dari $[0,1]$. Gunakan **selang skor Wilson** :

$$\text{CI}_{\text{Wilson}} = \frac{\hat{p} + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat{p}(1-\hat{p})}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}$$

Untuk jaminan cakupan eksak (konservatif), pakai **Clopper-Pearson** berbasis distribusi Beta:

$$\left[ B\left(\tfrac{\alpha}{2}; x, n-x+1\right), \; B\left(1-\tfrac{\alpha}{2}; x+1, n-x\right) \right]$$

### 3.3 Bootstrap BCa untuk metrik kompleks

mAP dan IoU tidak punya rumus selang kepercayaan tertutup. Gunakan **bootstrap bias-corrected and
accelerated (BCa)** , $B = 2000$ resampel:

$$\hat{\theta}^*_{(1)}, \ldots, \hat{\theta}^*_{(B)} \quad \text{dari resampling dengan pengembalian}$$

Koreksi bias:
$$\hat{z}_0 = \Phi^{-1}\!\left(\frac{\#\{\hat{\theta}^*_b < \hat{\theta}\}}{B}\right)$$

Akselerasi (dari jackknife):
$$\hat{a} = \frac{\sum_{i=1}^{n}(\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^3}{6\left[\sum_{i=1}^{n}(\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^2\right]^{3/2}}$$

Batas selang pada kuantil terkoreksi:
$$\alpha_1 = \Phi\!\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)$$

**Bentuk pelaporan yang dipakai:**> mAP@50 = 0,873 (95% BCa CI: [0,841 - 0,902], B = 2000)

---

## 4. Uji Signifikansi Fine-Tuning

> **Ini menjawab langsung kewajiban panitia.** Panitia mewajibkan model di-fine-tune. Sebagian besar
> peserta akan membuktikannya dengan menampilkan dua angka. Kita membuktikannya dengan **uji hipotesis** .

### 4.1 Uji McNemar

Baseline dan model hasil fine-tuning dievaluasi pada **test set yang sama** datanya berpasangan,
sehingga uji dua-proporsi biasa tidak sah. Yang benar adalah **uji McNemar** .

Tabel kontingensi 2×2:

| | Fine-tuned **benar** | Fine-tuned **salah** |
|---|---|---|
| **Baseline benar** | $a$ | $b$ |
| **Baseline salah** | $c$ | $d$ |

Hanya sel diskordan ($b$ dan $c$) yang membawa informasi.

$$\chi^2_{\text{McNemar}} = \frac{(|b - c| - 1)^2}{b + c}, \qquad df = 1$$

(angka $-1$ adalah koreksi kontinuitas Yates)

Bila $b + c < 25$, gunakan **versi binomial eksak** :

$$p = 2 \sum_{i=c}^{b+c} \binom{b+c}{i} (0{,}5)^{b+c}$$

**Hipotesis:** - $H_0$: tidak ada perbedaan kinerja antara baseline dan model fine-tuned
- $H_1$: ada perbedaan

**Bentuk pelaporan:**> Fine-tuning memperbaiki 87 kasus yang sebelumnya salah, dan merusak 12 kasus yang sebelumnya
> benar. Uji McNemar: χ² = 54,3, p < 0,001  peningkatan **signifikan secara statistik** .

### 4.2 Besar efek - Cohen's h

Nilai p hanya menyatakan *ada* perbedaan, bukan *seberapa besar*. Untuk selisih proporsi:

$$h = 2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_2}$$

| $|h|$ | Tafsiran |
|---|---|
| 0,20 | kecil |
| 0,50 | sedang |
| 0,80 | besar |

### 4.3 Bootstrap berpasangan untuk selisih mAP


$$\Delta = \text{mAP}_{\text{finetuned}} - \text{mAP}_{\text{baseline}}$$

Resampel gambar test **secara berpasangan** (kedua model dievaluasi pada resampel yang sama),
lalu bangun selang kepercayaan untuk $\Delta$. Bila selang **tidak memuat nol** , peningkatan
signifikan.

---

## 5. Kalibrasi Kepercayaan Model

> **Masalah yang jarang disadari:**jaringan saraf modern hampir selalu *overconfident*. Ketika model
> berkata "confidence 0,9", ia sebenarnya benar hanya sekitar 70-75% dari waktu. Padahal seluruh
> logika ambang batas kita bergantung pada angka kepercayaan itu. Kalau tidak dikalibrasi,
> ambang apa pun yang dipilih menjadi tidak bermakna.

### 5.1 Expected Calibration Error

Bagi prediksi ke dalam $M$ bin berdasarkan kepercayaan:

$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

$$MCE = \max_{m} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Model terkalibrasi sempurna: $ECE = 0$. Target kita: $ECE < 0{,}05$ setelah kalibrasi.

### 5.2 Temperature scaling

Kalibrasi pasca-latih paling sederhana dan efektif - **satu parameter** , tidak mengubah urutan
prediksi sehingga akurasi tidak berubah sama sekali:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

$T$ dioptimasi pada **himpunan validasi** (bukan test) dengan meminimumkan negative log-likelihood:

$$T^* = \arg\min_T \; -\sum_{i=1}^{n} \log \hat{p}_{i, y_i}(T)$$

$T > 1$ melunakkan kepercayaan (mengatasi *overconfidence*), $T < 1$ menajamkan.

### 5.3 Skor Brier

Ukuran gabungan akurasi dan kalibrasi:

$$BS = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i)^2$$

Dekomposisi Murphy: $BS = \underbrace{\text{Reliability}}_{\downarrow \text{ lebih baik}} - \underbrace{\text{Resolution}}_{\uparrow \text{ lebih baik}} + \underbrace{\text{Uncertainty}}_{\text{sifat data}}$

---

## 6. Conformal Prediction

> **Ini permata dari seluruh pendekatan.** Conformal prediction memberi **jaminan cakupan yang
> berlaku tanpa asumsi distribusi** dan **tanpa asumsi apa pun tentang modelnya** . Satu-satunya
> syarat adalah *exchangeability* data. Inilah yang mengubah kelas `REVIEW` dari tebakan menjadi
> konsekuensi matematis.

### 6.1 Split conformal prediction

**Prosedur:**1. Sisihkan himpunan kalibrasi berukuran $n$ (terpisah dari train dan test)
2. Hitung skor ketidaksesuaian tiap sampel kalibrasi. Untuk klasifikasi:
   $$s_i = 1 - \hat{p}(y_i \mid x_i)$$
3. Hitung kuantil terkoreksi:
   $$\hat{q} = \text{Quantile}\left(s_1, \ldots, s_n; \; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
4. Untuk sampel baru $x$, bentuk **himpunan prediksi** :
   $$C(x) = \{\, y : 1 - \hat{p}(y \mid x) \leq \hat{q} \,\}$$

**Jaminan yang diperoleh:**$$\boxed{\; \mathbb{P}\big(Y_{\text{test}} \in C(X_{\text{test}})\big) \;\geq\; 1 - \alpha \;}$$

Berlaku untuk **model apa pun** , **distribusi apa pun** , pada **ukuran sampel berhingga** .
Bukan asimptotik, bukan aproksimasi.

### 6.2 Aturan keputusan berbasis himpunan prediksi

$$
\text{Verdict}(x) =
\begin{cases}
\text{PASS} & |C(x)| = 1 \;\wedge\; C(x) = \{\text{normal}\} \\
\text{REJECT} & |C(x)| = 1 \;\wedge\; C(x) = \{\text{cacat}_k\} \\
\text{REVIEW} & |C(x)| \neq 1 \quad \text{(ambigu atau kosong)}
\end{cases}
$$

Inilah yang bisa dinyatakan di proposal:

> *"Sistem menahan keputusan bukan karena ambang batas yang dipilih secara arbitrer, melainkan
> karena himpunan prediksi conformal pada tingkat kepercayaan 95% memuat lebih dari satu label -
> artinya data tidak memberi cukup bukti untuk memisahkan keduanya."*

### 6.3 Conformal Mondrian (kondisional per kelas)

Cakupan marginal saja tidak cukup untuk data timpang: cakupan bisa 95% secara keseluruhan tapi
hanya 60% pada kelas cacat. Solusinya, hitung kuantil **terpisah per kelas** :

$$\hat{q}_k = \text{Quantile}\left(\{s_i : y_i = k\}; \; \frac{\lceil (n_k+1)(1-\alpha) \rceil}{n_k}\right)$$

Menghasilkan jaminan **kondisional per kelas** :
$$\mathbb{P}\big(Y \in C(X) \mid Y = k\big) \geq 1 - \alpha \quad \forall k$$

Untuk QC, inilah yang benar-benar dibutuhkan - jaminan pada kelas cacat tidak boleh dikorbankan
demi rata-rata.

### 6.4 Validasi empiris

Kita verifikasi jaminan itu benar-benar terpenuhi pada test set:

$$\text{Cakupan empiris} = \frac{1}{n_{\text{test}}}\sum_{i=1}^{n_{\text{test}}} \mathbb{1}\{y_i \in C(x_i)\} \;\approx\; 1-\alpha$$

$$\text{Ukuran himpunan rata-rata} = \frac{1}{n_{\text{test}}}\sum_{i=1}^{n_{\text{test}}} |C(x_i)|$$

Himpunan yang lebih kecil = model lebih informatif, pada tingkat cakupan yang sama.

---

## 7. Ambang Anomali via Extreme Value Theory

> **Masalah:**dari mana angka ambang 0,75 untuk skor anomali? Kalau jawabannya "kelihatannya bagus",
> itu titik lemah yang akan langsung dikejar juri. **Teori Nilai Ekstrem** memberi jawaban yang bisa
> dipertahankan.

### 7.1 Peaks Over Threshold + Generalized Pareto

**Teorema Pickands-Balkema-de Haan:**untuk ambang $u$ yang cukup tinggi, distribusi kelebihan
$(X - u \mid X > u)$ konvergen ke **Generalized Pareto Distribution** :

$$G_{\xi,\sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi}, \qquad y > 0, \; 1 + \frac{\xi y}{\sigma} > 0$$

**Prosedur:**1. Jalankan model anomali pada gambar **normal saja** (validasi)  kumpulkan skor
2. Pilih ambang awal $u$ = kuantil ke-95 dari skor tersebut
3. Ambil kelebihan $Y_i = X_i - u$ untuk semua $X_i > u$; misalkan ada $N_u$ buah
4. Estimasi $\hat{\xi}, \hat{\sigma}$ dengan maximum likelihood
5. Hitung ambang akhir untuk laju alarm palsu target $q$:

$$\boxed{\; z_q = u + \frac{\hat{\sigma}}{\hat{\xi}}\left[\left(\frac{q \, n}{N_u}\right)^{-\hat{\xi}} - 1\right] \;}$$

dengan $n$ = total sampel normal.

**Bentuk pelaporan:**> Ambang anomali 0,7412 diperoleh dengan memodelkan ekor distribusi skor pada sampel normal memakai
> GPD (ξ̂ = 0,183; σ̂ = 0,0417; $N_u$ = 47 dari n = 940), menjamin laju alarm palsu ≤ 1%.

### 7.2 Diagnostik kecocokan

- **Mean residual life plot** - memastikan $u$ dipilih di wilayah yang linear
- **Q-Q plot** terhadap GPD teoretis
- **Uji Anderson-Darling** untuk kecocokan ekor

### 7.3 Tafsiran parameter bentuk

| $\xi$ | Jenis ekor | Implikasi |
|---|---|---|
| $\xi > 0$ | berat (Fréchet) | skor ekstrem mungkin muncul  butuh margin lebih |
| $\xi = 0$ | eksponensial (Gumbel) | peluruhan wajar |
| $\xi < 0$ | terbatas (Weibull) | ada batas atas skor |

---


## 8. Keputusan Sensitif Biaya

> Di QC pangan, biaya kesalahan **sangat tidak simetris** . Cacat yang lolos ke konsumen bisa berarti
> keluhan, retur, bahkan penarikan produk. Salah menolak satu produk bagus hanya rugi sebesar harga
> produk itu. Ambang batas 0,5 secara diam-diam mengasumsikan kedua biaya sama besar - dan asumsi
> itu keliru.

### 8.1 Aturan keputusan Bayes

Dengan biaya salah tolak $C_{FP}$ dan biaya cacat lolos $C_{FN}$, tolak produk bila:

$$P(\text{cacat} \mid x) > \frac{C_{FP}}{C_{FP} + C_{FN}}$$

**Contoh dengan angka nyata (silakan sesuaikan):** | Besaran | Nilai |---|---|
| $C_{FN}$ - cacat lolos ke konsumen | Rp 50.000 | $C_{FP}$ - produk bagus salah ditolak | Rp 2.000 | $$\tau^* = \frac{2.000}{2.000 + 50.000} = \frac{2.000}{52.000} \approx 0{,}0385$$

Artinya sistem seharusnya menolak bahkan pada kecurigaan cacat sekitar **3,85%** - jauh lebih
rendah dari 0,5 yang biasa dipakai orang tanpa berpikir. Ini contoh nyata bahwa keputusan yang
"benar secara teknis" bisa sangat salah secara ekonomi.

### 8.2 Ekspektasi biaya total

$$\mathbb{E}[C] = C_{FP} \cdot n_{FP} + C_{FN} \cdot n_{FN} + C_{\text{review}} \cdot n_{\text{review}}$$

Ambang optimal diperoleh dengan meminimumkan fungsi di atas pada himpunan validasi. Kelas
`REVIEW` punya biayanya sendiri (waktu operator) - sehingga sistem tidak bisa "curang" dengan
melemparkan semuanya ke manusia.

### 8.3 Indeks Youden (bila biaya dianggap setara)

$$J = \text{Sensitivity} + \text{Specificity} - 1$$

Titik optimal = $\arg\max_\tau J(\tau)$. Kita laporkan **keduanya** - ambang Youden dan ambang
sensitif biaya - untuk menunjukkan seberapa jauh keduanya berbeda.

---

## 9. Statistical Process Control (SPC)

> **Kenapa ini kuat:**SPC adalah bahasa asli insinyur kualitas manufaktur. Menghubungkan keluaran
> AI ke kartu kendali menunjukkan sistem ini dirancang untuk pabrik sungguhan, bukan sekadar
> eksperimen laboratorium.

>  **Kepatuhan:**SPC hanya dipakai sebagai **analisis luring pada laporan evaluasi dan proposal** ,
> bukan sebagai layanan pemantauan berjalan - pemantauan otomatis termasuk yang dibatasi panitia
> untuk tahap penyisihan. Versi runtime-nya diparkir ke tahap Final.

### 9.1 Kartu p - proporsi cacat

$$\bar{p} = \frac{\sum_{i=1}^{m} d_i}{\sum_{i=1}^{m} n_i}, \qquad
UCL/LCL = \bar{p} \pm 3\sqrt{\frac{\bar{p}(1-\bar{p})}{n}}$$

### 9.2 Kartu c - jumlah cacat per unit

$$\bar{c} = \frac{1}{m}\sum_{i=1}^{m} c_i, \qquad UCL/LCL = \bar{c} \pm 3\sqrt{\bar{c}}$$

### 9.3 CUSUM - mendeteksi pergeseran kecil

Kartu Shewhart lambat menangkap pergeseran kecil. CUSUM jauh lebih peka:

$$C_i^{+} = \max\left(0, \; C_{i-1}^{+} + (x_i - \mu_0) - k\right)$$
$$C_i^{-} = \max\left(0, \; C_{i-1}^{-} - (x_i - \mu_0) - k\right)$$

Alarm bila $C_i^{\pm} > h$. Pilihan lazim: $k = \delta/2$ (setengah pergeseran yang ingin dideteksi),
$h = 5\sigma$.

### 9.4 EWMA - rata-rata bergerak terboboti eksponensial

$$z_i = \lambda x_i + (1-\lambda) z_{i-1}, \qquad 0 < \lambda \leq 1$$

$$UCL/LCL = \mu_0 \pm L\sigma\sqrt{\frac{\lambda}{2-\lambda}\left[1-(1-\lambda)^{2i}\right]}$$

Nilai umum: $\lambda = 0{,}2$, $L = 3$.

### 9.5 Kapabilitas proses

$$C_p = \frac{USL - LSL}{6\sigma}, \qquad
C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \; \frac{\mu - LSL}{3\sigma}\right)$$

Diterapkan pada **persentase luas cacat** , dengan $USL$ = batas toleransi cacat.
$C_{pk} \geq 1{,}33$ adalah standar industri yang umum diterima.

### 9.6 Average Run Length

$$ARL_0 = \frac{1}{\alpha} \quad \text{(saat proses terkendali - makin besar makin baik)}$$
$$ARL_1 = \frac{1}{1-\beta} \quad \text{(saat proses bergeser - makin kecil makin baik)}$$

---

## 10. Kesepakatan dengan Inspektor Manusia

### 10.1 Kappa Cohen

Akurasi mentah menyesatkan bila kelas timpang - dua penilai bisa "sepakat" 95% hanya karena
sama-sama menebak kelas mayoritas. Kappa mengoreksi kesepakatan yang terjadi secara kebetulan:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$ = proporsi kesepakatan teramati
- $p_e$ = proporsi kesepakatan yang diharapkan secara kebetulan

| $\kappa$ | Tafsiran (Landis & Koch) |
|---|---|
| < 0,20 | lemah |
| 0,21-0,40 | cukup |
| 0,41-0,60 | sedang |
| 0,61-0,80 | kuat |
| 0,81-1,00 | hampir sempurna |

### 10.2 Alfa Krippendorff


Untuk menilai konsistensi pelabelan (berguna bila lebih dari satu orang melabeli data):

$$\alpha = 1 - \frac{D_o}{D_e}$$

dengan $D_o$ ketidaksepakatan teramati dan $D_e$ ketidaksepakatan yang diharapkan. Ambang yang
lazim diterima: $\alpha \geq 0{,}80$.

---

## 11. Ringkasan Penerapan

| Metode | Rumus inti | Menjawab pertanyaan | Step |
|---|---|---|---|
| Uji khi-kuadrat | $\sum (O-E)^2/E$ | Apakah split dataset sah? | 2 |
| Entropi Shannon | $-\sum p_i \log p_i / \log k$ | Seberapa timpang kelasnya? | 2, 3 |
| Ukuran sampel | $z^2p(1-p)/E^2$ | Cukupkah data test-nya? | 2 |
| Wasserstein / KS | $\int\|F_P - F_Q\|$ | Apakah data sintetik realistis? | 3 |
| **McNemar** | $(\|b-c\|-1)^2/(b+c)$ | **Apakah fine-tuning benar-benar berdampak?** | **4, 5** |
| Cohen's h | $2\arcsin\sqrt{p_1} - 2\arcsin\sqrt{p_2}$ | Seberapa besar dampaknya? | 4 |
| Bootstrap BCa | resampling $B=2000$ | Seberapa presisi metriknya? | 4, 5 |
| Wilson CI | lihat bagian 3.2 | Selang untuk proporsi | 4 |
| MCC | lihat bagian 3.1 | Metrik tunggal untuk data timpang | 4 |
| ECE + Temperature | $\sum \frac{\|B_m\|}{n}\|acc - conf\|$ | Apakah kepercayaan model bermakna? | 7 |
| **Conformal** | $\hat{q} = Q(s; \frac{\lceil(n+1)(1-\alpha)\rceil}{n})$ | **Kapan sistem harus menahan keputusan?** | **7** |
| **EVT / GPD** | $z_q = u + \frac{\sigma}{\xi}[(\frac{qn}{N_u})^{-\xi}-1]$ | **Dari mana ambang anomali berasal?** | **6** |
| Bayes sensitif biaya | $\tau^* = \frac{C_{FP}}{C_{FP}+C_{FN}}$ | Ambang mana yang paling murah? | 7, 9 |
| SPC (p, c, CUSUM, EWMA) | lihat bagian 9 | Apakah proses produksi terkendali? | 10 |
| Kappa Cohen | $(p_o - p_e)/(1-p_e)$ | Sebanding dengan inspektor manusia? | 10 |

---

## 12. Kejujuran Ilmiah

Beberapa hal yang **tidak boleh** dilakukan, dan akan langsung terlihat oleh juri yang paham statistik:

| Jangan | Kenapa |
|---|---|
| Melaporkan metrik tanpa selang kepercayaan | Menyembunyikan ketidakpastian |
| Menyetel ambang batas pada test set | Kebocoran data - angkanya jadi tidak sah |
| Menyebut "signifikan" tanpa uji hipotesis | Penyalahgunaan istilah statistik |
| Memakai akurasi pada data yang sangat timpang | Menyesatkan; pakai MCC atau F2 |
| Mengganti seed berulang kali sampai hasilnya bagus | Peretasan hasil |
| Mengarang angka yang belum diukur | Melanggar ; fatal bila juri meminta demo langsung |

> Semua angka pada [EXPERIMENTS.md](./EXPERIMENTS.md) **wajib** berasal dari run yang benar-benar
> dijalankan dan bisa direproduksi ulang.

---

## 13. Rujukan

1. Angelopoulos & Bates (2023). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.*
2. Vovk, Gammerman & Shafer (2005). *Algorithmic Learning in a Random World.*
3. Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
4. Coles (2001). *An Introduction to Statistical Modeling of Extreme Values.*
5. Siffer et al. (2017). *Anomaly Detection in Streams with Extreme Value Theory.* KDD.
6. Montgomery (2020). *Introduction to Statistical Quality Control*, 8th ed.
7. Efron & Tibshirani (1993). *An Introduction to the Bootstrap.*
8. Dietterich (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.*
9. Chicco & Jurman (2020). *The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy.* BMC Genomics.
10. Brown, Cai & DasGupta (2001). *Interval Estimation for a Binomial Proportion.* Statistical Science.

>  Anggota 3: verifikasi tiap rujukan sebelum masuk daftar pustaka proposal - jangan mengutip
> yang belum dibaca.
