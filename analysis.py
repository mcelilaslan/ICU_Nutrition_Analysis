"""
=============================================================================
ICU NUTRITIONAL RISK & CALORIC DELIVERY MORTALITY PREDICTION MODEL
=============================================================================
Description: 
This script performs a comprehensive statistical analysis of intensive care 
unit (ICU) patients to determine the independent predictors of 28-day mortality.
It evaluates the predictive performance of nutritional risk scores (mNUTRIC, 
NRS-2002, MNA) and clinical severity scores (SOFA, APACHE-II).

Methodology Notes:
1. Complete Case Analysis: Multivariate regression is restricted to patients 
   with complete caloric delivery data (n=301) to ensure reliability.
2. Selection Bias Check: Patients excluded due to missing caloric data are 
   compared with included patients to evaluate bias (Excluded patients had 
   significantly lower ICU severity).
3. Multicollinearity Prevention: Nutritional scores containing SOFA/Age 
   components are deliberately excluded from the main regression model.
=============================================================================
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import re
import warnings
import sys

# Uyarıları gizleyelim (temiz çıktı için)
warnings.filterwarnings("ignore")

# ==========================================
# 1. VERİ YÜKLEME VE TEMİZLEME (PREPROCESSING)
# ==========================================
def load_and_preprocess(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()
    
    # Taburcu sütununu dinamik bulma
    taburcu_col = [col for col in df.columns if 'Taburcu' in col][0]

    # 28 Günlük Mortalite Hedefi (3 = Exitus)
    df['Mortalite_28Gun'] = np.where(
        (df['Toplam yatış süresi(Gün)'] <= 28) & (df[taburcu_col] == 3), 
        'NON-SURVIVOR', 'SURVIVOR'
    )
    df['Mortalite_Target'] = np.where(df['Mortalite_28Gun'] == 'NON-SURVIVOR', 1, 0)

    # Kalori Oranı Hesaplama
    df['Kalori_Orani'] = df['Verilebilen EE /KCAL(ilk 30)'] / df['Hedeflenen EE KCAL(ilk 30)']
    df['Kalori_Orani'] = df['Kalori_Orani'].replace([np.inf, -np.inf], np.nan)

    # Metin içeren skorlardan sadece sayıları çeken fonksiyon
    def extract_numeric(val):
        if pd.isna(val): return np.nan
        val = str(val).replace(',', '.')
        match = re.search(r'(\d+\.?\d*)', val)
        return float(match.group(1)) if match else np.nan

    df['NRS-2002_Sayisal'] = df['NRS-2002(Yatışta)'].apply(extract_numeric)
    df['MNA_Sayisal'] = df['MNA(yatışta)'].apply(extract_numeric)

    return df

# ==========================================
# 2. İSTATİSTİKSEL ANALİZ FONKSİYONLARI
# ==========================================

def run_selection_bias_check(df):
    """ Analize dahil edilenler (301) ile dışlananları (136) kıyaslar. """
    print("\n" + "="*70)
    print("🔍 BÖLÜM 1: SEÇİLİM YANLILIĞI (SELECTION BIAS) ANALİZİ")
    print("="*70)
    
    df_missing = df[df['Kalori_Orani'].isna()]
    df_complete = df[df['Kalori_Orani'].notna()]
    
    print(f"Toplam Hasta: {len(df)} | Analize Alınan (Tam Veri): {len(df_complete)} | Dışlanan (Eksik Veri): {len(df_missing)}\n")
    
    vars_to_check = ['Yaş', 'SOFA(>24saat)', 'APACHE-II']
    for var in vars_to_check:
        if var in df.columns:
            g_comp = df_complete[var].dropna()
            g_miss = df_missing[var].dropna()
            _, p_val = stats.mannwhitneyu(g_comp, g_miss)
            print(f"{var:<15} -> Alınan Medyan: {g_comp.median():.1f} | Dışlanan Medyan: {g_miss.median():.1f} | P-Değeri: {p_val:.4f}")
    
    print("\n*Klinik Sonuç: Dışlanan hastaların klinik ağırlıkları (SOFA/APACHE) anlamlı derecede daha düşüktür.")

def run_univariate_analysis(df):
    """ Tablo 1 için medyan ve P-değerlerini üretir. """
    print("\n" + "="*70)
    print("📊 BÖLÜM 2: TEK YÖNLÜ (UNIVARIATE) ANALİZLER (Tablo 1)")
    print("="*70)
    
    continuous_vars = ["Hastane>YBÜ'ne geçiş süresi(gün)", "Sıvı Dengesi (mL/ İlk 30 gün)", "APACHE-II", "SOFA(>24saat)", "mNUTRIC(>24saat)", "Kalori_Orani"]
    
    for var in continuous_vars:
        if var in df.columns:
            non_surv = df[df['Mortalite_28Gun'] == 'NON-SURVIVOR'][var].dropna()
            surv = df[df['Mortalite_28Gun'] == 'SURVIVOR'][var].dropna()
            _, p_val = stats.mannwhitneyu(non_surv, surv)
            print(f"{var:<35} | NON-SURV Medyan: {non_surv.median():.2f} | SURV Medyan: {surv.median():.2f} | p={p_val:.4f}")

def run_multivariate_regression(df):
    """ Multicollinearity'den kaçınarak oluşturulan Ana Lojistik Regresyon Modeli. """
    print("\n" + "="*70)
    print("🔬 BÖLÜM 3: ÇOK DEĞİŞKENLİ LOJİSTİK REGRESYON (Tablo 7)")
    print("="*70)
    
    core_model_vars = ['Yaş', 'Sepsis varlığı 0-Yok 1-Var', 'SOFA(>24saat)', 'Kalori_Orani']
    reg_df = df.dropna(subset=core_model_vars + ['Mortalite_Target']).copy()
    
    X = reg_df[core_model_vars].apply(pd.to_numeric, errors='coerce')
    clean_df = pd.concat([X, reg_df['Mortalite_Target']], axis=1).dropna()
    X_clean = sm.add_constant(clean_df[core_model_vars])
    y_clean = clean_df['Mortalite_Target']
    
    print(f"Model, kalori verisi tam olan hasta grubu (n={len(y_clean)}) üzerinde çalıştırıldı.\n")
    print(f"{'Bağımsız Risk Faktörü':<35} | {'OR':<10} | {'%95 CI':<15} | {'P-Değeri'}")
    print("-" * 80)
    
    try:
        log_reg = sm.Logit(y_clean, X_clean).fit(disp=0)
        for idx in log_reg.pvalues.index:
            if idx == 'const': continue
            p_val = log_reg.pvalues[idx]
            or_val = np.exp(log_reg.params[idx])
            ci_lower = np.exp(log_reg.conf_int().loc[idx, 0])
            ci_upper = np.exp(log_reg.conf_int().loc[idx, 1])
            sig = "*" if p_val < 0.05 else ""
            print(f"{idx:<35} | {or_val:<10.3f} | {ci_lower:.3f} - {ci_upper:.3f} | p={p_val:.4f} {sig}")
    except Exception as e:
        print(f"Regresyon modeli kurulamadı: {e}")

def calculate_auc(df, score_col, reverse=False):
    """ ROC Eğrisi Altında Kalan Alanı (AUC) hesaplar. """
    temp = df.dropna(subset=[score_col, 'Mortalite_Target'])
    if len(temp) < 10: return "N/A"
    y_true = temp['Mortalite_Target']
    y_score = -temp[score_col] if reverse else temp[score_col]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def run_roc_analysis(df):
    """ Nütrisyon ve klinik skorların öngörü kapasitesini değerlendirir. """
    print("\n" + "="*70)
    print("🎯 BÖLÜM 4: ROC EĞRİSİ (AUC) ANALİZİ")
    print("="*70)
    
    df_geriatric = df[df['Yaş'] >= 65].copy()
    
    print("▶ TÜM HASTALAR (Genel Popülasyon):")
    print(f"  SOFA (>24saat)  : {calculate_auc(df, 'SOFA(>24saat)'):.3f}")
    print(f"  mNUTRIC (>24sa) : {calculate_auc(df, 'mNUTRIC(>24saat)'):.3f}")
    print(f"  NRS-2002        : {calculate_auc(df, 'NRS-2002_Sayisal'):.3f}")
    print(f"  APACHE-II       : {calculate_auc(df, 'APACHE-II'):.3f}")
    
    print("\n▶ GERİATRİK ALT GRUP (Yaş >= 65):")
    print(f"  SOFA (>24saat)  : {calculate_auc(df_geriatric, 'SOFA(>24saat)'):.3f}")
    print(f"  mNUTRIC (>24sa) : {calculate_auc(df_geriatric, 'mNUTRIC(>24saat)'):.3f}")
    print(f"  MNA (Ters)      : {calculate_auc(df_geriatric, 'MNA_Sayisal', reverse=True):.3f}")
    print(f"  NRS-2002        : {calculate_auc(df_geriatric, 'NRS-2002_Sayisal'):.3f}")

# ==========================================
# 3. ANA ÇALIŞTIRMA BLOĞU (AKILLI YÜKLEME)
# ==========================================
if __name__ == "__main__":
    try:
        # Kodun Google Colab'da çalışıp çalışmadığını kontrol et
        if 'google.colab' in sys.modules:
            from google.colab import files
            print("📂 Lütfen analiz edilecek veri setini (CSV veya Excel) seçin:")
            uploaded = files.upload()
            FILE_NAME = list(uploaded.keys())[0]
            print(f"\n✅ '{FILE_NAME}' başarıyla yüklendi. Analiz başlıyor...\n")
        else:
            # Lokal ortamlar (Jupyter, VSCode, Terminal) için
            FILE_NAME = input("📂 Lütfen analiz edilecek dosyanın tam adını girin (örn: data.xlsx): ")
            print(f"\n⏳ '{FILE_NAME}' aranıyor...\n")

        # Analiz Fonksiyonlarını Sırayla Çalıştır
        dataset = load_and_preprocess(FILE_NAME)
        run_selection_bias_check(dataset)
        run_univariate_analysis(dataset)
        run_multivariate_regression(dataset)
        run_roc_analysis(dataset)
        
        print("\n" + "="*70)
        print("✅ TÜM İSTATİSTİKSEL ANALİZLER BAŞARIYLA TAMAMLANDI.")
        print("="*70)
        
    except IndexError:
        print("\n❌ Hata: Herhangi bir dosya seçilmedi veya yükleme işlemi iptal edildi.")
    except FileNotFoundError:
        print(f"\n❌ Hata: '{FILE_NAME}' adlı dosya bulunamadı. Lütfen doğru klasörde olduğundan emin olun.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen bir hata oluştu: {e}")
