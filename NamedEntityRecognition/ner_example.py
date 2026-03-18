import spacy
from spacy import displacy
import sys


def load_model(model_name="tr_core_news_lg"):
    """
    Belirtilen spaCy modelini yükler. Model bulunamazsa kullanıcıyı bilgilendirir.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"HATA: '{model_name}' modeli sistemde bulunamadı.")
        print(
            f"Lütfen şu komutu çalıştırın: python -m spacy download {model_name}")
        sys.exit(1)


def read_text(file_path=None):
    """
    Eğer bir dosya yolu verilmişse dosyayı okur, yoksa varsayılan uzun metni döner.
    """
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(
                f"Uyarı: {file_path} bulunamadı, varsayılan metin kullanılıyor.")

    # Varsayılan Uzun Metin
    return (
        "Türkiye'nin teknoloji hamlesi kapsamında, Togg'un Gemlik kampüsünde yeni üretim bantları "
        "devreye alındı. Sanayi ve Teknoloji Bakanı, ASELSAN ve HAVELSAN gibi kuruluşların "
        "yapay zeka projelerine verdiği desteği vurguladı. Öte yandan, İstanbul Teknik Üniversitesi "
        "(İTÜ) Arı Teknokent bünyesinde faaliyet gösteren girişimler, Avrupa Birliği'nden "
        "10 milyon Euro hibe almayı başardı. Geçtiğimiz Mart ayında Ankara'da düzenlenen "
        "TEKNOFEST, 1 milyondan fazla ziyaretçiyi ağırlayarak rekor kırdı."
    )


def analyze_entities(nlp, text):
    """
    Metin üzerindeki Adlandırılmış Varlık Tanıma (NER) işlemini yapar ve ekrana basar.
    """
    doc = nlp(text)

    print(f"\n{'Varlık':<35} | {'Etiket':<12} | {'Açıklama'}")
    print("-" * 75)

    for ent in doc.ents:
        aciklama = spacy.explain(ent.label_)
        print(f"{ent.text:<35} | {ent.label_:<12} | {aciklama}")

    return doc


def visualize(doc):
    """
    Analiz sonuçlarını lokal bir sunucu üzerinden (port 5000) görselleştirir.
    """
    print("\n>>> Görselleştirme başlatılıyor... http://127.0.0.1:5000 adresini ziyaret edin.")
    print(">>> Durdurmak için Terminal'de CTRL+C tuşlarına basın.")
    displacy.serve(doc, style="ent")


if __name__ == "__main__":
    # 1. Modeli hazırla
    nlp_model = load_model()

    # 2. Veriyi al (İstersen 'verilerim.txt' gibi bir dosya adı verebilirsin)
    txt_file = "text_file.txt"
    analysis_text = read_text(file_path=None)

    # 3. Analizi gerçekleştir
    processed_doc = analyze_entities(nlp_model, analysis_text)

    # 4. Web arayüzünde göster
    visualize(processed_doc)
