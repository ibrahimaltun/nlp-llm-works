# RAG

## Sıfırdan Vektör Arama Motoru

### Adım 1.Metni Elinizle Parçalayın (Manual Chunking)

- PDF'in içindeki ham metni doğrudan yapay zekaya verirsek model bunu sindiremez. Metni belirli karakter büyüklüklerinde (örneğin 500 karakter) küçük parçalara (chunk) bölmemiz gerekir. Cümlelerin ortadan ikiye bölünüp anlamını kaybetmemesi için de parçaları birbiri üzerine biraz bindiririz (overlap).

### Adım 2.Vektörleri Doğrudan Çıkarın

- sentence-transformers kütüphanesini kullanarak Türkçe destekli hafif bir modeli (multilingual-e5-small) doğrudan çağırıyoruz. Bu model, her bir metin parçası için bize 384 adet sayıdan oluşan uzun bir liste (vektör) verecek.

### Adım 3.Kosinüs Benzerliği (Cosine Similarity) ile Arama Yapın

- İşin en zevkli kısmı burası. Kullanıcı bir soru sorduğunda, sorunun da vektörünü alıyoruz. Ardından NumPy kullanarak, sorunun vektörü ile elimizdeki tüm metin parçalarının vektörleri arasındaki "açıyı" hesaplıyoruz. İki ok birbirine ne kadar yakın bakıyorsa, anlamları o kadar benzerdir.

----

## STAGE-by-STAGE

RAG mimarisi, **Offline Hazırlık ve Online Servis** şeklinde 2 ana safhaya bölünür.

**RAG Pipeline (İş Akışı)**:

## 1. FAZ: Hazırlık ve İndeksleme (Offline Ingestion Pipeline)

*Müşteri henüz masaya oturmadı; biz mutfakta malzemeleri önceden hazırlıyoruz.*

1. **Veri Toplama ve Temizleme (Data Ingestion & Parsing):**(1. Adım)
**Ne Yapılır?** PDF, TXT, SQL veya HTML gibi farklı kaynaklardaki veriler okunur. Metnin içindeki gereksiz boşluklar, sayfa numaraları veya dipnotlar temizlenir.
**Makarna Karşılığı:** Pazardan unu, domatesi, baharatı alıp yıkamak ve hazırlamak.

2. **Akıllı Parçalama (Semantic Chunking):**(2. Adım)
**Ne Yapılır?** Devasa metinler 500-1000 karakterlik küçük anlam bloklarına (chunks) bölünür. Cümle bütünlüğü bozulmasın diye parçalar arasında çakışma (overlap) bırakılır.
**Makarna Karşılığı:** Hamuru tencereye tek parça atmak yerine, pişecek boyutta erişte veya spagetti şeklinde doğramak.

3. **Vektörleştirme (Embedding Generation):**(3. Adım)
**Ne Yapılır?** Metin parçaları bir Bi-Encoder modeline (ör. `E5-Small`) sokulur ve her parça 384 veya 768 boyutlu sayı dizilerine (vektörlere) dönüştürülür.
**Makarna Karşılığı:** Doğranan malzemeleri türüne göre (sebze, baharat, et) etiketleyip raflara dizecek formata getirmek.

4. **İndeksleme ve Depolama (Vector DB Storage):**(4. Adım)
**Ne Yapılır?** Elde edilen vektörler ve ham metinler bir Vektör Veritabanına (Qdrant, Milvus, pgvector) HNSW grafik algoritmalarıyla kaydedilir.
**Makarna Karşılığı:** Hazırlanan malzemeleri ileride saniyeler içinde bulabilmek için düzenli bir kiler sistemine yerleştirmek.

---

## 2. FAZ: Arama ve Cevap Üretimi (Online Retrieval & Generation Pipeline)

*Müşteri siparişi verdi (kullanıcı soruyu sordu); şimdi yemeği pişirip sunma zamanı.*

1. **Sorgu İşleme (Query Transformation):**(5. Adım)
**Ne Yapılır?** Kullanıcının sorduğu ham soru temizlenir veya daha iyi arama yapabilmesi için LLM ile yeniden yazılır (Query Rewriting).
**Makarna Karşılığı:** Garsonun müşterinin "Acılı bir şeyler istiyorum" talebini mutfağa "1 Porsiyon Penne Arrabbiata" olarak iletmesi.

2. **Karma Arama (Hybrid Retrieval):**(6. Adım)
**Ne Yapılır?** Sorunun hem vektörü (Dense) hem de içindeki spesifik kelimeler (Sparse - BM25) kilerde aranır. En yakın 20-30 metin parçası çekilir.
**Makarna Karşılığı:** Aşçının kilerdeki raflardan siparişle en alakalı 20 malzemeyi tezgaha çıkarması.

3. **Yeniden Sıralama (Re-ranking):**(7. Adım)
**Ne Yapılır?** Çekilen 20 parça, Cross-Encoder (Reranker) modelinden geçirilerek soruyla en alakalı ilk 3-5 parçaya düşürülür.
**Makarna Karşılığı:** Tezgaha çıkarılan 20 malzemeden sadece taze ve yemeğe lezzet katacak en iyi 3 tanesini seçmek.

4. **Prompt Mühendisliği (Context Assembly):**(8. Adım)
**Ne Yapılır?** Seçilen 3 metin parçası, sistem talimatı (System Prompt) ve kullanıcının sorusu tek bir şablon içinde birleştirilir.
**Makarna Karşılığı:** Haşlanan makarna, hazırlanan sos ve baharatların tek bir tavada buluşturulması.

5. **Cevap Üretimi ve Akış (LLM Generation & Streaming):**(9. Adım)
**Ne Yapılır?** Hazırlanan nihai metin yerel LLM'e (Qwen, Llama) verilir ve gelen cevap kullanıcının ekranına kelime kelime (streaming) akıtılır.
**Makarna Karşılığı:** Yemeğin şık bir tabakla sıcak sıcak müşterinin masasına servis edilmesi.

6. **Değerlendirme ve İzleme (Evaluation & Observability):**(10. Adım)
**Ne Yapılır?** Cevap üretildikten sonra arka planda Ragas gibi sistemlerle "Cevap doğru mu? Uydurma (hallucination) var mı?" skorlaması yapılır.
**Makarna Karşılığı:** Müşterinin yemeği beğenip beğenmediğini kontrol edip tarifi iyileştirmek için nota almak.

---

### Bir "RAG Gurusu" Olmak İçin Akılda Tutulacak 3 Altın Kural

1. **Garbage In, Garbage Out:** 1. Adımda metni kötü temizler veya yanlış bölersen, dünyadaki en iyi LLM bile saçmalar.
2. **Retrieval is Everything:** Yanıt kalitesinin %80'i LLM'e değil, 6. ve 7. adımlarda doğru bilgiyi bulup getirmeye (Retrieval & Re-ranking) bağlıdır.
3. **Never Trust, Always Evaluate:** 10. adımı koymadığın bir RAG sistemi, karanlıkta araba kullanmaya benzer; nerede kaza yapacağını bilemezsin.
