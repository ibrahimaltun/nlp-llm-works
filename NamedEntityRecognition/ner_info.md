
# NER - Adlandırılmış Varlık Tanıma

NER, yapılandırılmamış metinler içerisindeki özel isimleri (kişi, yer, organizasyon vb.) otomatik olarak tespit edip önceden tanımlanmış kategorilere ayıran bir NLP tekniğidir.

NER, metindeki ham veriyi anlamlı ve yapılandırılmış bilgiye dönüştürür. Yaygın olarak kullanılan bazı standart kategoriler şunlardır:

- Kişi (PER): Albert Einstein, Nelson Mandela.
- Organizasyon (ORG): Google, Birleşmiş Milletler.
- Konum (LOC/GPE): İstanbul, Ağrı Dağı, Türkiye.
- Zaman (DATE/TIME): 5 Mayıs 2025, saat 15:00.
- Sayısal Veriler: %50 (Yüzde), $100 (Para birimi), 10 kg (Miktar).

## Kullanılan Yöntemler

-> Kural Tabanlı (Rule-based): Düzenli ifadeler (regex) ve sözlükler kullanılır. Belirli kalıplar için yüksek doğruluk sunar ancak esnek değildir.

-> İstatistiksel / Makine Öğrenmesi: CRF (Conditional Random Fields) ve HMM (Hidden Markov Models) gibi algoritmalarla bağlam analizi yapılır.

-> Derin Öğrenme: Günümüzde en popüler yöntemdir. BERT, RoBERTa gibi Transformer modelleri ve LSTM ağları ile karmaşık metinlerde yüksek başarı elde edilir. ￼

## Neden spaCy?

-> Hazır Modeller: Hiç veri toplamanıza gerek kalmadan milyonlarca kelimeyle eğitilmiş modelleri kullanırsınız.
-> Hız: C dilinde optimize edildiği için büyük metin yığınlarını saniyeler içinde tarar.
-> Görselleştirme: displacy modülü sayesinde varlıkları metin üzerinde renkli etiketlerle kolayca görselleştirebilirsiniz.

### spacy yükleme adımları

- Pip ve Setuptools Güncelleyin (En Yaygın Çözüm): python -m pip install --upgrade pip setuptools wheel
- "Build Dependencies" Kısmını Atlayın:      pip install spacy --no-build-isolation
- Hazır Derlenmiş (Binary) Sürümü Zorlayın:  pip install spacy --only-binary=:all:
- Alternatif: Conda Kullanım:                conda install -c conda-forge spacy

### Model indirme

- python -m spacy download tr_core_news_lg

=> Bu adım olmazsa aşağıdaki adresi tarayıcıya yaz elle indir

- <https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl>

=> inen dizinde aşağıdaki komutu çalıştır.

- $ pip install tr_core_news_lg-1.0-py3-none-any.whl
