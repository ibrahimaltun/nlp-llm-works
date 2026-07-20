# Sıfırdan Vektör Arama Motoru

## 1.Metni Elinizle Parçalayın (Manual Chunking)

### Adım 1

- PDF'in içindeki ham metni doğrudan yapay zekaya verirsek model bunu sindiremez. Metni belirli karakter büyüklüklerinde (örneğin 500 karakter) küçük parçalara (chunk) bölmemiz gerekir. Cümlelerin ortadan ikiye bölünüp anlamını kaybetmemesi için de parçaları birbiri üzerine biraz bindiririz (overlap).

## 2.Vektörleri Doğrudan Çıkarın

### Adım 2

- sentence-transformers kütüphanesini kullanarak Türkçe destekli hafif bir modeli (multilingual-e5-small) doğrudan çağırıyoruz. Bu model, her bir metin parçası için bize 384 adet sayıdan oluşan uzun bir liste (vektör) verecek.

## 3.Kosinüs Benzerliği (Cosine Similarity) ile Arama Yapın

### Adım 3

- İşin en zevkli kısmı burası. Kullanıcı bir soru sorduğunda, sorunun da vektörünü alıyoruz. Ardından NumPy kullanarak, sorunun vektörü ile elimizdeki tüm metin parçalarının vektörleri arasındaki "açıyı" hesaplıyoruz. İki ok birbirine ne kadar yakın bakıyorsa, anlamları o kadar benzerdir.
