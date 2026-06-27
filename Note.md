
# Flash Attention Nedir?
- Flash Attention, Transformer modellerinde bulunan "Attention" (Öz-dikkat) mekanizmasını çok daha hızlı ve bellek dostu hale getiren bir mühendislik çözümüdür.

- Standart Attention mekanizması, dizi uzunluğu arttıkça (örneğin 1000 kelimeden 10.000 kelimeye çıktığında) karesel olarak artan bir bellek ihtiyacı duyar. Flash Attention ise bu sorunu şu iki teknikle çözer:

1. Tiling (Döşeme): Veriyi küçük bloklara ayırarak GPU'nun çok hızlı olan ama kapasitesi düşük olan SRAM (yakın bellek) biriminde işler. Böylece yavaş olan ana belleğe (HBM) sürekli git-gel yapmaz.

2. Recomputation (Yeniden Hesaplama): Tüm devasa Attention matrisini bellekte saklamak yerine, ihtiyaç duyulduğunda bazı değerleri anlık olarak tekrar hesaplar. Bu kulağa yavaş gelse de, GPU'nun işlem hızı bellek hızından çok daha yüksek olduğu için toplamda büyük bir hız kazancı sağlar.


# 