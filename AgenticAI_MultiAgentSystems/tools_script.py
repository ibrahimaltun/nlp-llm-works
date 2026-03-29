""" """

import time
import psutil
import torch
from langchain_core.tools import tool


@tool
def __multiply(a: int, b: int) -> int:
    "ik isayıyı çarpar"
    return a * b


@tool
def __get_system_resource_usage():
    """Donanım kaynaklarının detaylı dökümünü yapar."""
    cpu_usage = psutil.cpu_percent(interval=0.7)
    ram = psutil.virtual_memory()

    vram_report = "GPU Tespit Edilemedi"
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024**2)
        vram_reserved = torch.cuda.memory_reserved() / (1024**2)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        vram_report = f"Kullanılan: {vram_allocated:.1f}MB, Ayrılan: {vram_reserved:.1f}MB, Toplam: {vram_total:.1f}MB"

    return {
        "CPU": f"%{cpu_usage}",
        "RAM_DETAY": f"Kullanılan: {ram.used / (1024**3):.2f}GB, Boş: {ram.available / (1024**3):.2f}GB, Toplam: {ram.total / (1024**3):.2f}GB (Doluluk: %{ram.percent})",
        "VRAM_DETAY": vram_report,
        "GENEL_SİSTEM_NOTU": "KRİTİK" if ram.percent > 85 else "STABİL",
    }


@tool
def __analyze_log_patterns(log_content: str):
    """
    Verilen log metni içinde anomali, hata veya şüpheli paternleri tarar.
    Örnek giriş: 'Connection refused', 'Memory leak detected', 'High latency'
    """
    time.sleep(1)  # İşlem simülasyonu
    log_content = log_content.lower()

    # Basit bir patern eşleşme motoru (Anomaly Detection başlangıcı)
    anomalies = []
    if "error" in log_content or "fail" in log_content:
        anomalies.append("❌ Kritik Hata Kaydı")
    if "memory" in log_content or "leak" in log_content:
        anomalies.append("⚠️ Olası Bellek Sızıntısı (Memory Leak)")
    if "latency" in log_content or "slow" in log_content:
        anomalies.append("🐢 Gecikme Sorunu (High Latency)")
    if "unauthorized" in log_content or "brute" in log_content:
        anomalies.append("🛡️ Güvenlik İhlali Şüphesi")

    if not anomalies:
        return "✅ Temiz: Loglarda bilinen bir anomali paternine rastlanmadı."

    return f"Bulunan Bulgular: {', '.join(anomalies)}"


# adım 1: tool'ların tanımlanması
tools = [__multiply, __get_system_resource_usage, __analyze_log_patterns]
