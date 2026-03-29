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
    """CPU, RAM ve VRAM (GPU) kullanım oranlarını anlık olarak döndürür."""
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram_usage = psutil.virtual_memory().percent

    vram_info = "GPU Mevcut Değil"
    if torch.cuda.is_available():
        # VRAM kullanımını hesapla (MB cinsinden)
        vram_allocated = torch.cuda.memory_allocated() / (1024**2)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        vram_info = f"{vram_allocated:.2f}MB / {vram_total:.2f}MB (%{ (vram_allocated/vram_total)*100 :.1f})"

    return {
        "cpu_percent": f"%{cpu_usage}",
        "ram_percent": f"%{ram_usage}",
        "vram_usage": vram_info,
        "status": "Tehlikeli" if ram_usage > 90 or cpu_usage > 90 else "Normal",
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
