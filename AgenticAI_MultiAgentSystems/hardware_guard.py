import psutil
import torch  # GPU kontrolü için (PyTorch)


class HardwareGuard:
    def __init__(self, required_vram_gb=6, required_ram_gb=8):
        self.req_vram = required_vram_gb
        self.req_ram = required_ram_gb

    def check_capabilities(self):
        print("--- Donanım Kontrolü Başlatılıyor (System Audit) ---")

        # 1. GPU Kontrolü (VRAM Check)
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Tespit Edilen GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Kapasitesi: {vram_total:.2f} GB")

            if vram_total < self.req_vram:
                print("⚠️ UYARI: VRAM düşük, model yavaş çalışabilir (Offloading).")
            else:
                print("✅ GPU yeterli.")
        else:
            print("❌ GPU bulunamadı. Model CPU üzerinde çalışacak (Yavaş olabilir!).")

        # 2. RAM Kontrolü (System Memory Check)
        ram_total = psutil.virtual_memory().total / (1024**3)
        print(f"Toplam Sistem RAM: {ram_total:.2f} GB")

        if ram_total < self.req_ram:
            raise MemoryError(
                f"HATA: En az {self.req_ram}GB RAM gerekiyor. Mevcut: {ram_total:.2f}GB"
            )

        print("✅ Sistem belleği uygun.\n")


# Test:
if __name__ == "__main__":
    guard = HardwareGuard()
    guard.check_capabilities()
