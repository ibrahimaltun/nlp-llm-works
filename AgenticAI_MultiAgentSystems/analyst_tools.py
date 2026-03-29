import time
import psutil
import torch
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize search tool globally or inside a class for reuse
search_api = DuckDuckGoSearchRun()


@tool
def get_detailed_metrics():
    """Reports CPU, RAM, and VRAM details of the system with technical thresholds."""
    try:
        # Get CPU usage percentage over a 0.5-second interval
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()

        # GPU Analysis - Checks if CUDA is available for VRAM metrics
        gpu_data = "N/A"
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**2
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_data = {
                "usage_mb": round(vram_used, 2),
                "total_mb": round(vram_total, 2),
                "load_percent": round((vram_used / vram_total) * 100, 1),
            }

        return {
            "cpu_status": {"value": f"%{cpu}", "critical": cpu > 90},
            "ram_status": {
                "percent": ram.percent,
                "available_gb": round(ram.available / 1024**3, 2),
                "critical": ram.percent > 85,
            },
            "gpu_status": gpu_data,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    except Exception as e:
        return f"Could not retrieve resource data: {str(e)}"


@tool
def log_anomaly_detector(log_text: str):
    """Categorizes errors within log text and determines a risk score."""
    if not log_text or len(log_text) < 5:
        return "Insufficient log data for analysis."

    log_lower = log_text.lower()
    score = 0
    findings = []

    # Risk Analysis Engine - Pattern matching for severity
    checks = {
        "critical": ["panic", "segfault", "critical"],
        "error": ["error", "failed", "unable to"],
        "warning": ["warning", "latency", "slow"],
    }

    if any(word in log_lower for word in checks["critical"]):
        score = 10
        findings.append("CRITICAL SYSTEM ERROR")
    elif any(word in log_lower for word in checks["error"]):
        score = 7
        findings.append("OPERATIONAL ERROR")

    return {
        "risk_score": score,
        "detected_issues": findings,
        "suggestion": (
            "Immediate intervention required" if score > 8 else "Monitor closely"
        ),
    }


@tool
def search_solution_online(error_message: str):
    """Searches for error messages online and provides technical solution suggestions."""
    # Using specific sites to improve search quality for technical issues
    query = f"site:stackoverflow.com OR site:github.com {error_message} solution"
    print(f"🌐 [Web Search] Searching for: '{error_message}'")
    try:
        return search_api.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"


# List of defined tools for the agent
tools = [get_detailed_metrics, log_anomaly_detector, search_solution_online]
