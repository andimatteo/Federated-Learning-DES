from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HostHardwareInfo:
    cpu_cores: int
    cpu_freq_ghz: float
    mem_total_gb: float
    mem_available_gb: float
    # Very rough aggregate compute capability in TFLOPs for accelerators.
    accelerator_tflops: float
    has_gpu: bool


@dataclass
class DeviceProfile:
    """
    Concrete per-device view combining hardware with network bandwidth.
    """

    hardware: HostHardwareInfo
    bandwidth_mean_mbps: float
    bandwidth_jitter: float


def _parse_cpuinfo() -> tuple[int, float]:
    """
    Parse /proc/cpuinfo to get logical cores and approx max frequency in GHz.
    """
    cores = os.cpu_count() or 1
    max_mhz = 0.0
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "cpu MHz" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        try:
                            mhz = float(parts[1])
                            if mhz > max_mhz:
                                max_mhz = mhz
                        except ValueError:
                            continue
    except OSError:
        pass
    if max_mhz <= 0:
        # Fallback: assume 2.5 GHz
        max_mhz = 2500.0
    return cores, max_mhz / 1000.0


def _parse_meminfo() -> tuple[float, float]:
    """
    Parse /proc/meminfo to get total and available memory in GB.
    """
    total_kb = 0.0
    avail_kb = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = float(line.split()[1])
    except OSError:
        pass

    total_gb = total_kb / (1024.0 * 1024.0) if total_kb > 0 else 0.0
    avail_gb = avail_kb / (1024.0 * 1024.0) if avail_kb > 0 else 0.0
    return total_gb, avail_gb


def _detect_gpu() -> tuple[bool, float]:
    """
    Very lightweight GPU detection.

    We avoid importing heavy frameworks. For now, we:
      - check NVIDIA via presence of /proc/driver/nvidia/version
      - if present, assign a conservative 5 TFLOPs

    This can be extended later (e.g. using nvidia-smi or ROCm).
    """
    has_nvidia = os.path.exists("/proc/driver/nvidia/version")
    if has_nvidia:
        return True, 5.0
    return False, 0.0


def collect_host_hardware() -> HostHardwareInfo:
    """
    Collect basic hardware info from the current Linux host.
    """
    cpu_cores, cpu_freq_ghz = _parse_cpuinfo()
    mem_total_gb, mem_available_gb = _parse_meminfo()
    has_gpu, accel_tflops = _detect_gpu()

    return HostHardwareInfo(
        cpu_cores=cpu_cores,
        cpu_freq_ghz=cpu_freq_ghz,
        mem_total_gb=mem_total_gb,
        mem_available_gb=mem_available_gb,
        accelerator_tflops=accel_tflops,
        has_gpu=has_gpu,
    )

