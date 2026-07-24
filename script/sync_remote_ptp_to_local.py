#!/usr/bin/env python3
"""Synchronize a remote linuxptp device to the local host clock over SSH."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterable, List, Sequence


DEFAULT_REMOTE_IP = "192.168.2.240"
DEFAULT_REMOTE_USER = "ubuntu"
DEFAULT_REMOTE_PASSWORD = "ubuntu"
DEFAULT_LOCAL_IFACE = "enp3s0"
DEFAULT_LOCAL_IP_CIDR = "192.168.2.100/24"
DEFAULT_SAMPLE_COUNT = 6
DEFAULT_PASSES = 3
DEFAULT_SETTLE_S = 0.15


@dataclass
class OffsetSample:
    offset_s: float
    rtt_s: float


@dataclass
class OffsetSummary:
    samples: List[OffsetSample]
    mean_offset_s: float
    best_offset_s: float
    mean_half_rtt_s: float
    min_rtt_s: float
    avg_rtt_s: float
    max_rtt_s: float


class CommandError(RuntimeError):
    pass


def run_command(
    args: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(args),
        input=input_text,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        details = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise CommandError(f"{' '.join(shlex.quote(part) for part in args)} failed: {details}")
    return proc


def require_command(name: str) -> None:
    if run_command(["bash", "-lc", f"command -v {shlex.quote(name)}"], check=False).returncode != 0:
        raise CommandError(f"Required command not found: {name}")


def ping_host(ip: str) -> bool:
    return run_command(["ping", "-c", "1", "-W", "1", ip], check=False).returncode == 0


def list_ipv4_addresses(interface: str) -> List[str]:
    proc = run_command(["ip", "-o", "-4", "addr", "show", "dev", interface], check=False)
    if proc.returncode != 0:
        return []

    addresses: List[str] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if "inet" not in fields:
            continue
        inet_index = fields.index("inet")
        if inet_index + 1 < len(fields):
            addresses.append(fields[inet_index + 1])
    return addresses


def ensure_local_ipv4(interface: str, cidr: str) -> None:
    existing = list_ipv4_addresses(interface)
    if cidr in existing:
        print(f"[local] {interface} already has {cidr}")
        return

    require_command("nmcli")
    print(f"[local] adding {cidr} to {interface} via NetworkManager runtime config")
    run_command(["nmcli", "device", "modify", interface, "+ipv4.addresses", cidr])


class SshSession:
    def __init__(self, host: str, password: str) -> None:
        self.host = host
        self.password = password
        self.control_path = os.path.join(
            tempfile.gettempdir(),
            f"vio_eval_ptp_{os.getpid()}_{int(time.time() * 1000)}.sock",
        )
        self._open()

    def _common_args(self) -> List[str]:
        return [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ControlPath=" + self.control_path,
        ]

    def _open(self) -> None:
        require_command("ssh")
        require_command("sshpass")
        run_command(
            [
                "sshpass",
                "-p",
                self.password,
                "ssh",
                *self._common_args(),
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=60",
                self.host,
                "true",
            ]
        )
        check_proc = run_command(
            ["ssh", *self._common_args(), "-O", "check", self.host],
            check=False,
        )
        if check_proc.returncode != 0:
            details = check_proc.stderr.strip() or check_proc.stdout.strip() or "control connection check failed"
            raise CommandError(f"Failed to establish SSH control connection: {details}")

    def close(self) -> None:
        run_command(
            ["ssh", *self._common_args(), "-O", "exit", self.host],
            check=False,
        )
        try:
            os.remove(self.control_path)
        except FileNotFoundError:
            pass

    def run(self, remote_command: str, *, sudo_password: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        return run_command(
            ["ssh", *self._common_args(), "-o", "BatchMode=yes", self.host, remote_command],
            input_text=(sudo_password + "\n") if sudo_password is not None else None,
            check=check,
        )


def summarize_offsets(samples: Iterable[OffsetSample]) -> OffsetSummary:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("No offset samples collected")

    mean_offset = sum(sample.offset_s for sample in sample_list) / len(sample_list)
    best_offset = min(sample_list, key=lambda sample: abs(sample.offset_s)).offset_s
    rtts = [sample.rtt_s for sample in sample_list]
    mean_half_rtt = sum(rtts) / len(rtts) / 2.0
    return OffsetSummary(
        samples=sample_list,
        mean_offset_s=mean_offset,
        best_offset_s=best_offset,
        mean_half_rtt_s=mean_half_rtt,
        min_rtt_s=min(rtts),
        avg_rtt_s=sum(rtts) / len(rtts),
        max_rtt_s=max(rtts),
    )


def measure_offset(session: SshSession, sample_count: int, interval_s: float) -> OffsetSummary:
    samples: List[OffsetSample] = []
    remote_cmd = "python3 -c 'import time; print(\"%.9f\" % time.time())'"
    for sample_idx in range(sample_count):
        t0 = time.time()
        proc = session.run(remote_cmd)
        t1 = time.time()
        remote_ts = float(proc.stdout.strip())
        midpoint = (t0 + t1) / 2.0
        samples.append(OffsetSample(offset_s=remote_ts - midpoint, rtt_s=t1 - t0))
        if sample_idx + 1 < sample_count:
            time.sleep(interval_s)
    return summarize_offsets(samples)


def print_offset_summary(label: str, summary: OffsetSummary) -> None:
    print(
        f"{label}: best={summary.best_offset_s * 1000:+.3f} ms, "
        f"mean={summary.mean_offset_s * 1000:+.3f} ms, "
        f"RTT min/avg/max={summary.min_rtt_s * 1000:.3f}/"
        f"{summary.avg_rtt_s * 1000:.3f}/{summary.max_rtt_s * 1000:.3f} ms"
    )


def detect_remote_interface(session: SshSession, remote_ip: str) -> str:
    proc = session.run("ip -o -4 addr show")
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        if fields[3].split("/")[0] == remote_ip:
            return fields[1]
    raise RuntimeError(f"Could not find the remote interface that owns {remote_ip}")


def get_remote_phc_devices(session: SshSession, interface: str, sync_all_phc: bool) -> List[str]:
    if sync_all_phc:
        proc = session.run("ls /dev/ptp* 2>/dev/null", check=False)
        devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not devices:
            raise RuntimeError("Remote host has no /dev/ptp* devices")
        return devices

    proc = session.run(
        (
            f"find /sys/class/net/{shlex.quote(interface)}/device/ptp "
            "-maxdepth 1 -mindepth 1 -name 'ptp*' -printf '%f\n' 2>/dev/null | head -n 1"
        ),
        check=False,
    )
    name = proc.stdout.strip()
    if not name.startswith("ptp"):
        raise RuntimeError(f"Could not resolve PHC device for remote interface {interface}")
    return [f"/dev/{name}"]


def run_remote_sudo(session: SshSession, password: str, script: str) -> subprocess.CompletedProcess[str]:
    remote_cmd = f"sudo -S -p '' bash -lc {shlex.quote(script)}"
    return session.run(remote_cmd, sudo_password=password)


def read_remote_clock_info(session: SshSession, interface: str, phc_devices: Sequence[str], sudo_password: str) -> None:
    print(f"[remote] interface: {interface}")
    print(f"[remote] PHC devices: {', '.join(phc_devices)}")
    info_cmd = (
        "hostname; "
        "date -Ins; "
        f"ethtool -T {shlex.quote(interface)} 2>/dev/null | sed -n '1,24p'"
    )
    proc = session.run(info_cmd, check=False)
    for line in proc.stdout.strip().splitlines():
        print(f"  {line}")
    for device in phc_devices:
        cmp_proc = run_remote_sudo(session, sudo_password, f"phc_ctl {shlex.quote(device)} cmp")
        print(f"  {cmp_proc.stdout.strip()}")


def sync_once(
    session: SshSession,
    sudo_password: str,
    phc_devices: Sequence[str],
    sample_count: int,
    sample_interval_s: float,
    settle_s: float,
    compensation_s: float,
) -> tuple[OffsetSummary, OffsetSummary]:
    before = measure_offset(session, sample_count=sample_count, interval_s=sample_interval_s)
    target_epoch = time.time() + before.mean_half_rtt_s + compensation_s
    commands = [f"date -u -s '@{target_epoch:.9f}' >/dev/null"]
    commands.extend(f"phc_ctl {shlex.quote(device)} set >/dev/null" for device in phc_devices)
    run_remote_sudo(session, sudo_password, "; ".join(commands))
    time.sleep(settle_s)
    after = measure_offset(session, sample_count=sample_count, interval_s=sample_interval_s)
    return before, after


def verify_with_sdk(remote_ip: str, port: int, count: int, interval_s: float) -> None:
    script_path = Path(__file__).with_name("check_time_sync.py")
    if not script_path.exists():
        raise RuntimeError(f"Cannot find {script_path}")

    print("[verify] running SDK time check")
    proc = run_command(
        [
            sys.executable,
            str(script_path),
            "--ip",
            remote_ip,
            "--port",
            str(port),
            "--count",
            str(count),
            "--interval",
            str(interval_s),
        ]
    )
    print(proc.stdout.rstrip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize a remote linuxptp device to the local host clock over SSH."
    )
    parser.add_argument("--remote-ip", default=DEFAULT_REMOTE_IP, help=f"Remote device IP, default: {DEFAULT_REMOTE_IP}")
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER, help=f"Remote SSH user, default: {DEFAULT_REMOTE_USER}")
    parser.add_argument("--remote-password", default=DEFAULT_REMOTE_PASSWORD, help="Remote SSH password")
    parser.add_argument("--sudo-password", default=None, help="Remote sudo password, default: same as --remote-password")
    parser.add_argument("--remote-iface", default=None, help="Remote network interface to sync, default: auto-detect from --remote-ip")
    parser.add_argument("--sync-all-phc", action="store_true", help="Sync every remote /dev/ptp* instead of only the PHC behind --remote-iface")
    parser.add_argument("--local-iface", default=DEFAULT_LOCAL_IFACE, help=f"Local interface used when adding an address, default: {DEFAULT_LOCAL_IFACE}")
    parser.add_argument("--local-ip-cidr", default=None, help="Temporarily add this local IPv4/CIDR via NetworkManager if the remote host is unreachable")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT, help=f"Offset samples per pass, default: {DEFAULT_SAMPLE_COUNT}")
    parser.add_argument("--sample-interval", type=float, default=0.04, help="Seconds between offset samples, default: 0.04")
    parser.add_argument("--passes", type=int, default=DEFAULT_PASSES, help=f"How many correction passes to run, default: {DEFAULT_PASSES}")
    parser.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S, help=f"Wait time after each set command in seconds, default: {DEFAULT_SETTLE_S}")
    parser.add_argument("--initial-compensation-ms", type=float, default=0.0, help="Initial remote command latency compensation in milliseconds, default: 0")
    parser.add_argument("--target-offset-ms", type=float, default=10.0, help="Stop early once the post-pass mean offset is within this magnitude in milliseconds, default: 10")
    parser.add_argument("--port", type=int, default=8080, help="Controller port used by optional SDK verification, default: 8080")
    parser.add_argument("--verify-sdk-time", action="store_true", help="Run script/check_time_sync.py after the SSH/PTP sync")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without setting the remote clocks")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be > 0")
    if args.passes <= 0:
        raise ValueError("--passes must be > 0")
    if args.sample_interval < 0:
        raise ValueError("--sample-interval must be >= 0")
    if args.settle < 0:
        raise ValueError("--settle must be >= 0")
    if args.target_offset_ms < 0:
        raise ValueError("--target-offset-ms must be >= 0")


def main() -> int:
    args = parse_args()
    validate_args(args)

    sudo_password = args.sudo_password or args.remote_password
    host = f"{args.remote_user}@{args.remote_ip}"
    compensation_s = args.initial_compensation_ms / 1000.0

    require_command("ip")
    require_command("ping")

    if not ping_host(args.remote_ip):
        if args.local_ip_cidr is None:
            raise RuntimeError(
                f"{args.remote_ip} is unreachable. Re-run with "
                f"--local-iface {args.local_iface} --local-ip-cidr {DEFAULT_LOCAL_IP_CIDR} "
                "if you need the script to add a local subnet address first."
            )
        ensure_local_ipv4(args.local_iface, args.local_ip_cidr)
        if not ping_host(args.remote_ip):
            raise RuntimeError(f"{args.remote_ip} is still unreachable after local interface setup")

    session = SshSession(host=host, password=args.remote_password)
    try:
        remote_iface = args.remote_iface or detect_remote_interface(session, args.remote_ip)
        phc_devices = get_remote_phc_devices(session, remote_iface, args.sync_all_phc)

        print(f"[remote] target host: {host}")
        read_remote_clock_info(session, remote_iface, phc_devices, sudo_password)

        if args.dry_run:
            print("[dry-run] no remote clocks were changed")
            return 0

        for pass_idx in range(1, args.passes + 1):
            print(f"\n[pass {pass_idx}/{args.passes}]")
            before, after = sync_once(
                session=session,
                sudo_password=sudo_password,
                phc_devices=phc_devices,
                sample_count=args.sample_count,
                sample_interval_s=args.sample_interval,
                settle_s=args.settle,
                compensation_s=compensation_s,
            )
            print_offset_summary("  before", before)
            print_offset_summary("  after ", after)
            compensation_s -= after.mean_offset_s
            if abs(after.mean_offset_s) * 1000.0 <= args.target_offset_ms:
                print(
                    f"  stopping early: |mean offset| <= {args.target_offset_ms:.3f} ms"
                )
                break

        print("\n[final PHC vs CLOCK_REALTIME]")
        for device in phc_devices:
            cmp_proc = run_remote_sudo(session, sudo_password, f"phc_ctl {shlex.quote(device)} cmp")
            print(f"  {cmp_proc.stdout.strip()}")

    finally:
        session.close()

    if args.verify_sdk_time:
        print()
        verify_with_sdk(args.remote_ip, args.port, count=3, interval_s=0.3)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        raise SystemExit(main())
    except (CommandError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
