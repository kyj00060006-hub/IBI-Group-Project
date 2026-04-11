"""Improved animated 30-bed ward simulation with random event timing, same-day discharge rules,
and 30-minute bed cooldown after discharge.

Reads admissions/discharges from ward_analysis2.py via load_ward_data.py, so you only edit one file.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

from load_data import get_dataset

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Patient:
    patient_id: int
    admit_abs_minute: int
    planned_same_day_discharge: bool = False
    bed_id: Optional[int] = None
    active: bool = True


@dataclass
class Bed:
    bed_id: int
    occupant_id: Optional[int] = None
    cooldown_until: int = -1


@dataclass
class FrameState:
    day: int
    time_label: str
    admissions_total: int
    discharges_total: int
    admissions_done: int
    discharges_done: int
    occupancy: int
    overflow: int
    occupied_visible: int
    bed_statuses: List[str]  # occupied | empty | cooldown
    event_label: str
    seed_label: str


@dataclass
class FrameBundle:
    frames: List[FrameState]
    day_end_occupancy: List[int]


DAY_MINUTES = 24 * 60


def _format_time(minutes_from_midnight: int) -> str:
    hour = (minutes_from_midnight // 60) % 24
    minute = minutes_from_midnight % 60
    return f"{hour:02d}:{minute:02d}"


def _abs_to_day_and_clock(abs_minute: int) -> Tuple[int, int]:
    day = abs_minute // DAY_MINUTES + 1
    minute_of_day = abs_minute % DAY_MINUTES
    return day, minute_of_day


def _sample_unique_times(count: int, start_min: int, end_min: int, rng: np.random.Generator) -> List[int]:
    if count <= 0:
        return []
    if end_min < start_min:
        raise ValueError(f"Invalid time range: {start_min}..{end_min}")
    population = end_min - start_min + 1
    if count <= population:
        chosen = rng.choice(np.arange(start_min, end_min + 1), size=count, replace=False)
        return sorted(int(x) for x in chosen)
    # Extremely unlikely here, but keep a fallback.
    chosen = rng.integers(start_min, end_min + 1, size=count)
    return sorted(int(x) for x in chosen)


def _parse_hhmm(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    h, m = int(hh), int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time: {hhmm}")
    return h * 60 + m


def simulate_animation_frames(
    admissions: List[int],
    discharges: List[int],
    capacity: int = 30,
    day_start: str = "00:00",
    day_end: str = "23:59",
    hold_frames: int = 2,
    min_stay_same_day_minutes: int = 120,
    bed_cooldown_minutes: int = 30,
    seed: Optional[int] = 42,
) -> FrameBundle:
    """
    Simulate a 30-bed ward with richer timing rules.

    Key rules:
    - Event times are random (optionally reproducible with a seed).
    - For patients admitted and discharged on the same day, discharge happens at least
      `min_stay_same_day_minutes` after admission.
    - After a patient leaves a bed, that bed cannot take a new patient until
      `bed_cooldown_minutes` later.
    - When no eligible bed is available, patients go to an overflow/waiting queue.
      The overflow count shown in the plot is the number of active patients without a visible bed.
    - If a bed cooldown ends and patients are waiting, the earliest waiting patient is assigned first.
    """
    if len(admissions) != 7 or len(discharges) != 7:
        raise ValueError("Admissions and discharges must each contain 7 values.")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if hold_frames < 1:
        raise ValueError("hold_frames must be at least 1")
    if min_stay_same_day_minutes < 0 or bed_cooldown_minutes < 0:
        raise ValueError("Minute settings must be non-negative")

    day_start_min = _parse_hhmm(day_start)
    day_end_min = _parse_hhmm(day_end)
    if day_end_min <= day_start_min:
        raise ValueError("day_end must be later than day_start within the same day")

    rng = np.random.default_rng(seed)
    seed_label = f"seed={seed}" if seed is not None else "seed=random"

    beds: List[Bed] = [Bed(bed_id=i) for i in range(capacity)]
    patients: Dict[int, Patient] = {}
    waiting_queue: Deque[int] = deque()
    active_patient_order: Deque[int] = deque()  # oldest active patients first
    next_patient_id = 1

    frames: List[FrameState] = []
    day_end_occupancy: List[int] = []

    def current_occupancy() -> int:
        return sum(1 for p in patients.values() if p.active)

    def visible_occupied_count() -> int:
        return sum(1 for bed in beds if bed.occupant_id is not None)

    def overflow_count() -> int:
        return current_occupancy() - visible_occupied_count()

    def bed_statuses(now_abs: int) -> List[str]:
        statuses = []
        for bed in beds:
            if bed.occupant_id is not None:
                statuses.append("occupied")
            elif bed.cooldown_until > now_abs:
                statuses.append("cooldown")
            else:
                statuses.append("empty")
        return statuses

    def make_frame(day: int, abs_minute: int, adm_total: int, dis_total: int,
                   adm_done: int, dis_done: int, event_label: str) -> FrameState:
        _, minute_of_day = _abs_to_day_and_clock(abs_minute)
        return FrameState(
            day=day,
            time_label=_format_time(minute_of_day),
            admissions_total=adm_total,
            discharges_total=dis_total,
            admissions_done=adm_done,
            discharges_done=dis_done,
            occupancy=current_occupancy(),
            overflow=overflow_count(),
            occupied_visible=visible_occupied_count(),
            bed_statuses=bed_statuses(abs_minute),
            event_label=event_label,
            seed_label=seed_label,
        )

    def first_available_bed(now_abs: int) -> Optional[int]:
        for bed in beds:
            if bed.occupant_id is None and bed.cooldown_until <= now_abs:
                return bed.bed_id
        return None

    def assign_waiting_to_beds(now_abs: int, day: int, adm_total: int, dis_total: int,
                               adm_done: int, dis_done: int) -> List[FrameState]:
        created: List[FrameState] = []
        moved = 0
        while waiting_queue:
            bed_id = first_available_bed(now_abs)
            if bed_id is None:
                break
            patient_id = waiting_queue.popleft()
            patient = patients[patient_id]
            if not patient.active:
                continue
            beds[bed_id].occupant_id = patient_id
            patient.bed_id = bed_id
            moved += 1
            created.append(
                make_frame(
                    day, now_abs, adm_total, dis_total, adm_done, dis_done,
                    f"等待患者入住床位 {bed_id + 1}（补位 {moved} 人）",
                )
            )
        return created

    def earliest_next_bed_ready(now_abs: int) -> Optional[int]:
        if not waiting_queue:
            return None
        candidates = [bed.cooldown_until for bed in beds if bed.occupant_id is None and bed.cooldown_until > now_abs]
        if not candidates:
            return None
        return min(candidates)

    for day_idx, (adm, dis) in enumerate(zip(admissions, discharges), start=1):
        if adm < 0 or dis < 0:
            raise ValueError("Admissions and discharges cannot be negative.")

        day_abs_start = (day_idx - 1) * DAY_MINUTES + day_start_min
        day_abs_end = (day_idx - 1) * DAY_MINUTES + day_end_min

        # Overnight, if any waiting patients can now enter beds, place them at day start.
        assign_frames = assign_waiting_to_beds(
            day_abs_start, day_idx, adm, dis, 0, 0
        )
        start_label = "新的一天开始"
        if assign_frames:
            start_label += "（已处理可入住的等待患者）"
        start_frame = make_frame(day_idx, day_abs_start, adm, dis, 0, 0, start_label)
        frames.extend([start_frame] * hold_frames)
        frames.extend(assign_frames)

        # Decide how many of today's discharges will come from same-day admissions.
        carryover_ids = [pid for pid in list(active_patient_order) if patients.get(pid) and patients[pid].active]
        min_same_day = max(0, dis - len(carryover_ids))
        max_same_day = min(adm, dis)
        if day_abs_start + min_stay_same_day_minutes > day_abs_end:
            max_same_day = 0
            min_same_day = 0
        if max_same_day < min_same_day:
            raise ValueError(
                f"Day {day_idx}: not enough existing patients to satisfy discharge count under current timing rules."
            )
        if max_same_day == min_same_day:
            same_day_discharge_n = max_same_day
        else:
            same_day_discharge_n = int(rng.integers(min_same_day, max_same_day + 1))
        carryover_discharge_n = dis - same_day_discharge_n

        # Pick which existing patients will be discharged today: oldest active first.
        carryover_discharge_ids = carryover_ids[:carryover_discharge_n]

        # Preallocate patient IDs for today's admissions.
        new_ids = list(range(next_patient_id, next_patient_id + adm))
        next_patient_id += adm
        same_day_ids = set(new_ids[:same_day_discharge_n])

        latest_same_day_admit = day_abs_end - min_stay_same_day_minutes
        early_admit_n = same_day_discharge_n
        late_admit_n = adm - early_admit_n

        early_admit_times = _sample_unique_times(early_admit_n, day_abs_start, latest_same_day_admit, rng)
        late_admit_times = _sample_unique_times(late_admit_n, day_abs_start, day_abs_end, rng)

        admissions_plan: List[Tuple[int, int]] = []
        for pid, t in zip(new_ids[:early_admit_n], early_admit_times):
            admissions_plan.append((pid, t))
        for pid, t in zip(new_ids[early_admit_n:], late_admit_times):
            admissions_plan.append((pid, t))
        admissions_plan.sort(key=lambda x: (x[1], x[0]))

        # Schedule carryover discharges.
        carryover_discharge_times = _sample_unique_times(carryover_discharge_n, day_abs_start, day_abs_end, rng)

        base_events: List[Tuple[int, int, str, int]] = []
        # priority order at same minute: discharge first, then admission. (Rare anyway.)
        for pid, t in zip(carryover_discharge_ids, carryover_discharge_times):
            base_events.append((t, 0, "discharge", pid))

        for pid, t in admissions_plan:
            base_events.append((t, 1, "admission", pid))
            if pid in same_day_ids:
                discharge_t = int(rng.integers(t + min_stay_same_day_minutes, day_abs_end + 1))
                base_events.append((discharge_t, 0, "discharge", pid))

        base_events.sort(key=lambda x: (x[0], x[1], x[3]))

        admissions_done = 0
        discharges_done = 0
        event_idx = 0
        now_abs = day_abs_start

        while True:
            next_base_time = base_events[event_idx][0] if event_idx < len(base_events) else None
            next_auto_time = earliest_next_bed_ready(now_abs)

            if next_base_time is None and next_auto_time is None:
                break

            if next_auto_time is not None and (next_base_time is None or next_auto_time < next_base_time):
                now_abs = next_auto_time
                auto_frames = assign_waiting_to_beds(
                    now_abs, day_idx, adm, dis, admissions_done, discharges_done
                )
                frames.extend(auto_frames)
                continue

            # Process one or more base events at this time.
            now_abs = next_base_time  # type: ignore[assignment]
            same_time_events: List[Tuple[int, int, str, int]] = []
            while event_idx < len(base_events) and base_events[event_idx][0] == now_abs:
                same_time_events.append(base_events[event_idx])
                event_idx += 1

            for _, _, event_type, patient_id in same_time_events:
                if event_type == "admission":
                    patient = Patient(
                        patient_id=patient_id,
                        admit_abs_minute=now_abs,
                        planned_same_day_discharge=(patient_id in same_day_ids),
                    )
                    patients[patient_id] = patient
                    active_patient_order.append(patient_id)
                    admissions_done += 1

                    bed_id = first_available_bed(now_abs)
                    if bed_id is not None:
                        beds[bed_id].occupant_id = patient_id
                        patient.bed_id = bed_id
                        label = f"入院 +1 → 入住床位 {bed_id + 1}（今日已入院 {admissions_done}/{adm}）"
                    else:
                        waiting_queue.append(patient_id)
                        label = f"入院 +1 → 暂无可用床位，进入等待/爆满队列（{len(waiting_queue)}）"
                    frames.append(make_frame(day_idx, now_abs, adm, dis, admissions_done, discharges_done, label))

                    # If some beds are already available at this exact moment, let waiting patients fill them.
                    frames.extend(assign_waiting_to_beds(now_abs, day_idx, adm, dis, admissions_done, discharges_done))

                else:  # discharge
                    patient = patients.get(patient_id)
                    if patient is None or not patient.active:
                        continue

                    discharges_done += 1
                    patient.active = False
                    # Remove from active order.
                    try:
                        active_patient_order.remove(patient_id)
                    except ValueError:
                        pass

                    if patient.bed_id is not None:
                        bed_id = patient.bed_id
                        beds[bed_id].occupant_id = None
                        beds[bed_id].cooldown_until = now_abs + bed_cooldown_minutes
                        patient.bed_id = None
                        label = f"出院 -1 ← 床位 {bed_id + 1} 进入冷却 {bed_cooldown_minutes} 分钟"
                    else:
                        # Waiting/overflow patient discharged before getting a visible bed.
                        try:
                            waiting_queue.remove(patient_id)
                        except ValueError:
                            pass
                        label = "出院 -1 ← 等待/爆满患者离院"
                    frames.append(make_frame(day_idx, now_abs, adm, dis, admissions_done, discharges_done, label))

                    # Bed cannot be reused immediately; only auto-assign after cooldown is over.

        end_frame = make_frame(day_idx, day_abs_end, adm, dis, admissions_done, discharges_done, "当天结束")
        frames.extend([end_frame] * hold_frames)
        day_end_occupancy.append(current_occupancy())

    return FrameBundle(frames=frames, day_end_occupancy=day_end_occupancy)


def _draw_single_ward(ax, state: FrameState, capacity: int = 30, columns: int = 6) -> None:
    rows = (capacity + columns - 1) // columns
    bed_w, bed_h = 1.15, 0.72
    gap_x, gap_y = 0.28, 0.28

    for idx in range(capacity):
        row = idx // columns
        col = idx % columns
        x = col * (bed_w + gap_x)
        y = (rows - 1 - row) * (bed_h + gap_y)

        status = state.bed_statuses[idx]
        if status == "occupied":
            light_color = "#E74C3C"
            bed_face = "#FFF5F5"
        elif status == "cooldown":
            light_color = "#F1C40F"
            bed_face = "#FFFBEA"
        else:
            light_color = "#2ECC71"
            bed_face = "#F4FFF5"

        bed = FancyBboxPatch(
            (x, y),
            bed_w, bed_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#555555",
            facecolor=bed_face,
        )
        ax.add_patch(bed)

        pillow = FancyBboxPatch(
            (x + 0.08, y + bed_h - 0.18),
            0.26, 0.12,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            linewidth=0.6,
            edgecolor="#777777",
            facecolor="#EDEDED",
        )
        ax.add_patch(pillow)

        light = Circle((x + bed_w - 0.16, y + bed_h - 0.12), 0.08, color=light_color, ec="#333333", lw=0.5)
        ax.add_patch(light)
        ax.text(x + bed_w / 2, y + 0.2, f"{idx + 1}", ha="center", va="center", fontsize=8)

    total_w = columns * bed_w + (columns - 1) * gap_x
    total_h = rows * bed_h + (rows - 1) * gap_y
    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-0.4, total_h + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(f"第{state.day}天 {state.time_label}", fontsize=14, fontweight="bold", pad=10)

    left_info = (
        f"今日入院: {state.admissions_total}   已发生: {state.admissions_done}\n"
        f"今日出院: {state.discharges_total}   已发生: {state.discharges_done}\n"
        f"当前在院: {state.occupancy}   可见占床: {state.occupied_visible}"
    )
    ax.text(
        0.02, 0.98, left_info,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB")
    )

    overflow_color = "#C0392B" if state.overflow > 0 else "#2C3E50"
    ax.text(
        0.98, 0.98,
        f"爆满/等待人数: {state.overflow}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10, color=overflow_color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB")
    )

    ax.text(
        0.5, -0.05,
        state.event_label,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F8F9FA", edgecolor="#CCCCCC")
    )

    legend = "红=有人  绿=空床  黄=冷却中(30min内不可入住)"
    ax.text(
        0.5, -0.12,
        legend,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=9, color="#444444"
    )

    ax.text(
        0.98, -0.12,
        state.seed_label,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="#666666"
    )


def animate_ward(
    frames: List[FrameState],
    capacity: int = 30,
    interval_ms: int = 500,
    save_path: str | Path | None = None,
    fps: int = 3,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.2))

    def update(frame_index: int):
        ax.clear()
        _draw_single_ward(ax, frames[frame_index], capacity=capacity)
        fig.suptitle("30床位病房动态模拟（随机时序版）", fontsize=16, fontweight="bold", y=0.98)
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path = str(save_path)
        suffix = Path(save_path).suffix.lower()
        if suffix == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif suffix == ".mp4":
            try:
                anim.save(save_path, fps=fps)
            except Exception as exc:
                raise RuntimeError(
                    "Saving MP4 usually requires ffmpeg. Try saving as .gif instead."
                ) from exc
        else:
            raise ValueError("save_path must end with .gif or .mp4")
        print(f"Saved animation: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved animated 30-bed ward simulation")
    parser.add_argument("--source-file", default="ward_analysis2.py", help="Python file containing the original demo data")
    parser.add_argument("--dataset", choices=["pre", "post"], default="pre", help="Choose pre- or post-vaccine data")
    parser.add_argument("--capacity", type=int, default=30, help="Maximum number of beds in the ward")
    parser.add_argument("--day-start", default="00:00", help="Displayed start time for each day")
    parser.add_argument("--day-end", default="23:59", help="Displayed end time for each day")
    parser.add_argument("--interval", type=int, default=500, help="Milliseconds between frames when previewing")
    parser.add_argument("--hold-frames", type=int, default=2, help="How many repeated frames to hold at each day start/end")
    parser.add_argument("--fps", type=int, default=3, help="Export frames per second for GIF/MP4")
    parser.add_argument("--min-stay", type=int, default=120, help="Minimum minutes before a same-day discharge can happen")
    parser.add_argument("--bed-cooldown", type=int, default=30, help="Minutes a bed stays unavailable after discharge")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible event timing; change it for a different schedule")
    parser.add_argument("--save", default=None, help="Optional output animation path, e.g. ward_animation_v2.gif")
    args = parser.parse_args()

    dataset = get_dataset(dataset=args.dataset, source_file=args.source_file)
    bundle = simulate_animation_frames(
        admissions=dataset["admissions"],
        discharges=dataset["discharges"],
        capacity=args.capacity,
        day_start=args.day_start,
        day_end=args.day_end,
        hold_frames=args.hold_frames,
        min_stay_same_day_minutes=args.min_stay,
        bed_cooldown_minutes=args.bed_cooldown,
        seed=args.seed,
    )

    print("Day-end occupancy:", bundle.day_end_occupancy)
    animate_ward(
        frames=bundle.frames,
        capacity=args.capacity,
        interval_ms=args.interval,
        save_path=args.save,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
