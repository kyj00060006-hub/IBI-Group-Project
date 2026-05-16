from __future__ import annotations

import math
import os
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DAY_MINUTES = 24 * 60
WARD_CAPACITY = 30
DEFAULT_WARD_A_ADMISSIONS = [5, 8, 12, 10, 7, 6, 4]
DEFAULT_WARD_A_DISCHARGES = [2, 3, 4, 6, 8, 5, 3]
DEFAULT_WARD_B_ADMISSIONS = [8, 6, 12, 18, 20, 5, 7]
DEFAULT_WARD_B_DISCHARGES = [4, 5, 3, 16, 14, 1, 3]


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
class WardState:
    ward_name: str
    abs_minute: int
    day: int
    time_label: str
    admissions_total: int
    discharges_total: int
    admissions_done: int
    discharges_done: int
    occupancy: int
    overflow: int
    occupied_visible: int
    bed_statuses: List[str]
    event_label: str
    avg_cooldown_minutes: float


@dataclass
class WardSimulationResult:
    ward_name: str
    history: List[WardState]
    day_end_occupancy: List[int]
    sampled_cooldowns: List[int]


@dataclass
class CombinedFrame:
    abs_minute: int
    day: int
    time_label: str
    state_a: WardState
    state_b: WardState
    show_summary: bool = False


@dataclass
class AnalysisSummary:
    task5: Dict[str, object]
    recovery: Dict[str, object]
    mle_used: bool
    mle_result: Optional[Dict[str, object]]
    mle_reason: str


def 格式化时间(分钟: int) -> str:
    小时 = (分钟 // 60) % 24
    分钟值 = 分钟 % 60
    return f"{小时:02d}:{分钟值:02d}"


def 解析时间(hhmm: str) -> int:
    小时, 分钟 = hhmm.split(":")
    h = int(小时)
    m = int(分钟)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time: {hhmm}")
    return h * 60 + m


def 解析7天整数列表(文本: str, 默认值: List[int]) -> List[int]:
    文本 = 文本.strip()
    if not 文本:
        return list(默认值)
    值 = [x.strip() for x in 文本.split(",") if x.strip()]
    if len(值) != 7:
        raise ValueError("Please enter exactly 7 integers separated by commas.")
    结果 = [int(x) for x in 值]
    if any(x < 0 for x in 结果):
        raise ValueError("Counts cannot be negative.")
    return 结果


def 读取浮点输入(提示: str, 默认值: float) -> float:
    文本 = input(f"{提示} (press Enter to use default {默认值}): ").strip()
    if not 文本:
        return 默认值
    数值 = float(文本)
    if 数值 < 0:
        raise ValueError("Average cooldown time cannot be negative.")
    return 数值


def 读取字符串输入(提示: str, 默认值: str) -> str:
    文本 = input(f"{提示} (press Enter to use default {默认值}): ").strip()
    return 文本 if 文本 else 默认值


def 计算占床人数(admissions: List[int], discharges: List[int]) -> List[int]:
    current = 0
    daily_count = []
    for a, d in zip(admissions, discharges):
        current += a - d
        daily_count.append(current)
    return daily_count


# ===== 来自 ward_analysis2.py 的 Task 5：R 值近似 =====
def 估计R值(admissions: List[int]) -> Dict[str, object]:
    r_values: List[Optional[float]] = []
    labels: List[str] = []

    for i in range(1, len(admissions)):
        if admissions[i - 1] == 0:
            r = None
        else:
            r = round(admissions[i] / admissions[i - 1], 3)
        r_values.append(r)
        labels.append(f"Day {i + 1}")

    valid_r = [r for r in r_values if r is not None]
    mean_r = round(float(np.mean(valid_r)), 3) if valid_r else None

    mid = len(valid_r) // 2
    first_half_mean = float(np.mean(valid_r[:mid])) if valid_r[:mid] else None
    second_half_mean = float(np.mean(valid_r[mid:])) if valid_r[mid:] else None

    if first_half_mean is not None and second_half_mean is not None:
        if second_half_mean < first_half_mean:
            trend = "R is decreasing, suggesting slower transmission"
        elif second_half_mean > first_half_mean:
            trend = "R is increasing, suggesting faster transmission"
        else:
            trend = "R is stable, suggesting little change in transmission speed"
    else:
        trend = "Insufficient valid data to determine the R-value trend"

    return {
        "labels": labels,
        "r_values": r_values,
        "mean_r": mean_r,
        "trend": trend,
    }


# ===== 来自 MLE Infection Growth Rate Estimation：MLE 指数增长率 =====
def mle感染增长估计(admissions: List[int]) -> Dict[str, object]:
    if len(admissions) != 7:
        raise ValueError("MLE is only applicable to 7-day admission data.")

    admissions_np = np.array(admissions, dtype=float)
    if np.any(admissions_np < 0):
        raise ValueError("Admissions cannot be negative.")

    t = np.arange(7)
    y_safe = np.maximum(admissions_np, 1e-6)
    Y = np.log(y_safe)

    r_mle, ln_A = np.polyfit(t, Y, deg=1, full=False)
    A_mle = float(np.exp(ln_A))

    Y_pred = ln_A + r_mle * t
    residuals = Y - Y_pred
    n = len(t)
    df = n - 2
    SSE = float(np.sum(residuals ** 2))
    MSE = SSE / df
    SS_t = float(np.sum((t - np.mean(t)) ** 2))
    SE_r = math.sqrt(MSE / SS_t)
    t_critical = float(stats.t.ppf(1 - 0.025, df))
    ci_lower = float(r_mle - t_critical * SE_r)
    ci_upper = float(r_mle + t_critical * SE_r)

    if r_mle > 0.05:
        risk_status = "High outbreak risk: clear and sustained growth detected"
    elif 0 < r_mle <= 0.05:
        risk_status = "Slow growth: continued close monitoring is recommended"
    else:
        risk_status = "Stable or declining: transmission appears controlled or weakening"

    return {
        "MLE增长率r": round(float(r_mle), 4),
        "95%置信区间": (round(ci_lower, 4), round(ci_upper, 4)),
        "初始幅度A": round(A_mle, 2),
        "风险状态": risk_status,
    }


# ===== 来自 Recovery Rate Trend Analysis.py 的恢复率趋势分析 =====
def 恢复率趋势分析(admissions: List[int], discharges: List[int]) -> Dict[str, object]:
    occupancy = [0] * 7
    recovery_ratios = [0.0] * 7
    low_period = 0
    prolonged_stress = False

    occupancy[0] = admissions[0] - discharges[0]
    for i in range(1, 7):
        occupancy[i] = occupancy[i - 1] + admissions[i] - discharges[i]

    for i in range(1, 7):
        if occupancy[i - 1] <= 0:
            recovery_ratios[i] = 0.0
        else:
            recovery_ratios[i] = discharges[i] / occupancy[i - 1]

    for i in range(1, 7):
        if recovery_ratios[i] < 0.2:
            low_period += 1
            if low_period >= 3:
                prolonged_stress = True
        else:
            low_period = 0

    valid_ratios = recovery_ratios[1:7]
    trend = "Stable"
    if all(valid_ratios[i] <= valid_ratios[i + 1] for i in range(len(valid_ratios) - 1)):
        trend = "Improving"
    elif all(valid_ratios[i] >= valid_ratios[i + 1] for i in range(len(valid_ratios) - 1)):
        trend = "Declining"
    elif max(valid_ratios) - min(valid_ratios) >= 0.15:
        trend = "Fluctuating"
    else:
        trend = "Stable"

    return {
        "occupancy": occupancy,
        "daily_recovery_ratios": [round(r, 2) for r in valid_ratios],
        "trend": trend,
        "prolonged_stress": prolonged_stress,
        "message": "Recovery is slow; the ward is under prolonged stress" if prolonged_stress else "Recovery performance is within an acceptable range",
    }


def 用Task5判断MLE是否可用(admissions: List[int], task5_result: Dict[str, object]) -> Tuple[bool, str]:
    r_values = [r for r in task5_result["r_values"] if r is not None]
    mean_r = task5_result["mean_r"]
    上升天数 = sum(1 for r in r_values if r > 1)

    if len(admissions) != 7:
        return False, "MLE is only applicable to 7-day data"
    if any(a <= 0 for a in admissions):
        return False, "Zero or negative admissions make log-exponential fitting unstable"
    if len(r_values) < 4:
        return False, "Too few valid R-values to assess a stable growth pattern"
    if mean_r is None or mean_r <= 1.0:
        return False, "Mean R-value is not above 1, so sustained growth is not supported"
    if 上升天数 < 4:
        return False, "Too few days with R > 1 to support a clear upward trend"
    return True, "Task 5 suggests a sustained growth pattern, so MLE exponential growth estimation can be used"


def 综合分析(admissions: List[int], discharges: List[int]) -> AnalysisSummary:
    task5_result = 估计R值(admissions)
    recovery_result = 恢复率趋势分析(admissions, discharges)
    can_use_mle, reason = 用Task5判断MLE是否可用(admissions, task5_result)

    if can_use_mle:
        mle_result = mle感染增长估计(admissions)
        return AnalysisSummary(
            task5=task5_result,
            recovery=recovery_result,
            mle_used=True,
            mle_result=mle_result,
            mle_reason=reason,
        )

    return AnalysisSummary(
        task5=task5_result,
        recovery=recovery_result,
        mle_used=False,
        mle_result=None,
        mle_reason=reason,
    )


def 抽样不重复时间(count: int, start_min: int, end_min: int, rng: np.random.Generator) -> List[int]:
    if count <= 0:
        return []
    if end_min < start_min:
        raise ValueError(f"Invalid time range: {start_min} to {end_min}")
    population = end_min - start_min + 1
    if count <= population:
        chosen = rng.choice(np.arange(start_min, end_min + 1), size=count, replace=False)
        return sorted(int(x) for x in chosen)
    chosen = rng.integers(start_min, end_min + 1, size=count)
    return sorted(int(x) for x in chosen)


def 抽样床位冷却时间(avg_minutes: float, rng: np.random.Generator) -> int:
    if avg_minutes < 0:
        raise ValueError("Average bed cooldown time cannot be negative.")
    if avg_minutes == 0:
        return 0
    return max(1, int(rng.poisson(avg_minutes)))


def 模拟单病房(
    ward_name: str,
    admissions: List[int],
    discharges: List[int],
    avg_bed_cooldown_minutes: float,
    capacity: int = 30,
    day_start: str = "00:00",
    day_end: str = "23:59",
    hold_frames: int = 2,
    min_stay_same_day_minutes: int = 120,
    seed: Optional[int] = 42,
) -> WardSimulationResult:
    if len(admissions) != 7 or len(discharges) != 7:
        raise ValueError(f"{ward_name} admissions and discharges must each contain exactly 7 days of data.")
    if any(a < 0 for a in admissions) or any(d < 0 for d in discharges):
        raise ValueError(f"{ward_name} admissions and discharges cannot be negative.")
    if capacity <= 0:
        raise ValueError("Bed capacity must be positive.")
    if hold_frames < 1:
        raise ValueError("hold_frames must be at least 1.")

    day_start_min = 解析时间(day_start)
    day_end_min = 解析时间(day_end)
    if day_end_min <= day_start_min:
        raise ValueError("Each day end time must be later than the start time.")

    rng = np.random.default_rng(seed)
    beds: List[Bed] = [Bed(bed_id=i) for i in range(capacity)]
    patients: Dict[int, Patient] = {}
    waiting_queue: Deque[int] = deque()
    active_patient_order: Deque[int] = deque()
    next_patient_id = 1

    history: List[WardState] = []
    day_end_occupancy: List[int] = []
    sampled_cooldowns: List[int] = []

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

    def make_state(abs_minute: int, adm_total: int, dis_total: int,
                   adm_done: int, dis_done: int, event_label: str) -> WardState:
        day = abs_minute // DAY_MINUTES + 1
        minute_of_day = abs_minute % DAY_MINUTES
        return WardState(
            ward_name=ward_name,
            abs_minute=abs_minute,
            day=day,
            time_label=格式化时间(minute_of_day),
            admissions_total=adm_total,
            discharges_total=dis_total,
            admissions_done=adm_done,
            discharges_done=dis_done,
            occupancy=current_occupancy(),
            overflow=overflow_count(),
            occupied_visible=visible_occupied_count(),
            bed_statuses=bed_statuses(abs_minute),
            event_label=event_label,
            avg_cooldown_minutes=avg_bed_cooldown_minutes,
        )

    def append_state(state: WardState, repeat_n: int = 1) -> None:
        for _ in range(repeat_n):
            history.append(state)

    def first_available_bed(now_abs: int) -> Optional[int]:
        for bed in beds:
            if bed.occupant_id is None and bed.cooldown_until <= now_abs:
                return bed.bed_id
        return None

    def assign_waiting_to_beds(now_abs: int, adm_total: int, dis_total: int,
                               adm_done: int, dis_done: int) -> None:
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
            append_state(
                make_state(now_abs, adm_total, dis_total, adm_done, dis_done,
                           f"Waiting patient assigned to bed {bed_id + 1} (replacement #{moved})")
            )

    def earliest_next_bed_ready(now_abs: int) -> Optional[int]:
        if not waiting_queue:
            return None
        candidates = [bed.cooldown_until for bed in beds if bed.occupant_id is None and bed.cooldown_until > now_abs]
        return min(candidates) if candidates else None

    for day_idx, (adm, dis) in enumerate(zip(admissions, discharges), start=1):
        day_abs_start = (day_idx - 1) * DAY_MINUTES + day_start_min
        day_abs_end = (day_idx - 1) * DAY_MINUTES + day_end_min

        assign_waiting_to_beds(day_abs_start, adm, dis, 0, 0)
        start_label = f"{ward_name}: Start of a new day"
        append_state(make_state(day_abs_start, adm, dis, 0, 0, start_label), repeat_n=hold_frames)

        carryover_ids = [pid for pid in list(active_patient_order) if patients.get(pid) and patients[pid].active]
        min_same_day = max(0, dis - len(carryover_ids))
        max_same_day = min(adm, dis)
        if day_abs_start + min_stay_same_day_minutes > day_abs_end:
            max_same_day = 0
            min_same_day = 0
        if max_same_day < min_same_day:
            raise ValueError(f"{ward_name} Day {day_idx}: not enough current patients to satisfy the discharge constraints.")
        if max_same_day == min_same_day:
            same_day_discharge_n = max_same_day
        else:
            same_day_discharge_n = int(rng.integers(min_same_day, max_same_day + 1))
        carryover_discharge_n = dis - same_day_discharge_n
        carryover_discharge_ids = carryover_ids[:carryover_discharge_n]

        new_ids = list(range(next_patient_id, next_patient_id + adm))
        next_patient_id += adm
        same_day_ids = set(new_ids[:same_day_discharge_n])

        latest_same_day_admit = day_abs_end - min_stay_same_day_minutes
        early_admit_n = same_day_discharge_n
        late_admit_n = adm - early_admit_n
        early_admit_times = 抽样不重复时间(early_admit_n, day_abs_start, latest_same_day_admit, rng)
        late_admit_times = 抽样不重复时间(late_admit_n, day_abs_start, day_abs_end, rng)

        admissions_plan: List[Tuple[int, int]] = []
        for pid, t in zip(new_ids[:early_admit_n], early_admit_times):
            admissions_plan.append((pid, t))
        for pid, t in zip(new_ids[early_admit_n:], late_admit_times):
            admissions_plan.append((pid, t))
        admissions_plan.sort(key=lambda x: (x[1], x[0]))

        carryover_discharge_times = 抽样不重复时间(carryover_discharge_n, day_abs_start, day_abs_end, rng)
        base_events: List[Tuple[int, int, str, int]] = []
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
                assign_waiting_to_beds(now_abs, adm, dis, admissions_done, discharges_done)
                continue

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
                        label = f"{ward_name}: Admission +1, assigned to bed {bed_id + 1} ({admissions_done}/{adm})"
                    else:
                        waiting_queue.append(patient_id)
                        label = f"{ward_name}: Admission +1, no bed available, moved to waiting queue ({len(waiting_queue)})"
                    append_state(make_state(now_abs, adm, dis, admissions_done, discharges_done, label))
                    assign_waiting_to_beds(now_abs, adm, dis, admissions_done, discharges_done)

                else:
                    patient = patients.get(patient_id)
                    if patient is None or not patient.active:
                        continue

                    discharges_done += 1
                    patient.active = False
                    try:
                        active_patient_order.remove(patient_id)
                    except ValueError:
                        pass

                    if patient.bed_id is not None:
                        bed_id = patient.bed_id
                        sampled_cooldown = 抽样床位冷却时间(avg_bed_cooldown_minutes, rng)
                        sampled_cooldowns.append(sampled_cooldown)
                        beds[bed_id].occupant_id = None
                        beds[bed_id].cooldown_until = now_abs + sampled_cooldown
                        patient.bed_id = None
                        label = (
                            f"{ward_name}: Discharge -1, bed {bed_id + 1} cooling for {sampled_cooldown} minutes"
                            f"(mean {avg_bed_cooldown_minutes:.1f})"
                        )
                    else:
                        try:
                            waiting_queue.remove(patient_id)
                        except ValueError:
                            pass
                        label = f"{ward_name}: Waiting patient leaves"
                    append_state(make_state(now_abs, adm, dis, admissions_done, discharges_done, label))

        append_state(make_state(day_abs_end, adm, dis, admissions_done, discharges_done, f"{ward_name}: End of day"), repeat_n=hold_frames)
        day_end_occupancy.append(current_occupancy())

    return WardSimulationResult(
        ward_name=ward_name,
        history=history,
        day_end_occupancy=day_end_occupancy,
        sampled_cooldowns=sampled_cooldowns,
    )


def 构建联合时间线(result_a: WardSimulationResult, result_b: WardSimulationResult, summary_hold_frames: int = 12) -> List[CombinedFrame]:
    times = sorted(set([state.abs_minute for state in result_a.history] + [state.abs_minute for state in result_b.history]))
    times_a = [state.abs_minute for state in result_a.history]
    times_b = [state.abs_minute for state in result_b.history]

    frames: List[CombinedFrame] = []
    for t in times:
        idx_a = max(0, bisect_right(times_a, t) - 1)
        idx_b = max(0, bisect_right(times_b, t) - 1)
        state_a = result_a.history[idx_a]
        state_b = result_b.history[idx_b]
        day = t // DAY_MINUTES + 1
        frames.append(
            CombinedFrame(
                abs_minute=t,
                day=day,
                time_label=格式化时间(t % DAY_MINUTES),
                state_a=state_a,
                state_b=state_b,
                show_summary=False,
            )
        )

    final_frame = frames[-1]
    for _ in range(summary_hold_frames):
        frames.append(
            CombinedFrame(
                abs_minute=final_frame.abs_minute,
                day=final_frame.day,
                time_label=final_frame.time_label,
                state_a=final_frame.state_a,
                state_b=final_frame.state_b,
                show_summary=True,
            )
        )
    return frames


def 绘制病房床位(ax, state: WardState, capacity: int = 30, columns: int = 6) -> None:
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
            linewidth=1.0,
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
        ax.text(x + bed_w / 2, y + 0.2, f"{idx + 1}", ha="center", va="center", fontsize=7)

    total_w = columns * bed_w + (columns - 1) * gap_x
    total_h = rows * bed_h + (rows - 1) * gap_y
    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-0.2, total_h + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(state.ward_name, fontsize=13, fontweight="bold", pad=8)


def 绘制信息面板(ax, state: WardState, sampled_cooldowns: List[int]) -> None:
    ax.clear()
    ax.axis("off")
    实际均值 = float(np.mean(sampled_cooldowns)) if sampled_cooldowns else 0.0
    文本 = (
        f"Admissions today: {state.admissions_total}    Completed: {state.admissions_done}\n"
        f"Discharges today: {state.discharges_total}    Completed: {state.discharges_done}\n"
        f"Current in ward: {state.occupancy}    Visible occupied beds: {state.occupied_visible}\n"
        f"Waiting count: {state.overflow}    Set mean cooldown: {state.avg_cooldown_minutes:.1f} min\n"
        f"Observed mean cooldown so far: {实际均值:.1f} min\n"
        f"Latest event: {state.event_label}"
    )
    ax.text(
        0.02, 0.98, 文本,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor="#CCCCCC")
    )


def 格式化Task5与MLE(summary: AnalysisSummary) -> str:
    task5 = summary.task5
    r_values = ["None" if r is None else str(r) for r in task5["r_values"]]
    文本 = [
        f"MLE used: {'Yes' if summary.mle_used else 'No'}",
        f"Task 5 mean R-value: {task5['mean_r']}",
        f"Task 5 trend: {task5['trend']}",
        f"Daily R-values: {' / '.join(r_values)}",
        f"MLE decision note: {summary.mle_reason}",
    ]
    if summary.mle_used and summary.mle_result is not None:
        文本.extend([
            f"MLE growth rate r: {summary.mle_result['MLE增长率r']}",
            f"95% confidence interval: {summary.mle_result['95%置信区间']}",
            f"MLE risk status: {summary.mle_result['风险状态']}",
        ])
    else:
        文本.append("MLE not run: Task 5 R-value approximation kept as the final output")
    return "\n".join(文本)


def 格式化恢复率(summary: AnalysisSummary) -> str:
    recovery = summary.recovery
    return "\n".join([
        f"Recovery trend: {recovery['trend']}",
        f"Daily recovery ratios: {' / '.join(str(x) for x in recovery['daily_recovery_ratios'])}",
        f"Prolonged stress: {'Yes' if recovery['prolonged_stress'] else 'No'}",
        f"Note: {recovery['message']}",
    ])


def compare_cooldown_strategies(
    result_a: WardSimulationResult,
    result_b: WardSimulationResult,
    summary_a: AnalysisSummary,
    summary_b: AnalysisSummary,
) -> str:
    """Compare which mean cooldown performs better within this model.

    Note: this model does not explicitly simulate infection events, so "better" here
    means operationally better: lower waiting burden first, then lower Day-7 occupancy,
    with recovery trend used as a tie-breaker.
    """
    avg_wait_a = float(np.mean([s.overflow for s in result_a.history])) if result_a.history else 0.0
    avg_wait_b = float(np.mean([s.overflow for s in result_b.history])) if result_b.history else 0.0
    final_occ_a = result_a.day_end_occupancy[-1] if result_a.day_end_occupancy else 0
    final_occ_b = result_b.day_end_occupancy[-1] if result_b.day_end_occupancy else 0
    actual_cool_a = float(np.mean(result_a.sampled_cooldowns)) if result_a.sampled_cooldowns else 0.0
    actual_cool_b = float(np.mean(result_b.sampled_cooldowns)) if result_b.sampled_cooldowns else 0.0
    recovery_mean_a = float(np.mean(summary_a.recovery['daily_recovery_ratios'])) if summary_a.recovery['daily_recovery_ratios'] else 0.0
    recovery_mean_b = float(np.mean(summary_b.recovery['daily_recovery_ratios'])) if summary_b.recovery['daily_recovery_ratios'] else 0.0

    lines = [
        'Cooldown strategy comparison (operational interpretation within this model):',
        f"Ward A | set mean cooldown={result_a.history[-1].avg_cooldown_minutes:.1f} min | observed mean cooldown={actual_cool_a:.1f} min | average waiting={avg_wait_a:.2f} | Day-7 occupancy={final_occ_a} | mean recovery ratio={recovery_mean_a:.2f}",
        f"Ward B | set mean cooldown={result_b.history[-1].avg_cooldown_minutes:.1f} min | observed mean cooldown={actual_cool_b:.1f} min | average waiting={avg_wait_b:.2f} | Day-7 occupancy={final_occ_b} | mean recovery ratio={recovery_mean_b:.2f}",
    ]

    eps = 1e-9
    if (avg_wait_a + eps < avg_wait_b and final_occ_a <= final_occ_b) or (abs(avg_wait_a - avg_wait_b) <= eps and final_occ_a + eps < final_occ_b):
        lines.append('Overall judgement: Ward A has the better mean cooldown strategy in this simulation, because it maintains a lower waiting burden and no worse end-of-Day-7 occupancy.')
    elif (avg_wait_b + eps < avg_wait_a and final_occ_b <= final_occ_a) or (abs(avg_wait_a - avg_wait_b) <= eps and final_occ_b + eps < final_occ_a):
        lines.append('Overall judgement: Ward B has the better mean cooldown strategy in this simulation, because it maintains a lower waiting burden and no worse end-of-Day-7 occupancy.')
    else:
        if avg_wait_a < avg_wait_b and final_occ_a > final_occ_b:
            lines.append('Overall judgement: There is no single best cooldown strategy here. Ward A performs better for waiting burden, while Ward B performs better for Day-7 occupancy.')
        elif avg_wait_b < avg_wait_a and final_occ_b > final_occ_a:
            lines.append('Overall judgement: There is no single best cooldown strategy here. Ward B performs better for waiting burden, while Ward A performs better for Day-7 occupancy.')
        elif recovery_mean_a > recovery_mean_b:
            lines.append('Overall judgement: The primary operational metrics are mixed, but Ward A shows a slightly better recovery pattern and is the more favorable option overall in this run.')
        elif recovery_mean_b > recovery_mean_a:
            lines.append('Overall judgement: The primary operational metrics are mixed, but Ward B shows a slightly better recovery pattern and is the more favorable option overall in this run.')
        else:
            lines.append('Overall judgement: The two mean cooldown strategies perform very similarly in this simulation; neither is clearly better on the available metrics.')

    lines.append('Important note: this is an operational comparison only. Because the current model does not simulate actual infection transmission events, it cannot claim which cooldown is biologically safer—only which one performs better on queueing and ward-load outcomes.')
    return "\n".join(lines)


def 绘制总结面板(ax, summary_a: AnalysisSummary, summary_b: AnalysisSummary, comparison_text: str, show_summary: bool) -> None:
    ax.clear()
    ax.axis("off")
    if not show_summary:
        ax.text(
            0.5, 0.5,
            "Animation in progress: after Day 7 ends, Task 5 + MLE and recovery trend results for Wards A and B will be shown here.",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#DDDDDD")
        )
        return

    左文本 = "Ward A summary\n" + "-" * 24 + "\n" + 格式化Task5与MLE(summary_a) + "\n\n" + 格式化恢复率(summary_a)
    右文本 = "Ward B summary\n" + "-" * 24 + "\n" + 格式化Task5与MLE(summary_b) + "\n\n" + 格式化恢复率(summary_b)

    ax.text(
        0.02, 0.96, 左文本,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#CCCCCC")
    )
    ax.text(
        0.52, 0.96, 右文本,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#CCCCCC")
    )


def 运行动画(
    frames: List[CombinedFrame],
    result_a: WardSimulationResult,
    result_b: WardSimulationResult,
    summary_a: AnalysisSummary,
    summary_b: AnalysisSummary,
    comparison_text: str,
    interval_ms: int = 500,
    save_path: Optional[str] = None,
    fps: int = 4,
) -> None:
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 2, height_ratios=[0.22, 1.0, 0.42, 0.72], hspace=0.18, wspace=0.08, figure=fig)

    ax_top = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    ax_info_a = fig.add_subplot(gs[2, 0])
    ax_info_b = fig.add_subplot(gs[2, 1])
    ax_summary = fig.add_subplot(gs[3, :])

    def update(frame_index: int):
        frame = frames[frame_index]

        ax_top.clear()
        ax_top.axis("off")
        顶部标题 = f"Pilot hospital two-ward 7-day dynamic simulation | Shared time: Day {frame.day} {frame.time_label}"
        if frame.show_summary:
            顶部标题 += " | End-of-Day-7 summary"
        ax_top.text(0.5, 0.65, 顶部标题, ha="center", va="center", fontsize=18, fontweight="bold")
        ax_top.text(0.5, 0.20, "Legend: red=occupied, green=empty, yellow=cooling; Wards A and B share the same time axis.", ha="center", va="center", fontsize=10, color="#555555")

        ax_a.clear()
        ax_b.clear()
        绘制病房床位(ax_a, frame.state_a, capacity=WARD_CAPACITY)
        绘制病房床位(ax_b, frame.state_b, capacity=WARD_CAPACITY)
        绘制信息面板(ax_info_a, frame.state_a, result_a.sampled_cooldowns)
        绘制信息面板(ax_info_b, frame.state_b, result_b.sampled_cooldowns)
        绘制总结面板(ax_summary, summary_a, summary_b, comparison_text, frame.show_summary)
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        repeat=False,
        blit=False,
    )
    fig._anim = anim

    plt.tight_layout()

    if save_path:
        suffix = os.path.splitext(save_path)[1].lower()
        if suffix == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=fps))
        elif suffix == ".mp4":
            try:
                anim.save(save_path, fps=fps)
            except Exception as exc:
                raise RuntimeError("Saving mp4 usually requires ffmpeg; if it fails, try gif instead.") from exc
        else:
            raise ValueError("Save path must end with .gif or .mp4.")
        print(f"Animation saved to: {save_path}")

    plt.show(block=True)


def 读取用户输入() -> Dict[str, object]:
    print("=" * 72)
    print("Pilot hospital two-ward 7-day dynamic simulation")
    print("Ward A uses the original example admissions/discharges from ward_analysis2.py.")
    print("      Ward B admissions/discharges are entered interactively below.")
    print("      Random seed is fixed at 42, frame interval is fixed at 450 ms, and the animation is displayed directly without asking for a save path.")
    print("=" * 72)

    avg_a = 读取浮点输入("Enter the mean bed cooldown time for Ward A (minutes)", 40.0)

    文本_b_adm = input(
        f"Enter Ward B 7-day admissions (comma-separated; default {DEFAULT_WARD_B_ADMISSIONS})\n> "
    )
    admissions_b = 解析7天整数列表(文本_b_adm, DEFAULT_WARD_B_ADMISSIONS)

    文本_b_dis = input(
        f"Enter Ward B 7-day discharges (comma-separated; default {DEFAULT_WARD_B_DISCHARGES})\n> "
    )
    discharges_b = 解析7天整数列表(文本_b_dis, DEFAULT_WARD_B_DISCHARGES)

    avg_b = 读取浮点输入("Enter the mean bed cooldown time for Ward B (minutes)", 30.0)

    return {
        "admissions_a": list(DEFAULT_WARD_A_ADMISSIONS),
        "discharges_a": list(DEFAULT_WARD_A_DISCHARGES),
        "avg_a": avg_a,
        "admissions_b": admissions_b,
        "discharges_b": discharges_b,
        "avg_b": avg_b,
        "seed": 42,
        "interval_ms": 450,
        "save_path": None,
    }

def main() -> None:
    参数 = 读取用户输入()

    admissions_a = 参数["admissions_a"]
    discharges_a = 参数["discharges_a"]
    admissions_b = 参数["admissions_b"]
    discharges_b = 参数["discharges_b"]

    summary_a = 综合分析(admissions_a, discharges_a)
    summary_b = 综合分析(admissions_b, discharges_b)

    result_a = 模拟单病房(
        ward_name="Ward A",
        admissions=admissions_a,
        discharges=discharges_a,
        avg_bed_cooldown_minutes=float(参数["avg_a"]),
        capacity=WARD_CAPACITY,
        seed=int(参数["seed"]),
    )
    result_b = 模拟单病房(
        ward_name="Ward B",
        admissions=admissions_b,
        discharges=discharges_b,
        avg_bed_cooldown_minutes=float(参数["avg_b"]),
        capacity=WARD_CAPACITY,
        seed=int(参数["seed"]) + 1000,
    )

    frames = 构建联合时间线(result_a, result_b)
    comparison_text = compare_cooldown_strategies(result_a, result_b, summary_a, summary_b)

    print("\nWard A day-end occupancy counts across 7 days:", result_a.day_end_occupancy)
    print("Ward B day-end occupancy counts across 7 days:", result_b.day_end_occupancy)
    print("Ward A MLE used:", "Yes" if summary_a.mle_used else "No")
    print("Ward B MLE used:", "Yes" if summary_b.mle_used else "No")
    print("\n" + comparison_text)

    运行动画(
        frames=frames,
        result_a=result_a,
        result_b=result_b,
        summary_a=summary_a,
        summary_b=summary_b,
        comparison_text=comparison_text,
        interval_ms=int(参数["interval_ms"]),
        save_path=参数["save_path"],
    )


if __name__ == "__main__":
    main()
