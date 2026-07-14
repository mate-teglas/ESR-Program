import os
import re
import shutil
import numpy as np


# ======================
# Regex
# ======================

RE_AMC = re.compile(r"AMC=([0-9\.\-]+)\s*deg")
RE_SWEEP = re.compile(r"sweeph1\s+([0-9\.\-]+)\s+([0-9\.\-]+)")
RE_SPLIT = re.compile(r"[,\s;]+")
RE_SAMPLE_TEMP = re.compile(
    r"(?:TB\s*\(\s*Sample\s*\)|T(?:emp(?:erature)?)?[_\s-]*(?:Sample)?)\s*[=:]\s*"
    r"([+-]?[0-9]+(?:\.[0-9]+)?)(?:\s*@\s*([+-]?[0-9]+(?:\.[0-9]+)?))?",
    re.IGNORECASE,
)
RE_APSYN = re.compile(
    r"APSyn\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*GHz",
    re.IGNORECASE,
)
RE_FREQUENCY = re.compile(
    r"(?:frequency|freq(?:uency)?|mw[_\s-]*freq(?:uency)?)\s*[=:]\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(GHz|MHz)?",
    re.IGNORECASE,
)
RE_MULTIPLIER = re.compile(
    r"(?:Anapico|APSyn|frequency|freq)?\s*[x×]\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
RE_PATH_GHZ = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GHz", re.IGNORECASE)
RE_PATH_TEMP = re.compile(r"(?:^|[/_\s-])([0-9]+(?:\.[0-9]+)?)\s*K(?:$|[/_\s-])", re.IGNORECASE)


# ======================
# Header parsing
# ======================

def parse_angle_and_direction(header_text: str):
    """
    Extract angle and sweep direction from ESR header.
    """

    m = RE_AMC.search(header_text)
    angle = float(m.group(1)) if m else None

    m = RE_SWEEP.search(header_text)

    direction = None

    if m:
        start = float(m.group(1))
        stop = float(m.group(2))

        direction = "up" if start < stop else "down"

    return angle, direction


def parse_measurement_metadata(header_text: str, filename=""):
    """Extract sample temperature and microwave frequency.

    The ESR headers used in this project normally store sample temperature as
    ``TB(Sample)=...`` and synthesizer frequency as ``APSyn=... GHz``.  When a
    multiplier such as ``Anapico x6`` is present, ``frequency_GHz`` is the
    effective sample frequency while ``synth_frequency_GHz`` preserves the
    instrument setting.  Folder names such as ``108 GHz`` and ``10K`` are used
    only as fallbacks.
    """
    text = header_text or ""
    path_text = os.path.abspath(filename) if filename else ""

    temp = None
    measured_temp = None
    m = RE_SAMPLE_TEMP.search(text)
    if m:
        measured_temp = float(m.group(1))
        # The value after @ is the nominal/set temperature and is the correct
        # grouping key; the value before @ is preserved as the measured value.
        temp = float(m.group(2)) if m.group(2) is not None else measured_temp
    if temp is None:
        matches = RE_PATH_TEMP.findall(path_text)
        if matches:
            temp = float(matches[-1])

    synth = None
    m = RE_APSYN.search(text)
    if m:
        synth = float(m.group(1))
    else:
        m = RE_FREQUENCY.search(text)
        if m:
            synth = float(m.group(1))
            if (m.group(2) or "").lower() == "mhz":
                synth /= 1000.0

    multiplier = 1.0
    m = RE_MULTIPLIER.search(text)
    if m:
        multiplier = float(m.group(1))

    path_frequency = None
    matches = RE_PATH_GHZ.findall(path_text)
    if matches:
        path_frequency = float(matches[-1])

    effective = synth * multiplier if synth is not None else path_frequency
    # Prefer an explicit folder frequency when it agrees with a multiplied
    # source only up to normal rounding (e.g. 17.9785 x 6 -> folder "108 GHz").
    if effective is not None and path_frequency is not None:
        if abs(effective - path_frequency) <= max(0.25, 0.005 * path_frequency):
            effective = path_frequency

    return {
        "temperature_K": temp,
        "measured_temperature_K": measured_temp,
        "frequency_GHz": effective,
        "synth_frequency_GHz": synth,
        "frequency_multiplier": multiplier,
    }


# ======================
# File reader
# ======================

def read_file(filename):
    """
    Reads ESR .dat file.

    Returns:

        angle
        direction
        B
        ch1
        ch2
        header_lines
    """

    with open(filename, "r", errors="ignore") as f:
        lines = f.read().splitlines()

    header = []
    data = []

    in_data = False
    first_numeric_checked = False

    for line in lines:

        s = line.strip()

        if not s:
            continue

        if not in_data:

            c0 = s[0]

            if c0.isdigit() or c0 in "+-.":
                in_data = True
            else:
                header.append(line)
                continue

        parts = RE_SPLIT.split(s)

        nums = []

        ok = True

        for p in parts:

            if p == "":
                continue

            try:
                nums.append(float(p))
            except ValueError:
                ok = False
                break

        if (not ok) or (len(nums) < 2):
            continue

        #
        # Skip old style
        # "3 8400 ..."
        #

        if not first_numeric_checked:

            first_numeric_checked = True

            a0, a1 = nums[0], nums[1]

            # Skip metadata/count line like:
            # 3,7200
            # 3 8400 ...
            # This is not real B-field data.
            if (
                    len(nums) == 2
                    and float(a0).is_integer()
                    and 1 <= a0 <= 10
                    and float(a1).is_integer()
                    and a1 >= 1000
            ):
                continue

        B = nums[0]
        ch1 = nums[1]

        if len(nums) >= 3:
            ch2 = nums[2]
        else:
            ch2 = np.nan

        data.append((B, ch1, ch2))

    if not data:
        return None, None, None, None, None, header

    data = np.asarray(data, dtype=float)

    B = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 2]

    # Safety filter:
    # Keep only physically reasonable ESR field points.
    # This removes accidental metadata lines like 3,7200
    # if they somehow slipped into the data.
    mask = np.isfinite(B) & np.isfinite(ch1) & (B >= 0) & (B <= 20)

    B = B[mask]
    ch1 = ch1[mask]
    ch2 = ch2[mask]

    if B.size == 0:
        return None, None, None, None, None, header

    idx = np.argsort(B)

    B = B[idx]
    ch1 = ch1[idx]
    ch2 = ch2[idx]

    angle, direction = parse_angle_and_direction(
        "\n".join(header)
    )

    return angle, direction, B, ch1, ch2, header


# ======================
# File helpers
# ======================

def file_number_from_name(name: str):

    digits = "".join(
        ch for ch in name
        if ch.isdigit()
    )

    if digits:
        return int(digits)

    return None


def take_last_n_by_number_then_name(items, n):

    def key(m):

        num = file_number_from_name(
            m["name"]
        )

        return (
            num if num is not None else 10**9,
            m["name"]
        )

    s = sorted(items, key=key)

    if len(s) > n:
        return s[-n:]

    return s


# ======================
# Angle clustering
# ======================

def cluster_by_angle(items, tol_deg):

    items = sorted(
        items,
        key=lambda it: it["angle"]
    )

    clusters = []

    current = []
    current_mean = None

    for it in items:

        if not current:

            current = [it]
            current_mean = it["angle"]

            continue

        if abs(
            it["angle"] - current_mean
        ) <= tol_deg:

            current.append(it)

            current_mean = np.mean(
                [x["angle"] for x in current]
            )

        else:

            clusters.append(current)

            current = [it]
            current_mean = it["angle"]

    if current:
        clusters.append(current)

    return clusters


# ======================
# Export helpers
# ======================

def make_export_header_from_original(
    original_header_lines,
    extra_lines
):

    out = []

    if original_header_lines:
        out.extend(original_header_lines)
    else:
        out.append("ESR export")

    out.append("")

    out.extend(extra_lines)

    out.append("")

    return out


def write_dat_3col(
    path,
    header_lines,
    B,
    y1,
    y2
):
    """
    Save:

        B
        CH1
        CH2

    preserving ESR-style header.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for h in header_lines:
            f.write(h + "\n")

        for Bv, a, b in zip(
            B,
            y1,
            y2
        ):
            f.write(
                f"{Bv:.10g}\t{a:.10g}\t{b:.10g}\n"
            )


# ======================
# Folder scan
# ======================

def load_folder(
    folder,
    prefix="k",
    ext=".dat"
):
    """
    Load all ESR files from folder.

    Returns:
        items list
    """

    files = []

    for n in os.listdir(folder):

        prefix_ok = (

            True if prefix == ""

            else n.startswith(prefix)

        )

        ext_ok = (

            True if ext == ""

            else n.endswith(ext)

        )

        if prefix_ok and ext_ok:
            files.append(

                os.path.join(folder, n)

            )

    files.sort()

    items = []

    for fp in files:

        angle, direction, B, ch1, ch2, header = read_file(fp)

        if angle is None:
            continue

        metadata = parse_measurement_metadata("\n".join(header), fp)

        items.append({

            "name":
                os.path.basename(fp),

            "path":
                fp,

            "angle":
                float(angle),

            "direction":
                (direction or "").lower(),

            "B":
                B,

            "ch1":
                ch1,

            "ch2":
                ch2,

            "header":
                header,

            **metadata,
        })

    return items


def metadata_value_key(value, decimals=3):
    """Stable grouping key for floating metadata, preserving unknown values."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), int(decimals))


def split_files_by_metadata(items, destination, mode="copy",
                            include_direction=True, include_temperature=True,
                            include_frequency=True):
    """Copy or move loaded files into metadata-named subfolders.

    The original filename is preserved.  Existing different files are never
    overwritten: a numeric suffix is added instead.  Returns a list of
    ``(source, destination)`` pairs.
    """
    mode = str(mode).lower()
    if mode not in ("copy", "move"):
        raise ValueError("mode must be 'copy' or 'move'")
    os.makedirs(destination, exist_ok=True)
    written = []

    def label(prefix, value, unit=""):
        if value is None or not np.isfinite(value):
            return f"{prefix}_unknown"
        number = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return f"{prefix}_{number}{unit}"

    for item in items:
        parts = []
        if include_temperature:
            parts.append(label("T", item.get("temperature_K"), "K"))
        if include_frequency:
            parts.append(label("F", item.get("frequency_GHz"), "GHz"))
        if include_direction:
            parts.append(str(item.get("direction") or "direction_unknown").upper())
        target_dir = os.path.join(destination, *parts) if parts else destination
        os.makedirs(target_dir, exist_ok=True)
        source = item["path"]
        target = os.path.join(target_dir, os.path.basename(source))
        stem, ext = os.path.splitext(target)
        serial = 2
        while os.path.exists(target) and not os.path.samefile(source, target):
            target = f"{stem}_{serial}{ext}"
            serial += 1
        if os.path.abspath(source) == os.path.abspath(target):
            continue
        if mode == "move":
            shutil.move(source, target)
        else:
            shutil.copy2(source, target)
        written.append((source, target))
    return written
