#!/usr/bin/env python3
"""List or rename DICOM patient names found under CT and RT folders.

The script scans each target folder, reads DICOM metadata without pixel data,
and prints one row per case/series folder. It supports both of these common
layouts:

    dataset/
      CT/
        case_001/*.dcm
        case_002/*.dcm
      RT/
        case_001/*.dcm
        case_002/*.dcm

    dataset/
      patient_or_case_001/
        CT/*.dcm
        RT/*.dcm
      patient_or_case_002/
        CT/*.dcm
        RT/*.dcm

    dataset/
      1.2.410.200113.1.20421.20260415003711893/*.dcm  # RT
      2.25.181506724508997602582525749623665176700/*.dcm  # CT

Examples:
    python scripts/list_patient_names.py /path/to/dataset
    python scripts/list_patient_names.py --ct /path/to/CT --rt /path/to/RT
    python scripts/list_patient_names.py /path/to/dataset --csv patient_names.csv
    python scripts/list_patient_names.py /path/to/dataset --export-rename-csv rename.csv
    python scripts/list_patient_names.py /path/to/dataset --rename-csv rename.csv --apply
    python scripts/list_patient_names.py /path/to/dataset --interactive-rename --apply
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_TARGET_NAMES = ("CT", "RT")
UID_PREFIX_TO_GROUP = {"1": "RT", "2": "CT"}
RENAME_CSV_COLUMNS = [
    "current_patient_name",
    "new_patient_name",
    "groups",
    "folders",
    "dicom_files",
]
DICOM_FIELDS = {
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0008, 0x0060): "Modality",
}
LONG_VALUE_REPRESENTATIONS = {
    "OB",
    "OD",
    "OF",
    "OL",
    "OV",
    "OW",
    "SQ",
    "UC",
    "UR",
    "UT",
    "UN",
}


@dataclass(frozen=True)
class RenameRequest:
    """A requested PatientName replacement."""

    old_name: str
    new_name: str


@dataclass(frozen=True)
class ScanTarget:
    """A CT or RT folder selected for scanning."""

    group: str
    folder: Path
    include_root_files: bool = False


@dataclass
class PatientNameSummary:
    """Locations where one PatientName appears."""

    groups: set[str] = field(default_factory=set)
    folders: set[str] = field(default_factory=set)
    dicom_files: int = 0


@dataclass
class FolderSummary:
    """Patient metadata summarized for one folder."""

    group: str
    folder: Path
    dicom_files: int = 0
    patient_names: set[str] = field(default_factory=set)
    patient_ids: set[str] = field(default_factory=set)
    modalities: set[str] = field(default_factory=set)
    read_errors: int = 0
    rename_matches: int = 0
    renamed_files: int = 0

    def as_row(self) -> dict[str, str | int]:
        return {
            "group": self.group,
            "folder": str(self.folder),
            "dicom_files": self.dicom_files,
            "patient_names": join_values(self.patient_names),
            "patient_ids": join_values(self.patient_ids),
            "modalities": join_values(self.modalities),
            "read_errors": self.read_errors,
            "rename_matches": self.rename_matches,
            "renamed_files": self.renamed_files,
        }


def join_values(values: Iterable[str]) -> str:
    """Return a stable, human-readable representation for a set of values."""

    cleaned = sorted(value for value in values if value)
    return "; ".join(cleaned) if cleaned else "-"


def parse_rename_request(value: str) -> RenameRequest:
    """Parse OLD=NEW rename syntax from the command line."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Rename values must use OLD=NEW syntax, for example: 'JOHN^DOE=ANON001'"
        )

    old_name, new_name = (part.strip() for part in value.split("=", 1))
    if not old_name or not new_name:
        raise argparse.ArgumentTypeError("Both OLD and NEW names must be non-empty.")
    return RenameRequest(old_name=old_name, new_name=new_name)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read DICOM files under CT/RT folders and print or rename the "
            "PatientName values found in each folder."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help=(
            "Dataset root that contains CT/RT folders, case folders that each "
            "contain CT/RT folders, or UID-named series folders where names "
            "starting with 1 are RT and names starting with 2 are CT. Ignored "
            "for a group when --ct or --rt is supplied. Default: current "
            "directory."
        ),
    )
    parser.add_argument("--ct", type=Path, help="Path to the CT folder to scan.")
    parser.add_argument("--rt", type=Path, help="Path to the RT folder to scan.")
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path where the same summary should be written as CSV.",
    )
    parser.add_argument(
        "--include-root-files",
        action="store_true",
        help=(
            "Also summarize DICOM files placed directly inside explicit/root "
            "CT/RT folders, not only files inside child folders. This is "
            "enabled automatically for case-folder layouts like case_001/CT."
        ),
    )
    parser.add_argument(
        "--rename-patient",
        action="append",
        default=[],
        metavar="OLD=NEW",
        type=parse_rename_request,
        help=(
            "Plan a PatientName replacement. May be supplied multiple times. "
            "The command is a dry run unless --apply is also supplied."
        ),
    )
    parser.add_argument(
        "--export-rename-csv",
        type=Path,
        help=(
            "Write a CSV template with current_patient_name and blank "
            "new_patient_name cells so you can edit names manually."
        ),
    )
    parser.add_argument(
        "--rename-csv",
        type=Path,
        help=(
            "Read PatientName replacements from a CSV created by "
            "--export-rename-csv. Rows with blank new_patient_name are skipped."
        ),
    )
    parser.add_argument(
        "--interactive-rename",
        action="store_true",
        help=(
            "Ask for a new name for each discovered PatientName in the terminal. "
            "Blank input keeps the original name."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write PatientName changes to matching DICOM files.",
    )
    return parser.parse_args(argv)


def classify_uid_series_folder(folder_name: str) -> str | None:
    """Classify UID-like series folders by the leading number convention.

    In the dataset this script targets, RT series folders are named with UIDs
    that start with ``1`` and CT series folders are named with UIDs that start
    with ``2``. The check stays limited to digit/dot UID-like names so ordinary
    case names such as ``101_HM10395`` are not misclassified as RT.
    """

    if not folder_name or "." not in folder_name:
        return None
    if any(character not in "0123456789." for character in folder_name):
        return None
    return UID_PREFIX_TO_GROUP.get(folder_name[0])


def add_target(
    targets: list[ScanTarget],
    seen: set[Path],
    group: str,
    folder: Path,
    include_root_files: bool,
) -> None:
    """Add a target once, using resolved paths to avoid duplicate scans."""

    resolved = folder.resolve()
    if resolved not in seen:
        targets.append(
            ScanTarget(
                group=group,
                folder=folder,
                include_root_files=include_root_files,
            )
        )
        seen.add(resolved)


def group_label_for_folder(root: Path, folder: Path, group: str) -> str:
    """Return a readable group label for a discovered CT/RT folder."""

    try:
        relative_folder = folder.relative_to(root)
    except ValueError:
        return group

    if str(relative_folder) == "." or not relative_folder.parts[:-1]:
        return group
    return f"{relative_folder.parent.as_posix()}/{group}"


def iter_candidate_folders(root: Path) -> Iterable[Path]:
    """Yield root and every nested folder under root."""

    yield root
    for path in root.rglob("*"):
        if path.is_dir():
            yield path


def resolve_targets(args: argparse.Namespace) -> list[ScanTarget]:
    """Resolve explicit and recursively discovered CT/RT or UID folders."""

    targets: list[ScanTarget] = []
    seen: set[Path] = set()
    explicit_targets = {"CT": args.ct, "RT": args.rt}

    for group in DEFAULT_TARGET_NAMES:
        explicit_path = explicit_targets[group]
        if explicit_path is None:
            continue
        if not explicit_path.exists():
            raise FileNotFoundError(f"{group} folder does not exist: {explicit_path}")
        add_target(targets, seen, group, explicit_path, args.include_root_files)

    if targets:
        return targets

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")
    if not args.root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a folder: {args.root}")

    for folder in iter_candidate_folders(args.root):
        folder_name = folder.name
        name_group = (
            folder_name.upper() if folder_name.upper() in DEFAULT_TARGET_NAMES else None
        )
        uid_group = classify_uid_series_folder(folder_name)

        if name_group:
            add_target(
                targets,
                seen,
                group_label_for_folder(args.root, folder, name_group),
                folder,
                include_root_files=(folder == args.root and args.include_root_files)
                or folder != args.root,
            )
        elif uid_group:
            add_target(
                targets,
                seen,
                group_label_for_folder(args.root, folder, uid_group),
                folder,
                include_root_files=True,
            )

    if not targets:
        direct_expected = ", ".join(
            str(args.root / name) for name in DEFAULT_TARGET_NAMES
        )
        nested_expected = ", ".join(
            str(args.root / "101_HM10395" / "100_HM10395" / example)
            for example in (
                "1.2.410.200113.1.20421.20260415003711893",
                "2.25.181506724508997602582525749623665176700",
            )
        )
        raise FileNotFoundError(
            "No CT or RT folder found. The script now searches recursively, "
            "so check that the selected root contains folders named CT/RT or "
            "UID-like folders where 1.* is RT and 2.* is CT. Examples: "
            f"{direct_expected}; {nested_expected}"
        )

    return targets


def iter_dicom_candidates(folder: Path) -> Iterable[Path]:
    """Yield files that may be DICOM files.

    DICOM files often have no extension, so this intentionally yields every file
    and lets pydicom decide whether the file can be read as DICOM metadata.
    """

    for path in folder.rglob("*"):
        if path.is_file():
            yield path


def summary_bucket_for_file(
    base_folder: Path, file_path: Path, include_root_files: bool
) -> Path | None:
    """Choose which immediate child folder should own a scanned file."""

    relative_parts = file_path.relative_to(base_folder).parts
    if len(relative_parts) == 1:
        return base_folder if include_root_files else None
    return base_folder / relative_parts[0]


def read_with_pydicom(file_path: Path) -> dict[str, str] | None:
    """Read selected DICOM tags with pydicom when it is installed."""

    if importlib.util.find_spec("pydicom") is None:
        return None

    pydicom = importlib.import_module("pydicom")
    dataset = pydicom.dcmread(
        str(file_path),
        stop_before_pixels=True,
        force=True,
        specific_tags=list(DICOM_FIELDS.values()),
    )
    if not dataset:
        return {}
    return {
        "PatientName": str(getattr(dataset, "PatientName", "")).strip(),
        "PatientID": str(getattr(dataset, "PatientID", "")).strip(),
        "Modality": str(getattr(dataset, "Modality", "")).strip(),
    }


def require_pydicom_for_rename():
    """Return pydicom or raise a helpful error for write operations."""

    if importlib.util.find_spec("pydicom") is None:
        raise RuntimeError(
            "Writing DICOM PatientName changes requires pydicom. "
            "Install it with: python -m pip install -r requirements.txt"
        )
    return importlib.import_module("pydicom")


def rename_patient_name(
    file_path: Path, rename_map: dict[str, str], apply: bool
) -> bool:
    """Rename PatientName in a DICOM file when it matches a request."""

    pydicom = require_pydicom_for_rename()
    dataset = pydicom.dcmread(str(file_path), force=True)
    current_name = str(getattr(dataset, "PatientName", "")).strip()

    if current_name not in rename_map:
        return False

    dataset.PatientName = rename_map[current_name]
    if apply:
        dataset.save_as(str(file_path), write_like_original=True)
    return True


def decode_dicom_text(value: bytes) -> str:
    """Decode common DICOM single-byte text values."""

    return value.rstrip(b" \0").decode("utf-8", errors="replace").strip()


def read_with_builtin_parser(file_path: Path) -> dict[str, str]:
    """Read selected tags with a small metadata-only DICOM parser.

    This fallback is intentionally limited, but it handles the usual explicit
    and implicit VR little-endian files well enough for PatientName checks when
    pydicom is not installed.
    """

    data = file_path.read_bytes()
    offset = 132 if len(data) >= 132 and data[128:132] == b"DICM" else 0
    values: dict[str, str] = {}

    while offset + 8 <= len(data) and len(values) < len(DICOM_FIELDS):
        group, element = struct.unpack_from("<HH", data, offset)
        offset += 4

        if group == 0x7FE0 and element == 0x0010:
            break

        vr_candidate = data[offset : offset + 2]
        if vr_candidate.isalpha():
            vr = vr_candidate.decode("ascii", errors="ignore")
            offset += 2
            if vr in LONG_VALUE_REPRESENTATIONS:
                if offset + 6 > len(data):
                    break
                offset += 2
                value_length = struct.unpack_from("<I", data, offset)[0]
                offset += 4
            else:
                if offset + 2 > len(data):
                    break
                value_length = struct.unpack_from("<H", data, offset)[0]
                offset += 2
        else:
            if offset + 4 > len(data):
                break
            value_length = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        if value_length == 0xFFFFFFFF:
            break
        if value_length < 0 or offset + value_length > len(data):
            break

        field_name = DICOM_FIELDS.get((group, element))
        if field_name:
            values[field_name] = decode_dicom_text(
                data[offset : offset + value_length]
            )
        offset += value_length + (value_length % 2)

    return values


def read_dicom_fields(file_path: Path) -> dict[str, str]:
    """Read patient fields using pydicom when possible, then the fallback parser."""

    pydicom_values = read_with_pydicom(file_path)
    if pydicom_values is not None:
        return pydicom_values
    return read_with_builtin_parser(file_path)


def scan_target(
    target: ScanTarget,
    rename_map: dict[str, str],
    apply_renames: bool,
) -> list[FolderSummary]:
    summaries: dict[Path, FolderSummary] = {}

    for file_path in iter_dicom_candidates(target.folder):
        bucket = summary_bucket_for_file(
            target.folder, file_path, target.include_root_files
        )
        if bucket is None:
            continue

        summary = summaries.setdefault(
            bucket, FolderSummary(group=target.group, folder=bucket)
        )
        try:
            fields = read_dicom_fields(file_path)
        except Exception:
            summary.read_errors += 1
            continue

        patient_name = fields.get("PatientName", "")
        patient_id = fields.get("PatientID", "")
        modality = fields.get("Modality", "")
        display_patient_name = patient_name

        if patient_name in rename_map:
            summary.rename_matches += 1
            if apply_renames:
                try:
                    if rename_patient_name(file_path, rename_map, apply_renames):
                        summary.renamed_files += 1
                        display_patient_name = rename_map[patient_name]
                except Exception:
                    summary.read_errors += 1

        if display_patient_name or patient_id or modality:
            summary.dicom_files += 1
            summary.patient_names.add(display_patient_name)
            summary.patient_ids.add(patient_id)
            summary.modalities.add(modality)

    return sorted(summaries.values(), key=lambda item: (item.group, str(item.folder)))


def scan_targets(
    targets: Sequence[ScanTarget], rename_map: dict[str, str], apply_renames: bool
) -> list[FolderSummary]:
    """Scan all selected CT/RT targets."""

    return [
        summary
        for target in targets
        for summary in scan_target(target, rename_map, apply_renames)
    ]


def unique_patient_names(summaries: Sequence[FolderSummary]) -> list[str]:
    """Collect unique non-empty PatientName values from scan results."""

    return sorted(
        patient_name
        for summary in summaries
        for patient_name in summary.patient_names
        if patient_name
    )


def print_table(rows: list[dict[str, str | int]]) -> None:
    headers = [
        "group",
        "folder",
        "dicom_files",
        "patient_names",
        "patient_ids",
        "modalities",
        "read_errors",
        "rename_matches",
        "renamed_files",
    ]
    widths = defaultdict(int)

    for header in headers:
        widths[header] = len(header)
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def write_csv(rows: list[dict[str, str | int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rename_requests_to_map(rename_requests: Sequence[RenameRequest]) -> dict[str, str]:
    """Convert command-line rename requests into a mapping."""

    rename_map: dict[str, str] = {}
    for request in rename_requests:
        rename_map[request.old_name] = request.new_name
    return rename_map


def read_rename_csv(csv_path: Path) -> dict[str, str]:
    """Read current_patient_name -> new_patient_name replacements from CSV."""

    rename_map: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = {
            "current_patient_name",
            "new_patient_name",
        } - set(reader.fieldnames or [])
        if missing_columns:
            raise RuntimeError(
                f"Rename CSV is missing required columns: {join_values(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            current_name = (row.get("current_patient_name") or "").strip()
            new_name = (row.get("new_patient_name") or "").strip()
            if not current_name or not new_name or current_name == new_name:
                continue
            if current_name in rename_map and rename_map[current_name] != new_name:
                raise RuntimeError(
                    "Conflicting replacements for "
                    f"{current_name!r} in {csv_path} near line {line_number}."
                )
            rename_map[current_name] = new_name

    return rename_map


def write_rename_template(
    summaries: Sequence[FolderSummary], csv_path: Path
) -> None:
    """Write a user-editable rename CSV template from scan summaries."""

    per_name: dict[str, PatientNameSummary] = {}
    for summary in summaries:
        for patient_name in summary.patient_names:
            if not patient_name:
                continue
            info = per_name.setdefault(patient_name, PatientNameSummary())
            info.groups.add(summary.group)
            info.folders.add(str(summary.folder))
            info.dicom_files += summary.dicom_files

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RENAME_CSV_COLUMNS)
        writer.writeheader()
        for patient_name in sorted(per_name):
            info = per_name[patient_name]
            writer.writerow(
                {
                    "current_patient_name": patient_name,
                    "new_patient_name": "",
                    "groups": join_values(info.groups),
                    "folders": join_values(info.folders),
                    "dicom_files": info.dicom_files,
                }
            )


def prompt_for_renames(patient_names: Sequence[str]) -> dict[str, str]:
    """Ask the user for replacement names in an interactive terminal."""

    if not sys.stdin.isatty():
        raise RuntimeError("--interactive-rename requires an interactive terminal.")

    rename_map: dict[str, str] = {}
    print("\nDiscovered PatientName values. Press Enter to keep the current name.")
    for patient_name in patient_names:
        new_name = input(f"{patient_name} -> ").strip()
        if new_name and new_name != patient_name:
            rename_map[patient_name] = new_name
    return rename_map


def rows_from_summaries(summaries: Sequence[FolderSummary]) -> list[dict[str, str | int]]:
    """Convert summaries to printable/CSV rows."""

    return [
        summary.as_row()
        for summary in summaries
        if summary.dicom_files or summary.read_errors
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if args.apply and not (
            args.rename_patient or args.rename_csv or args.interactive_rename
        ):
            raise RuntimeError(
                "--apply can only be used with --rename-patient, --rename-csv, "
                "or --interactive-rename."
            )
        if args.apply:
            require_pydicom_for_rename()

        targets = resolve_targets(args)
        rename_map = rename_requests_to_map(args.rename_patient)
        if args.rename_csv:
            rename_map.update(read_rename_csv(args.rename_csv))

        summaries = scan_targets(targets, rename_map, apply_renames=args.apply)

        if args.interactive_rename:
            interactive_map = prompt_for_renames(unique_patient_names(summaries))
            rename_map.update(interactive_map)
            if rename_map:
                summaries = scan_targets(targets, rename_map, apply_renames=args.apply)

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows = rows_from_summaries(summaries)
    if not rows:
        print("No DICOM metadata with PatientName/PatientID/Modality was found.")
        return 0

    print_table(rows)

    if args.export_rename_csv:
        write_rename_template(summaries, args.export_rename_csv)
        print(f"\nRename CSV template written to: {args.export_rename_csv}")
    if rename_map and not args.apply:
        print("\nDry run only. Re-run with --apply to write PatientName changes.")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nCSV written to: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
