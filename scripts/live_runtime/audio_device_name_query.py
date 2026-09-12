"""Resolve MME device indexes from a named route profile. Memory only."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


PROFILE_LOCAL_USB = "local_usb"
PROFILE_ZOOM = "zoom"

PROFILE_LABELS = {
    PROFILE_LOCAL_USB: "手元 USB",
    PROFILE_ZOOM: "Zoom",
}

# Third-party path = name set only. Indexes are queried each process start.
_ROUTE_RULES: dict[str, dict[str, Any]] = {
    PROFILE_LOCAL_USB: {
        "label": PROFILE_LABELS[PROFILE_LOCAL_USB],
        "output": {
            "include": ("USB PnP",),
            "exclude": ("WebCamera",),
            "kind": "output",
        },
        "input": {
            "include": ("USB PnP",),
            "exclude": ("WebCamera",),
            "kind": "input",
        },
    },
    PROFILE_ZOOM: {
        "label": PROFILE_LABELS[PROFILE_ZOOM],
        "output": {
            "include": ("CABLE Input",),
            "exclude": ("CABLE In 16ch",),
            "kind": "output",
        },
        "input": {
            "include": ("Voicemeeter Out B1",),
            "exclude": ("CABLE Output",),
            "kind": "input",
        },
    },
}


@dataclass(frozen=True)
class MmeDevice:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    hostapi_name: str


@dataclass(frozen=True)
class ResolvedEndpoint:
    index: int
    name: str
    kind: str


@dataclass(frozen=True)
class ResolvedRoute:
    profile: str
    label: str
    output: ResolvedEndpoint
    input: ResolvedEndpoint


class AudioRouteResolveError(Exception):
    def __init__(
        self,
        reason: str,
        *,
        profile: str,
        listing: Sequence[MmeDevice],
        out_matches: Sequence[MmeDevice],
        in_matches: Sequence[MmeDevice],
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.profile = profile
        self.listing = list(listing)
        self.out_matches = list(out_matches)
        self.in_matches = list(in_matches)

    def format_listing(self) -> str:
        lines = ["[audio_route][mme]"]
        if not self.listing:
            lines.append("  (none)")
            return "\n".join(lines)
        for dev in self.listing:
            lines.append(
                f"  {dev.index} ha={dev.hostapi_name} "
                f"in={dev.max_input_channels} out={dev.max_output_channels} "
                f"{dev.name}"
            )
        return "\n".join(lines)

    def format_refuse(self) -> str:
        def _fmt_matches(kind: str, matches: Sequence[MmeDevice]) -> str:
            if not matches:
                return f"[audio_route][matches] {kind} n=0"
            shown = ", ".join(f"{d.index}:{d.name}" for d in matches)
            return f"[audio_route][matches] {kind} n={len(matches)} [{shown}]"

        return "\n".join(
            [
                f"[audio_route][refuse] profile={self.profile} reason={self.reason}",
                _fmt_matches("out", self.out_matches),
                _fmt_matches("in", self.in_matches),
                self.format_listing(),
            ]
        )


def list_mme_devices(
    *,
    query_devices: Callable[[], Any] | None = None,
    query_hostapis: Callable[[], Any] | None = None,
) -> list[MmeDevice]:
    if query_devices is None or query_hostapis is None:
        import sounddevice as sd

        if query_devices is None:
            query_devices = sd.query_devices
        if query_hostapis is None:
            query_hostapis = sd.query_hostapis

    raw_devices = query_devices()
    raw_hostapis = query_hostapis()
    hostapi_names = [str(item.get("name", "")) for item in raw_hostapis]
    out: list[MmeDevice] = []
    for index, item in enumerate(raw_devices):
        hostapi_i = int(item.get("hostapi", -1))
        hostapi_name = (
            hostapi_names[hostapi_i]
            if 0 <= hostapi_i < len(hostapi_names)
            else ""
        )
        if hostapi_name != "MME":
            continue
        out.append(
            MmeDevice(
                index=int(index),
                name=str(item.get("name", "")),
                max_input_channels=int(item.get("max_input_channels", 0) or 0),
                max_output_channels=int(item.get("max_output_channels", 0) or 0),
                hostapi_name=hostapi_name,
            )
        )
    return out


def _name_matches(name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    folded = name.casefold()
    if not all(token.casefold() in folded for token in include):
        return False
    if any(token.casefold() in folded for token in exclude):
        return False
    return True


def _kind_ok(dev: MmeDevice, kind: str) -> bool:
    if kind == "output":
        return dev.max_output_channels > 0
    if kind == "input":
        return dev.max_input_channels > 0
    return False


def _filter_matches(
    devices: Sequence[MmeDevice],
    *,
    include: Iterable[str],
    exclude: Iterable[str],
    kind: str,
) -> list[MmeDevice]:
    return [
        dev
        for dev in devices
        if _kind_ok(dev, kind) and _name_matches(dev.name, include, exclude)
    ]


def _match_reason(
    *,
    profile: str,
    side: str,
    include: Iterable[str],
    matches: Sequence[MmeDevice],
) -> str | None:
    n = len(matches)
    needle = " + ".join(include)
    if n == 1:
        return None
    if n == 0:
        if profile == PROFILE_ZOOM and side == "in":
            return (
                "Voicemeeter Out B1 無し（Banana 未起動の可能性）。"
                "推測しない。一覧を見て Banana を起動してから再クエリ。"
            )
        return f"{side} 0件: MME に「{needle}」が無い。推測しない。"
    return f"{side} {n}件: MME の「{needle}」が曖昧。推測しない。"


def resolve_audio_route_profile(
    profile: str,
    *,
    devices: Sequence[MmeDevice] | None = None,
    query_devices: Callable[[], Any] | None = None,
    query_hostapis: Callable[[], Any] | None = None,
) -> ResolvedRoute:
    key = str(profile or "").strip()
    rules = _ROUTE_RULES.get(key)
    if rules is None:
        raise AudioRouteResolveError(
            f"unknown profile={profile!r} (allowed: {', '.join(_ROUTE_RULES)})",
            profile=str(profile),
            listing=[],
            out_matches=[],
            in_matches=[],
        )

    listing = (
        list(devices)
        if devices is not None
        else list_mme_devices(
            query_devices=query_devices,
            query_hostapis=query_hostapis,
        )
    )
    out_rules = rules["output"]
    in_rules = rules["input"]
    out_matches = _filter_matches(listing, **out_rules)
    in_matches = _filter_matches(listing, **in_rules)

    reasons = [
        reason
        for reason in (
            _match_reason(
                profile=key,
                side="out",
                include=out_rules["include"],
                matches=out_matches,
            ),
            _match_reason(
                profile=key,
                side="in",
                include=in_rules["include"],
                matches=in_matches,
            ),
        )
        if reason
    ]
    if reasons:
        raise AudioRouteResolveError(
            " / ".join(reasons),
            profile=key,
            listing=listing,
            out_matches=out_matches,
            in_matches=in_matches,
        )

    out_dev = out_matches[0]
    in_dev = in_matches[0]
    return ResolvedRoute(
        profile=key,
        label=str(rules["label"]),
        output=ResolvedEndpoint(
            index=out_dev.index,
            name=out_dev.name,
            kind="output",
        ),
        input=ResolvedEndpoint(
            index=in_dev.index,
            name=in_dev.name,
            kind="input",
        ),
    )


def argv_has_flag(argv: Sequence[str], flag: str) -> bool:
    for tok in argv:
        if tok == flag or tok.startswith(flag + "="):
            return True
    return False


def format_ok_log(resolved: ResolvedRoute) -> str:
    return "\n".join(
        [
            f"[audio_route][ok] profile={resolved.profile} label={resolved.label}",
            f"[audio_route][ok] out name={resolved.output.name!r} "
            f"index={resolved.output.index}",
            f"[audio_route][ok] in name={resolved.input.name!r} "
            f"index={resolved.input.index}",
        ]
    )


def apply_audio_route_profile_to_args(
    args: Any,
    *,
    argv: Sequence[str] | None = None,
    devices: Sequence[MmeDevice] | None = None,
) -> ResolvedRoute | None:
    profile = getattr(args, "audio_route_profile", None)
    if not profile:
        return None

    argv_list = list(argv if argv is not None else sys.argv)
    override_out = argv_has_flag(argv_list, "--audio_device")
    override_in = argv_has_flag(argv_list, "--mic_input_device")
    if override_out and override_in:
        print(
            "[audio_route][override] profile="
            f"{profile} audio_device={args.audio_device} "
            f"mic_input_device={args.mic_input_device} "
            "(CLI keep; name query skipped)",
            flush=True,
        )
        return None

    resolved = resolve_audio_route_profile(str(profile), devices=devices)
    print(format_ok_log(resolved), flush=True)

    if override_out:
        print(
            "[audio_route][override] audio_device kept from CLI "
            f"value={args.audio_device}",
            flush=True,
        )
    else:
        args.audio_device = int(resolved.output.index)

    if override_in:
        print(
            "[audio_route][override] mic_input_device kept from CLI "
            f"value={args.mic_input_device}",
            flush=True,
        )
    else:
        args.mic_input_device = int(resolved.input.index)

    if getattr(args, "ai_audio_output_device", None) is None:
        args.ai_audio_output_device = args.audio_device

    print(
        "[audio_route][bind] "
        f"audio_device={args.audio_device} "
        f"mic_input_device={args.mic_input_device} "
        f"ai_audio_output_device={args.ai_audio_output_device}",
        flush=True,
    )
    return resolved


def bind_audio_route_or_refuse(
    args: Any,
    *,
    argv: Sequence[str] | None = None,
    devices: Sequence[MmeDevice] | None = None,
) -> int:
    try:
        apply_audio_route_profile_to_args(args, argv=argv, devices=devices)
    except AudioRouteResolveError as exc:
        print(exc.format_refuse(), flush=True)
        return 2
    return 0
