from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_ACTIONS = {
    "set_background",
    "play_sound",
    "set_energy_level",
    "show_barrage",
    "set_timer_state",
    "screen_effect",
}


@dataclass
class OBSRouterConfig:
    mode: str = "dry_run"  # dry_run | websocket | html_overlay_ws
    log_jsonl_path: Optional[str] = None
    websocket_url: Optional[str] = None


class OBSActionRouter:
    """
    obs_actions.json の action + params を受け取り、
    mode に応じて dry_run / websocket / html_overlay_ws へ振り分ける。
    まずは dry_run を本体とし、他モードは stub で残す。
    """

    def __init__(self, cfg: OBSRouterConfig) -> None:
        self.cfg = cfg

    def dispatch_all(self, actions: List[Dict[str, Any]]) -> None:
        for action in actions:
            self.dispatch(action)

    def dispatch(self, action_obj: Dict[str, Any]) -> None:
        action = str(action_obj.get("action", "") or "")
        params = action_obj.get("params", {}) or {}

        if not isinstance(params, dict):
            params = {}

        if action not in ALLOWED_ACTIONS:
            self._emit_log(
                {
                    "kind": "obs_action_ignored",
                    "reason": "unknown_action",
                    "action": action,
                    "params": params,
                }
            )
            return

        if self.cfg.mode == "dry_run":
            self._dispatch_dry_run(action, params)
            return

        if self.cfg.mode == "websocket":
            self._dispatch_websocket(action, params)
            return

        if self.cfg.mode == "html_overlay_ws":
            self._dispatch_html_overlay_ws(action, params)
            return

        raise ValueError(f"unsupported obs router mode: {self.cfg.mode}")

    def _dispatch_dry_run(self, action: str, params: Dict[str, Any]) -> None:
        self._emit_log(
            {
                "kind": "obs_action_dry_run",
                "action": action,
                "params": params,
            }
        )

    def _dispatch_websocket(self, action: str, params: Dict[str, Any]) -> None:
        """
        将来:
        - obs-websocket へ command 変換して送る
        """
        self._emit_log(
            {
                "kind": "obs_action_websocket_stub",
                "action": action,
                "params": params,
                "websocket_url": self.cfg.websocket_url,
            }
        )

    def _dispatch_html_overlay_ws(self, action: str, params: Dict[str, Any]) -> None:
        """
        将来:
        - HTML5/Canvas overlay 用 websocket サーバへ送る
        """
        self._emit_log(
            {
                "kind": "obs_action_html_overlay_ws_stub",
                "action": action,
                "params": params,
                "websocket_url": self.cfg.websocket_url,
            }
        )

    def _emit_log(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        print(line)

        if self.cfg.log_jsonl_path:
            p = Path(self.cfg.log_jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")