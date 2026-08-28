"""Frontend consistency guard: catches silent replace failures between
index.html and app.js (the retrospective incident where the picks-page
rewrite silently no-op'd and users saw a stale UI)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "server" / "static" / "index.html").read_text(encoding="utf-8")
JS = (ROOT / "server" / "static" / "app.js").read_text(encoding="utf-8")
CSS = (ROOT / "server" / "static" / "style.css").read_text(encoding="utf-8")


@pytest.mark.unit
class TestFrontendConsistency:
    def test_asset_versions_in_sync(self):
        css_v = re.search(r"style\.css\?v=([\w]+)", HTML)
        js_v = re.search(r"app\.js\?v=([\w]+)", HTML)
        assert css_v and js_v, "index.html must reference versioned assets"
        assert css_v.group(1) == js_v.group(1), "css/js 版本号不一致"

    def test_ids_referenced_by_js_exist_in_html_or_js_templates(self):
        # JS 模板字符串里动态创建的 id 也算存在（全文合并提取）
        combined = HTML + JS
        html_ids = set(re.findall(r'id="([\w-]+)"', combined))
        js_refs = set(re.findall(r'\$\("#([\w-]+)"\)', JS))
        missing = sorted(js_refs - html_ids)
        assert not missing, f"app.js 引用了不存在的元素 id: {missing}"

    def test_picks_page_functions_present(self):
        for fn in ("loadPicks", "startScreening", "cancelScreening",
                   "pollScreening", "renderScreenStatus", "renderScreenHistory",
                   "deepResearch", "showToast", "STAGE_HINTS"):
            assert fn in JS, f"app.js 缺少 {fn}"

    def test_picks_page_dom_present(self):
        for el in ('id="screen-status"', 'id="ss-stop"', 'id="ss-bar-wrap"',
                   'id="screen-history"', 'id="picks-list"', 'id="screen-meta"',
                   'data-view="picks"', 'id="s-autoscreen"'):
            assert el in HTML, f"index.html 缺少 {el}"

    def test_css_for_picks_components_present(self):
        for cls in (".screen-status", ".spinner", ".pick-card", ".toast",
                    ".mini-bar.indeterminate", ".sh-table"):
            assert cls in CSS, f"style.css 缺少 {cls}"

    def test_no_stale_version_strings(self):
        assert "20260827" not in HTML, "index.html 仍引用旧版本资源"
