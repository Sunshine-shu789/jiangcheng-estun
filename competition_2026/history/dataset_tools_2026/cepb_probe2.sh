#!/usr/bin/env bash
# 第二批探测：GitHub / Wayback / Jina / RGMC 官网
set -u

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT=/tmp/cepb_probe2
rm -rf "$OUT"; mkdir -p "$OUT"

echo "===== GitHub 搜索 ====="
curl -sS -L -m 25 -A "$UA" -o "$OUT/gh_repos_cepb.json" 'https://api.github.com/search/repositories?q=cepb'
curl -sS -L -m 25 -A "$UA" -o "$OUT/gh_repos_pic.json" 'https://api.github.com/search/repositories?q=%22picking+in+clutter%22'
curl -sS -L -m 25 -A "$UA" -o "$OUT/gh_users.json" 'https://api.github.com/search/users?q=davella+santanna'
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for f in sorted(os.listdir(out)):
    if not f.startswith("gh_"):
        continue
    try:
        d = json.load(open(os.path.join(out, f), encoding="utf-8", errors="replace"))
    except Exception as exc:
        print(f, "解析失败", exc); continue
    print(f"--- {f} total={d.get('total_count')}")
    for i in (d.get("items") or [])[:10]:
        print("   ", i.get("full_name") or i.get("login"), "|", (i.get("description") or "")[:70])
PY

echo
echo "===== Wayback（https）====="
curl -sS -L -m 40 -A "$UA" -o "$OUT/wb_cdx.txt" \
  -w 'cdx HTTP %{http_code} bytes=%{size_download}\n' \
  'https://web.archive.org/cdx/search/cdx?url=cepbbenchmark.eu&output=text&limit=100'
head -10 "$OUT/wb_cdx.txt" 2>/dev/null
SNAP=$(awk 'END{print $2}' "$OUT/wb_cdx.txt" 2>/dev/null)
if [ -n "${SNAP:-}" ]; then
  echo "抓取快照 $SNAP ..."
  curl -sS -L -m 60 -A "$UA" -o "$OUT/wb_page.html" \
    -w 'snapshot HTTP %{http_code} bytes=%{size_download}\n' \
    "https://web.archive.org/web/${SNAP}id_/http://cepbbenchmark.eu/"
  grep -oE 'https?://[^"'"'"'<> ]*(sharepoint|cameras|subset)[^"'"'"'<> ]*' "$OUT/wb_page.html" 2>/dev/null | sort -u | head -20
fi

echo
echo "===== Streamlit websocket 握手测试 ====="
curl -sS -i -m 15 -N \
  -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
  -H 'Sec-WebSocket-Version: 13' \
  -H 'Sec-WebSocket-Key: SGVsbG9Xb3JsZDEyMzQ1Njc4OTA=' \
  'https://cepbbenchmark.eu/_stcore/stream' 2>&1 | head -12

echo
echo "===== Jina 渲染代理 ====="
curl -sS -L -m 90 -A "$UA" -o "$OUT/jina.txt" \
  -w 'jina HTTP %{http_code} bytes=%{size_download}\n' \
  'https://r.jina.ai/http://cepbbenchmark.eu/'
grep -oE 'https?://[^) \t]+' "$OUT/jina.txt" 2>/dev/null | sort -u | grep -i -E 'sharepoint|camera|zip|dataset' | head -20

echo
echo "===== RGMC 官网（Google Sites）====="
curl -sS -L -m 40 -A "$UA" -o "$OUT/rgmcomp.html" \
  -w 'rgmcomp HTTP %{http_code} bytes=%{size_download}\n' \
  'https://sites.google.com/view/rgmcomp'
grep -oE 'https?://[^"'"'"'<> ]+' "$OUT/rgmcomp.html" 2>/dev/null | sort -u | \
  grep -i -E 'sharepoint|cepbbench|dataset|drive\.google|github' | head -20

echo
echo "===== 汇总 sharepoint 线索 ====="
grep -rhoE 'https://alumnisssup[^"'"'"'<> ]*' "$OUT" 2>/dev/null | sort -u | head -20
echo "结果目录 $OUT"
