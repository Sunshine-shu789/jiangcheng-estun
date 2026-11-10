#!/usr/bin/env bash
# 一次性探测 CEPB 数据集文件（含 cameras_parameters.yaml）的匿名下载线索
set -u

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT=/tmp/cepb_probe
SHARE='https://alumnisssup-my.sharepoint.com/:u:/g/personal/salvatore_davella_santannapisa_it/IQAmKYbE4_OCTLRYRuEXIabmAQ0puyfcDih4-miKvTPLxfg?e=xsE2yH'

rm -rf "$OUT"
mkdir -p "$OUT"

echo "===== 1. cepbbenchmark.eu 路由探测 ====="
for p in "" "_stcore/health" "robots.txt" "sitemap.xml" "dataset" "Dataset" "datasets" "download"; do
  url="http://cepbbenchmark.eu/$p"
  name="site_$(printf '%s' "$p" | tr '/?=' '___')"
  printf '%-45s ' "$url"
  curl -sS -L -m 20 -A "$UA" -o "$OUT/$name.body" \
    -w 'HTTP %{http_code} bytes=%{size_download} type=%{content_type}\n' "$url" || echo 'curl失败'
done

echo
echo "===== 2. models.zip 分享页里暴露的元数据 ====="
curl -sS -L -m 30 -A "$UA" -c "$OUT/jar" -b "$OUT/jar" -o "$OUT/share.html" \
  -w 'HTTP %{http_code} bytes=%{size_download}\n' "$SHARE"
grep -oE 'https://[^"]*\.sharepoint\.com[^"]*' "$OUT/share.html" 2>/dev/null | sort -u | head -20
grep -oE '"(parentReference|webUrl|name)":"[^"]*"' "$OUT/share.html" 2>/dev/null | sort -u | head -20

echo
echo "===== 3. Graph 匿名 shares 接口 ====="
TOKEN=$(python3 -c "import base64,sys;print('u!'+base64.urlsafe_b64encode(sys.argv[1].encode()).decode().rstrip('='))" "$SHARE")
for suffix in "driveItem" "driveItem/children"; do
  printf '%-70s ' "/shares/.../$suffix"
  curl -sS -L -m 30 -A "$UA" -o "$OUT/graph_$(printf '%s' "$suffix" | tr '/' '_').json" \
    -w 'HTTP %{http_code} bytes=%{size_download}\n' \
    "https://graph.microsoft.com/v1.0/shares/$TOKEN/$suffix"
done
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for f in sorted(os.listdir(out)):
    if not f.startswith("graph_"):
        continue
    p = os.path.join(out, f)
    body = open(p, encoding="utf-8", errors="replace").read()
    print(f"--- {f} ---")
    try:
        d = json.loads(body)
        if "error" in d:
            print("  error:", d["error"].get("code"), str(d["error"].get("message"))[:200])
        else:
            pr = d.get("parentReference", {})
            print("  name:", d.get("name"), "| size:", d.get("size"))
            print("  webUrl:", d.get("webUrl"))
            print("  parent: driveId=", pr.get("driveId"), "path=", pr.get("path"))
            if "value" in d:
                print("  children:", [i.get("name") for i in d["value"]][:20])
    except Exception as exc:
        print("  解析失败:", exc, body[:200])
PY

echo
echo "===== 4. Wayback 存档 ====="
curl -sS -L -m 30 -A "$UA" -o "$OUT/wb_avail.json" \
  -w 'avail HTTP %{http_code}\n' 'http://archive.org/wayback/available?url=cepbbenchmark.eu'
cat "$OUT/wb_avail.json" | head -c 600; echo
curl -sS -L -m 40 -A "$UA" -o "$OUT/wb_cdx.txt" \
  -w 'cdx HTTP %{http_code} bytes=%{size_download}\n' \
  'http://web.archive.org/cdx/search/cdx?url=cepbbenchmark.eu*&output=text&limit=200&collapse=urlkey'
sort -u "$OUT/wb_cdx.txt" 2>/dev/null | awk '{print $3}' | sort -u | head -30

echo
echo "===== 5. GitHub 搜索 ====="
curl -sS -L -m 25 -A "$UA" -o "$OUT/gh_repos.json" \
  'https://api.github.com/search/repositories?q=cepb+cluttered+picking'
curl -sS -L -m 25 -A "$UA" -o "$OUT/gh_repos2.json" \
  'https://api.github.com/search/repositories?q=cepbbenchmark'
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
for f in ("gh_repos.json", "gh_repos2.json"):
    p = os.path.join(out, f)
    try:
        d = json.load(open(p, encoding="utf-8", errors="replace"))
        print(f, "total:", d.get("total_count"))
        for i in d.get("items", [])[:10]:
            print("   ", i.get("full_name"), "|", (i.get("description") or "")[:80])
    except Exception as exc:
        print(f, "解析失败:", exc)
PY

echo
echo "===== 6. 搜索结果里的 sharepoint 链接 ====="
grep -rhoE 'https://alumnisssup[^" <]*' "$OUT" 2>/dev/null | sort -u | head -20

echo
echo "探测完成，原始文件在 $OUT"
