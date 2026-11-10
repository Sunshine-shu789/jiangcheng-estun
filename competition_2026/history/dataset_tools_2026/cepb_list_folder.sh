#!/usr/bin/env bash
# 尝试用匿名分享会话列出 CEPB_dataset 文件夹，找到 cameras_parameters.yaml
set -u

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT=/tmp/cepb_folder
SHARE='https://alumnisssup-my.sharepoint.com/:u:/g/personal/salvatore_davella_santannapisa_it/IQAmKYbE4_OCTLRYRuEXIabmAQ0puyfcDih4-miKvTPLxfg?e=xsE2yH'
DRIVE='b!ojm4Csb_OkOpDIvDJpoHWnIzmxTa53hHnAXLghZ46KJF3apy4sOQT4chsIOOv-Ju'
ITEM='01ZFZG7ZZGFGDMJY7TQJGLIWCG4ELSDJXG'
API="https://alumnisssup-my.sharepoint.com/_api/v2.0/drives/$DRIVE/items/$ITEM"

rm -rf "$OUT"
mkdir -p "$OUT"

# 1. 建立匿名分享会话
curl -sS -L -m 40 -A "$UA" -c "$OUT/jar" -b "$OUT/jar" -o "$OUT/share.html" \
  -w 'share-page HTTP %{http_code}\n' "$SHARE" || true

call() {
  local name="$1" url="$2"
  printf '%-22s ' "$name"
  curl -sS -m 40 -A "$UA" -b "$OUT/jar" -c "$OUT/jar" \
    -H 'Accept: application/json' -H 'X-Requested-With: XMLHttpRequest' \
    -o "$OUT/$name.json" -w 'HTTP %{http_code} bytes=%{size_download}\n' "$url" || echo 'curl失败'
}

echo "===== OneDrive v2 匿名接口 ====="
call item "$API?version=Published"
call item_select "$API?version=Published&\$select=id,name,size,parentReference,folder,file,@content.downloadUrl"
call children "$API/children?version=Published"

python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
parent_id = None
for f in sorted(os.listdir(out)):
    if not f.endswith('.json'):
        continue
    body = open(os.path.join(out, f), encoding='utf-8', errors='replace').read()
    print(f'--- {f} ---')
    try:
        d = json.loads(body)
    except Exception as exc:
        print('  非 JSON:', body[:200].replace('\n', ' '))
        continue
    if 'error' in d:
        print('  error:', d['error'].get('code'), str(d['error'].get('message'))[:180])
        continue
    pr = d.get('parentReference') or {}
    if pr.get('id'):
        parent_id = pr['id']
    print('  name:', d.get('name'), '| size:', d.get('size'))
    if pr:
        print('  parent:', pr.get('id'), pr.get('path'))
    if 'value' in d:
        names = [(i.get('name'), i.get('size'), i.get('folder') is not None) for i in d['value']]
        for n in names[:40]:
            print('    child:', n)
open(os.path.join(out, 'parent_id.txt'), 'w').write(parent_id or '')
PY

PARENT=$(cat "$OUT/parent_id.txt" 2>/dev/null || true)
if [ -n "$PARENT" ]; then
  echo "===== 用父目录 id 列 children: $PARENT ====="
  call "parent_children" "https://alumnisssup-my.sharepoint.com/_api/v2.0/drives/$DRIVE/items/$PARENT/children?version=Published&\$top=200"
  python3 - "$OUT/parent_children.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding='utf-8', errors='replace'))
except Exception as exc:
    print('解析失败:', exc)
    raise SystemExit
if 'error' in d:
    print('error:', d['error'].get('code'), str(d['error'].get('message'))[:200])
else:
    for i in d.get('value', []):
        print('   ', i.get('name'), i.get('size'), 'folder' if i.get('folder') is not None else 'file', i.get('id'))
PY
else
  echo "没有拿到父目录 id"
fi

echo
echo "===== 备选：公开搜索引擎 ====="
for q in '%22cameras_parameters.yaml%22' 'cepbbenchmark+cameras+parameters'; do
  printf '%-40s ' "$q"
  curl -sS -L -m 25 -A "$UA" -o "$OUT/bing_$(printf '%s' "$q" | tr -cd 'a-zA-Z0-9').html" \
    -w 'HTTP %{http_code} bytes=%{size_download}\n' "https://www.bing.com/search?q=$q" || echo 'curl失败'
done
grep -ohE 'https?://[^" ]*cameras_parameters[^" ]*' "$OUT"/bing_*.html 2>/dev/null | sort -u | head -10

echo
echo "===== 备选：grep.app 代码搜索 ====="
curl -sS -L -m 25 -A "$UA" -o "$OUT/grepapp.json" \
  -w 'HTTP %{http_code} bytes=%{size_download}\n' \
  'https://grep.app/api/search?q=cameras_parameters.yaml' || true
head -c 800 "$OUT/grepapp.json" 2>/dev/null; echo

echo
echo "结果在 $OUT"
