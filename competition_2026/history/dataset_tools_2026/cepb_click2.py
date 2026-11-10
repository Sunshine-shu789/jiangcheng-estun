#!/usr/bin/env python3
"""用 WebDriver 原生点击 CEPB 站点 iframe 菜单里的 Dataset 项。"""
import json
import os
import re
import subprocess
import sys
import time

import requests

PORT = 4449
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_dataset_page2"
os.makedirs(OUT, exist_ok=True)
W3C = "element-6066-11e4-a52e-4f735466cecf"


def wait_gd(timeout=25):
    end = time.time() + timeout
    while time.time() < end:
        try:
            requests.get(f"{BASE}/status", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    gd = subprocess.Popen(["geckodriver", "--port", str(PORT), "--log", "fatal"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        if not wait_gd():
            print("geckodriver 启动失败")
            return 1
        caps = {"capabilities": {"alwaysMatch": {
            "browserName": "firefox",
            "moz:firefoxOptions": {"args": ["-headless", "--width=1600", "--height=1400"]}}}}
        sid = requests.post(f"{BASE}/session", json=caps, timeout=90).json()["value"]["sessionId"]
        print("会话:", sid)

        def post(path, payload=None, timeout=120):
            return requests.post(f"{BASE}/session/{sid}{path}", json=payload or {}, timeout=timeout)

        post("/url", {"url": "http://cepbbenchmark.eu/"})
        time.sleep(20)

        iframes = post("/elements", {"using": "css selector", "value": "iframe"}).json()["value"]
        print("iframe 数:", len(iframes))
        target = None
        for el in iframes:
            eid = el[W3C]
            src = post(f"/element/{eid}/attribute/src").json().get("value") or ""
            print("  ", src[:90])
            if "option_menu" in src:
                target = eid
        if not target:
            print("没找到菜单 iframe")
            return 1

        r = post("/frame", {"id": {W3C: target}})
        print("切进 iframe:", r.status_code)
        time.sleep(6)
        els = post("/elements", {"using": "xpath", "value": "//*[normalize-space(text())='Dataset']"}).json().get("value") or []
        print("Dataset 元素数:", len(els))
        if not els:
            els = post("/elements", {"using": "link text", "value": "Dataset"}).json().get("value") or []
            print("用 link text 再找:", len(els))
        if not els:
            print("菜单里没有 Dataset")
            return 1
        clicked = post(f"/element/{els[0][W3C]}/click")
        print("原生点击:", clicked.status_code, clicked.text[:200])
        post("/frame", {"id": None})
        time.sleep(20)

        text = post("/execute/sync", {"script": "return document.body.innerText||'';", "args": []}).json().get("value") or ""
        links = post("/execute/sync", {
            "script": "return [...document.querySelectorAll('a')].map(a=>((a.innerText||'').trim().replace(/\\n/g,' '))+' ||| '+a.href).join('\\n');",
            "args": []}).json().get("value") or ""
        html = post("/execute/sync", {"script": "return document.documentElement.outerHTML;", "args": []}).json().get("value") or ""
        open(f"{OUT}/text.txt", "w", encoding="utf-8").write(text)
        open(f"{OUT}/links.txt", "w", encoding="utf-8").write(links)
        open(f"{OUT}/page.html", "w", encoding="utf-8").write(html)
        print("URL:", post("/url").json().get("value"))
        print("---- 页面文本(前 3000) ----")
        print(text[:3000])
        print("---- 链接 ----")
        print(links[:3000])
        urls = sorted(set(re.findall(r"https?://[^\s\"'<>\\\\]+", html)))
        interesting = [u for u in urls if re.search(r"sharepoint|alumnisssup|camera|\.zip|subset", u, re.I)]
        print(f"---- 数据集相关链接 {len(interesting)} 条 ----")
        for u in interesting[:40]:
            print("   ", u[:250])
        return 0
    finally:
        try:
            if 'sid' in locals():
                requests.delete(f"{BASE}/session/{sid}", timeout=20)
        except Exception:
            pass
        gd.terminate()
        try:
            gd.wait(timeout=10)
        except Exception:
            gd.kill()


if __name__ == "__main__":
    sys.exit(main())
