#!/usr/bin/env python3
"""遍历 CEPB 站点的 Streamlit 多页面路径，收集所有页面文本与链接。"""
import json
import os
import re
import subprocess
import sys
import time

import requests

PORT = 4445
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_scan"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def wait_geckodriver(timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
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
        if not wait_geckodriver():
            log("geckodriver 没起来")
            return 1
        caps = {"capabilities": {"alwaysMatch": {
            "browserName": "firefox",
            "moz:firefoxOptions": {"args": ["-headless", "--width=1600", "--height=1400"]}}}}
        data = requests.post(f"{BASE}/session", json=caps, timeout=90).json()
        sid = data["value"]["sessionId"]
        log("会话:", sid)

        def post(path, payload, timeout=90):
            return requests.post(f"{BASE}/session/{sid}{path}", json=payload, timeout=timeout)

        candidates = ["/", "/Dataset", "/dataset", "/Data", "/Downloads",
                      "/Objects", "/Task", "/Downloads-and-Dataset", "/Download"]
        all_links = {}
        for path in candidates:
            url = "http://cepbbenchmark.eu" + path
            post("/url", {"url": url})
            time.sleep(6)
            text = post("/execute/sync", {"script": "return document.body.innerText||'';", "args": []}).json().get("value") or ""
            links = post("/execute/sync", {
                "script": "return [...document.querySelectorAll('a')].map(a=>((a.innerText||'').trim().replace(/\\n/g,' '))+' ||| '+a.href).join('\\n');",
                "args": []}).json().get("value") or ""
            nav = post("/execute/sync", {
                "script": "return [...document.querySelectorAll('[data-testid=\"stSidebarNav\"] a, nav a, [role=\"tab\"]')].map(a=>((a.innerText||'').trim())+' ||| '+(a.href||'')).join('\\n');",
                "args": []}).json().get("value") or ""
            sp = sorted(set(re.findall(r"https?://[^\s\"'<>]*(?:sharepoint|cameras)[^\s\"'<>]*", text + links + nav, re.I)))
            log(f"== {path} == 文本 {len(text)} 字符, 链接 {len(links.splitlines())} 条, 导航 {len(nav.splitlines())} 条, sharepoint {len(sp)}")
            safe = path.strip("/") or "root"
            open(f"{OUT}/{safe}_text.txt", "w", encoding="utf-8").write(text)
            open(f"{OUT}/{safe}_links.txt", "w", encoding="utf-8").write(links)
            if nav.strip():
                log("   导航:", nav.replace("\n", " ; ")[:400])
            if sp:
                for u in sp[:10]:
                    log("   SP:", u[:180])
            all_links[path] = links
            if len(text) > 300 and "404" not in text[:80]:
                log("   文本开头:", text[:200].replace("\n", " "))

        log("\n===== 汇总：所有 sharepoint 链接 =====")
        joined = "\n".join(all_links.values())
        for u in sorted(set(re.findall(r"https?://[^\s\"'<>]*sharepoint[^\s\"'<>]*", joined, re.I))):
            log(u[:220])
        log("结果目录:", OUT)
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
