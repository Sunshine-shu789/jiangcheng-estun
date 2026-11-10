#!/usr/bin/env python3
"""点击 CEPB 站点菜单里的 Dataset 项，抓取数据集页面的全部链接。"""
import json
import os
import re
import subprocess
import sys
import time

import requests

PORT = 4448
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_dataset_page"
os.makedirs(OUT, exist_ok=True)


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

        def post(path, payload, timeout=120):
            return requests.post(f"{BASE}/session/{sid}{path}", json=payload, timeout=timeout)

        post("/url", {"url": "http://cepbbenchmark.eu/"})

        clicked = None
        for attempt in range(8):
            time.sleep(6)
            script = """
            const f=[...document.querySelectorAll('iframe')].find(f=>/option_menu/.test(f.src));
            if(!f||!f.contentDocument) return '';
            const doc=f.contentDocument;
            const el=[...doc.querySelectorAll('a,li')].find(e=>(e.innerText||'').trim()==='Dataset');
            if(!el) return 'menu-not-ready';
            const ev=new MouseEvent('click',{bubbles:true,cancelable:true,view:f.contentWindow});
            el.dispatchEvent(ev);
            if(el.click) el.click();
            return 'clicked:'+(el.innerText||'').trim();
            """
            clicked = post("/execute/sync", {"script": script, "args": []}).json().get("value")
            print(f"尝试 {attempt + 1}: {clicked}")
            if clicked and clicked.startswith("clicked"):
                break

        time.sleep(20)
        text = post("/execute/sync", {"script": "return document.body.innerText||'';", "args": []}).json().get("value") or ""
        links = post("/execute/sync", {
            "script": "return [...document.querySelectorAll('a')].map(a=>((a.innerText||'').trim().replace(/\\n/g,' '))+' ||| '+a.href).join('\\n');",
            "args": []}).json().get("value") or ""
        html = post("/execute/sync", {"script": "return document.documentElement.outerHTML;", "args": []}).json().get("value") or ""
        open(f"{OUT}/text.txt", "w", encoding="utf-8").write(text)
        open(f"{OUT}/links.txt", "w", encoding="utf-8").write(links)
        open(f"{OUT}/page.html", "w", encoding="utf-8").write(html)

        print("---- 页面文本 ----")
        print(text[:4000])
        print("---- 链接 ----")
        print(links[:4000])

        urls = sorted(set(re.findall(r"https?://[^\s\"'<>\\\\]+", html)))
        interesting = [u for u in urls if re.search(r"sharepoint|alumnisssup|camera|\.zip|subset|drive\.google", u, re.I)]
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
