#!/usr/bin/env python3
"""用无头 firefox(geckodriver) 渲染 cepbbenchmark.eu，提取数据集下载链接并尝试下载相机参数文件。"""
import json
import os
import re
import subprocess
import sys
import time

import requests

PORT = 4444
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_render"
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
    gd = subprocess.Popen(
        ["geckodriver", "--port", str(PORT), "--log", "fatal"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_geckodriver():
            log("geckodriver 没起来:", (gd.stdout.read() or "")[:2000] if gd.stdout else "")
            return 1
        log("geckodriver 就绪")

        caps = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "firefox",
                    "moz:firefoxOptions": {"args": ["-headless", "--width=1600", "--height=1200"]},
                }
            }
        }
        r = requests.post(f"{BASE}/session", json=caps, timeout=90)
        data = r.json()
        if "value" not in data or "sessionId" not in data.get("value", {}):
            log("创建会话失败:", json.dumps(data)[:1500])
            return 1
        sid = data["value"]["sessionId"]
        log("firefox 会话:", sid)

        def post(path, payload, timeout=90):
            return requests.post(f"{BASE}/session/{sid}{path}", json=payload, timeout=timeout)

        post("/url", {"url": "http://cepbbenchmark.eu/"})
        log("已打开页面，等待渲染 ...")

        script_text = "return document.body ? document.body.innerText : '';"
        text = ""
        for _ in range(20):
            time.sleep(2)
            try:
                text = post("/execute/sync", {"script": script_text, "args": []}).json()["value"] or ""
            except Exception:
                continue
            if len(text) > 200:
                break
        log(f"页面文本长度: {len(text)}")
        open(f"{OUT}/page_text.txt", "w", encoding="utf-8").write(text)
        log("---- 页面文本(前 1500 字) ----")
        log(text[:1500])

        # 尝试点击 Dataset/数据集 标签
        click_script = """
        const els=[...document.querySelectorAll('a,button,[role=tab],li')];
        const t=els.find(e=>/dataset|数据集|data set/i.test((e.innerText||'').trim()));
        if(t){t.click();return (t.innerText||'').trim();}
        return '';
        """
        clicked = post("/execute/sync", {"script": click_script, "args": []}).json().get("value")
        if clicked:
            log("点击了标签:", clicked)
            time.sleep(6)
            text = post("/execute/sync", {"script": script_text, "args": []}).json()["value"] or ""
            open(f"{OUT}/page_text_after_click.txt", "w", encoding="utf-8").write(text)
            log(f"点击后文本长度: {len(text)}")
            log(text[:1500])

        link_script = """
        return [...document.querySelectorAll('a')].map(a=>((a.innerText||'').trim().replace(/\\n/g,' '))+' ||| '+a.href).join('\\n');
        """
        links = post("/execute/sync", {"script": link_script, "args": []}).json().get("value") or ""
        open(f"{OUT}/links.txt", "w", encoding="utf-8").write(links)
        log("---- 链接 ----")
        log(links[:4000])

        html = post("/execute/sync", {"script": "return document.documentElement.outerHTML;", "args": []}).json().get("value") or ""
        open(f"{OUT}/page.html", "w", encoding="utf-8").write(html)
        urls = sorted(set(re.findall(r"https?://[^\s\"'<>]+", html)))
        interesting = [u for u in urls if "sharepoint" in u or "cameras" in u.lower()]
        log(f"页面里 sharepoint/相机 相关链接 {len(interesting)} 条:")
        for u in interesting[:30]:
            log("   ", u[:200])

        # 找 cameras_parameters.yaml 的链接并下载
        target = None
        for line in links.splitlines():
            name, _, href = line.partition("|||")
            if "camera" in name.lower() and ("yaml" in name.lower() or "param" in name.lower()):
                target = href.strip()
                break
        if not target:
            for u in urls:
                if "camera" in u.lower() and "yaml" in u.lower():
                    target = u
                    break
        if target:
            log("找到相机参数链接:", target[:200])
            try:
                resp = requests.get(target, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                log("下载 HTTP", resp.status_code, "字节数", len(resp.content))
                if resp.ok and len(resp.content) > 100:
                    open(f"{OUT}/cameras_parameters.yaml", "wb").write(resp.content)
                    log("已保存到", f"{OUT}/cameras_parameters.yaml")
            except Exception as exc:
                log("下载失败:", exc)
        else:
            log("页面链接里没找到 cameras_parameters.yaml")
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
