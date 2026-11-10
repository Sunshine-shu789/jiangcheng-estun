#!/usr/bin/env python3
"""读取 CEPB 站点顶部 option-menu 组件里的菜单项，点击 Dataset 相关项并抓取内容。"""
import base64
import json
import os
import re
import subprocess
import sys
import time

import requests

PORT = 4446
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_menu"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


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
            log("geckodriver 启动失败")
            return 1
        caps = {"capabilities": {"alwaysMatch": {
            "browserName": "firefox",
            "moz:firefoxOptions": {"args": ["-headless", "--width=1600", "--height=1400"]}}}}
        data = requests.post(f"{BASE}/session", json=caps, timeout=90).json()
        sid = data["value"]["sessionId"]
        log("会话:", sid)

        def post(path, payload, timeout=120):
            return requests.post(f"{BASE}/session/{sid}{path}", json=payload, timeout=timeout)

        post("/url", {"url": "http://cepbbenchmark.eu/"})
        time.sleep(12)

        probe = """
        const frames=[...document.querySelectorAll('iframe')];
        return frames.map(f=>{
            let inner='';
            let items=[];
            try{
                inner=(f.contentDocument&&f.contentDocument.body)?f.contentDocument.body.innerText:'';
                if(f.contentDocument){
                    items=[...f.contentDocument.querySelectorAll('*')]
                        .filter(e=>e.children.length===0 && (e.innerText||'').trim())
                        .map(e=>e.innerText.trim());
                }
            }catch(e){inner='ERR:'+e;}
            return {src:f.src, items:items.slice(0,60), inner:inner.slice(0,600)};
        });
        """
        frames = post("/execute/sync", {"script": probe, "args": []}).json().get("value") or []
        log(f"页面 iframe 数量: {len(frames)}")
        for fr in frames:
            log("iframe:", fr.get("src", "")[:120])
            log("   items:", json.dumps(fr.get("items", []), ensure_ascii=False)[:800])

        click = """
        const frames=[...document.querySelectorAll('iframe')];
        for(const f of frames){
            let doc; try{doc=f.contentDocument;}catch(e){continue;}
            if(!doc) continue;
            const els=[...doc.querySelectorAll('*')].filter(e=>e.children.length===0);
            const target=els.find(e=>/dataset|数据/i.test((e.innerText||'').trim()));
            if(target){
                target.click();
                if(target.parentElement) target.parentElement.click();
                return (target.innerText||'').trim()+' @ '+(f.src||'').slice(0,80);
            }
        }
        return '';
        """
        clicked = post("/execute/sync", {"script": click, "args": []}).json().get("value") or ""
        log("点击结果:", clicked or "(没找到 Dataset 菜单项)")
        time.sleep(15)

        text = post("/execute/sync", {"script": "return document.body.innerText||'';", "args": []}).json().get("value") or ""
        links = post("/execute/sync", {
            "script": "return [...document.querySelectorAll('a')].map(a=>((a.innerText||'').trim().replace(/\\n/g,' '))+' ||| '+a.href).join('\\n');",
            "args": []}).json().get("value") or ""
        html = post("/execute/sync", {"script": "return document.documentElement.outerHTML;", "args": []}).json().get("value") or ""
        open(f"{OUT}/text.txt", "w", encoding="utf-8").write(text)
        open(f"{OUT}/links.txt", "w", encoding="utf-8").write(links)
        open(f"{OUT}/page.html", "w", encoding="utf-8").write(html)
        log("---- 点击后文本(前 2000 字) ----")
        log(text[:2000])
        log("---- 链接 ----")
        log(links[:3000])
        urls = sorted(set(re.findall(r"https?://[^\s\"'<>]+", html)))
        interesting = [u for u in urls if re.search(r"sharepoint|camera|\.zip|subset", u, re.I)]
        log(f"sharepoint/数据集相关链接 {len(interesting)} 条:")
        for u in interesting[:30]:
            log("   ", u[:220])
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
