#!/usr/bin/env python3
"""把 streamlit_option_menu 组件的 iframe 内容整个 dump 出来，看看菜单项到底有没有渲染。"""
import json
import os
import subprocess
import sys
import time

import requests

PORT = 4447
BASE = f"http://127.0.0.1:{PORT}"
OUT = "/tmp/cepb_menu_dump"
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
        time.sleep(25)

        dump = """
        const out=[];
        for(const f of document.querySelectorAll('iframe')){
            let d=null; try{d=f.contentDocument;}catch(e){}
            if(!d){out.push({src:f.src, err:'cross-origin'});continue;}
            const info={
                src:f.src,
                htmlLen:(d.documentElement.outerHTML||'').length,
                text:(d.body?d.body.innerText:'').slice(0,800),
                html:(d.documentElement.outerHTML||'').slice(0,3000),
                counts:{}
            };
            for(const sel of ['a','button','div','span','li','img','svg']){
                info.counts[sel]=d.querySelectorAll(sel).length;
            }
            out.push(info);
        }
        return out;
        """
        frames = post("/execute/sync", {"script": dump, "args": []}).json().get("value") or []
        for i, fr in enumerate(frames):
            print(f"===== iframe {i}: {fr.get('src','')[:100]}")
            print("元素计数:", fr.get("counts"))
            print("文本:", (fr.get("text") or "")[:400].replace("\n", " | "))
            open(f"{OUT}/iframe{i}.html", "w", encoding="utf-8").write(fr.get("html") or "")
            print("HTML 片段:", (fr.get("html") or "")[:1200].replace("\n", " "))
            print()

        # 试着把 iframe 内部的按钮全部点一遍，看会不会出现新内容
        clickall = """
        const f=[...document.querySelectorAll('iframe')].find(f=>/option_menu/.test(f.src));
        if(!f||!f.contentDocument) return 'no-frame';
        const d=f.contentDocument;
        const cands=[...d.querySelectorAll('a,button,div[role=button],li')];
        const names=cands.map(e=>(e.innerText||'').trim()).filter(Boolean);
        return JSON.stringify(names.slice(0,40));
        """
        names = post("/execute/sync", {"script": clickall, "args": []}).json().get("value")
        print("候选菜单项:", names)
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
