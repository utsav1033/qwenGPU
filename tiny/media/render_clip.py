"""Renders clip.html to clip.mp4 (1920x1080, 30 fps) frame by frame, and a few stills.

Run: uv run --no-project --with playwright --with imageio-ffmpeg python tiny/media/render_clip.py
(needs Google Chrome installed)
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FPS, W, H = 30, 1920, 1080
stills = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.add_init_script("window.FRAME_MODE = true; window.FIXED_DPR = 1;")
    pg.goto((HERE / "clip.html").as_uri())
    pg.evaluate("window.ready")
    end = pg.evaluate("window.END")
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe",
                           "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "16", "-preset", "slow", "-movflags", "+faststart", str(HERE / "clip.mp4")],
                          stdin=subprocess.PIPE)
    for f in range(int(end * FPS) + 1):
        t = f / FPS
        pg.evaluate(f"window.render({t})")
        png = pg.screenshot(type="png")
        ff.stdin.write(png)
        if f % (2 * FPS) == 0 or f == int(end * FPS):
            (stills / f"clip-{t:04.1f}.png").write_bytes(png)
    ff.stdin.close()
    ff.wait()
    b.close()
print("wrote", HERE / "clip.mp4")
