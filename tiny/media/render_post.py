"""Records playground.html#post (the run card, 4:3) to post.mp4, 1440x1080, 30 fps, with a fake
clock so every frame is exact. Run: uv run --no-project --with playwright --with imageio-ffmpeg
python tiny/media/render_post.py [seconds]
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FPS, W, H = 30, 1440, 1080
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": W, "height": H}, color_scheme="dark")
    pg.clock.install()
    pg.goto((HERE / "playground.html").as_uri() + "#post")
    pg.evaluate("document.fonts.ready")
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe",
                           "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "17", "-preset", "slow", "-movflags", "+faststart", str(HERE / "post.mp4")],
                          stdin=subprocess.PIPE)
    for f in range(int(secs * FPS)):
        pg.clock.run_for(1000 // FPS)
        png = pg.screenshot(type="png")
        ff.stdin.write(png)
        if len(sys.argv) > 2 and f % 150 == 0:
            Path(sys.argv[2]).joinpath(f"post-{f}.png").write_bytes(png)
    ff.stdin.close(); ff.wait(); b.close()
print("wrote", HERE / "post.mp4")
