"""Build the 30s Instagram reel (1080x1920) for the Kanoya lobby lounge.

Every photo is shown uncropped; motion comes from a slow scale of the whole
frame and a drifting blurred backdrop. Frames are piped to ffmpeg.

Usage: python3 tools/make_lobby_reel.py FONT_DIR OUT.mp4
FONT_DIR must contain NotoSansJP_300Light.ttf, NotoSansJP_400Regular.ttf,
NotoSerifJP_400Regular.ttf, CormorantGaramond_400Regular.ttf
(e.g. from the @expo-google-fonts npm packages).
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
TOTAL = 30.0
ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = Path(sys.argv[1])
OUT = sys.argv[2]

GOLD = (201, 168, 112)
WHITE = (246, 240, 230)

# Photo area: centred on PHOTO_CY, at most PHOTO_W wide / PHOTO_H tall.
PHOTO_W, PHOTO_H, PHOTO_CY = 1000, 1100, 830
TEXT_Y = 1500

# (file, english label, caption, seconds on screen incl. transition in, transition into this shot)
SHOTS = [
    ("7C1A5530.jpg", "LOBBY LOUNGE", "受付の先に、もうひとつの寛ぎを", 3.6, "fade_in"),
    ("7C1A5481.jpg", "PASSAGE", "灯りに導かれ、奥へ", 2.9, "shoji"),
    ("7C1A5480.jpg", "TSUBONIWA", "坪庭の緑が、静かに出迎える", 2.8, "walk"),
    ("7C1A5366.jpg", "CRAFTSMANSHIP", "職人の手仕事が、光を編む", 2.9, "push"),
    ("7C1A5528.jpg", "DETAIL", "一枚ごとに宿る、繊細な陰影", 2.8, "glow"),
    ("7C1A4396.jpg", "LOUNGE", "障子越しの灯りに、包まれて", 3.0, "sweep"),
    ("7C1A4392.jpg", "RELAX", "深く腰掛け、時を忘れる", 2.8, "blur"),
    ("7C1A4394.jpg", "WINDOW", "窓の向こうに、春日の森", 2.8, "walk"),
    ("7C1A4389.jpg", "UNWIND", "旅のあいまに、ひと息を", 3.0, "rise"),
]
END_DUR = TOTAL - sum(s[3] for s in SHOTS)
TRANS = 0.9  # transition length (s)


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_LABEL = font("NotoSansJP_400Regular.ttf", 26)
F_CAP = font("NotoSansJP_400Regular.ttf", 46)
F_LOGO = font("NotoSerifJP_400Regular.ttf", 74)
F_LOGO_EN = font("CormorantGaramond_400Regular.ttf", 34)
F_END = font("NotoSansJP_300Light.ttf", 34)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def affine(img, scale, cx, cy, size):
    """Render img scaled by `scale`, its centre placed at (cx, cy), sub-pixel accurate."""
    w, h = size
    iw, ih = img.size
    a = 1 / scale
    data = (a, 0, iw / 2 - a * cx, 0, a, ih / 2 - a * cy)
    return img.transform((w, h), Image.AFFINE, data, Image.BICUBIC)


class Shot:
    def __init__(self, fname, label, caption, dur, trans):
        src = Image.open(ROOT / "images" / fname).convert("RGB")
        fit = min(PHOTO_W / src.width, PHOTO_H / src.height)
        # Keep 1.1x headroom so the slow push stays sharp.
        self.photo = src.resize((round(src.width * fit * 1.1), round(src.height * fit * 1.1)), Image.LANCZOS)
        self.base = fit * 1.0 / (fit * 1.1)  # scale that shows the photo at its fitted size
        self.pw, self.ph = src.width * fit, src.height * fit
        cover = max(W / src.width, H / src.height) * 1.25
        bg = src.resize((round(src.width * cover / 4), round(src.height * cover / 4)), Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(14)).resize(
            (round(src.width * cover), round(src.height * cover)), Image.BICUBIC)
        # Normalise backdrop brightness so captions read the same on every shot.
        gain = 48 / max(np.asarray(bg, dtype=np.float32).mean(), 1)
        self.bg = Image.eval(bg, lambda v: int(min(v * gain, 255)))
        self.label, self.caption, self.dur, self.trans = label, caption, dur, trans
        self.text = text_layer(label, caption)

    def render(self, t, extra_scale=1.0, dx=0.0, dy=0.0, with_bg=True):
        """Return (bg, photo RGBA layer) at local time t seconds."""
        k = ease(t / (self.dur + TRANS))
        s = self.base * (1.0 + 0.045 * k) * extra_scale
        bg = None
        if with_bg:
            bg = affine(self.bg, 1.0 + 0.04 * k, W / 2 + 30 * (k - 0.5), H / 2 + dy * 0.3, (W, H))
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        photo = affine(self.photo, s, W / 2 + dx, PHOTO_CY + dy, (W, H))
        mask = affine(Image.new("L", self.photo.size, 255), s, W / 2 + dx, PHOTO_CY + dy, (W, H))
        layer.paste(photo, (0, 0), mask)
        # Hairline gold frame, offset outside the photo.
        fw, fh = self.pw * (s / self.base), self.ph * (s / self.base)
        d = ImageDraw.Draw(layer)
        x0, y0 = W / 2 + dx - fw / 2 - 14, PHOTO_CY + dy - fh / 2 - 14
        d.rectangle([x0, y0, x0 + fw + 28, y0 + fh + 28], outline=GOLD + (110,), width=1)
        return bg, layer


def spaced(draw, xy_center, text, fnt, fill, tracking):
    widths = [draw.textlength(c, font=fnt) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = xy_center[0] - total / 2
    for c, w in zip(text, widths):
        draw.text((x, xy_center[1]), c, font=fnt, fill=fill, anchor="lm")
        x += w + tracking


def text_layer(label, caption):
    layer = Image.new("RGBA", (W, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    spaced(d, (W / 2, 40), label, F_LABEL, GOLD + (255,), 10)
    d.line([(W / 2 - 22, 82), (W / 2 + 22, 82)], fill=GOLD + (170,), width=1)
    spaced(d, (W / 2, 140), caption, F_CAP, WHITE + (255,), 3)
    return layer


def place_text(canvas, shot, t):
    """Fade + slight rise in, fade out before the shot leaves."""
    a_in = ease_out((t - 0.35) / 0.8)
    a_out = 1 - ease((t - (shot.dur - 0.25)) / 0.55)
    a = max(0.0, min(a_in, a_out))
    if a <= 0:
        return
    rise = 18 * (1 - a_in)
    txt = shot.text.copy()
    txt.putalpha(txt.getchannel("A").point(lambda v: int(v * a)))
    canvas.alpha_composite(txt, (0, int(TEXT_Y - 40 + rise)))


def vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.62)) ** 2)
    v = np.clip(1.0 - 0.55 * np.clip(r - 0.55, 0, 1) ** 1.4, 0, 1)
    # Soft dark band behind the captions, as in the previous post.
    band = 1.0 - 0.38 * np.exp(-((y - (TEXT_Y + 30)) / 170) ** 2)
    return (v * band)[..., None]


VIG = vignette()


def compose(bg, layer):
    c = bg.convert("RGBA")
    c.alpha_composite(layer)
    return c


def blend(a, b, k):
    return Image.blend(a, b, k)


def frame_shot(shot, t, **kw):
    bg, layer = shot.render(t, **kw)
    c = compose(bg, layer)
    place_text(c, shot, t)
    return c


def transition(prev, cur, tp, tc, k, kind):
    """prev at its local time tp, cur at local time tc, progress k in 0..1."""
    e = ease(k)
    if kind == "shoji":
        # The lattice slides apart like shoji doors, revealing the next room.
        out = frame_shot(cur, tc)
        top = frame_shot(prev, tp)
        gap = int(e * W / 2)
        fade = int(255 * (1 - ease((k - 0.55) / 0.45)))
        if gap < W // 2:
            left = top.crop((gap, 0, W // 2, H))
            right = top.crop((W // 2, 0, W - gap, H))
            for piece in (left, right):
                piece.putalpha(piece.getchannel("A").point(lambda v: v * fade // 255))
            out.alpha_composite(left, (0, 0))
            out.alpha_composite(right, (W // 2 + gap, 0))
        return out
    if kind == "walk":
        # Step forward: outgoing grows and dissolves into the incoming.
        a = frame_shot(prev, tp, extra_scale=1 + 0.12 * e)
        b = frame_shot(cur, tc, extra_scale=0.94 + 0.06 * e)
        return blend(a, b, e)
    if kind == "push":
        # Gallery-style push: both frames glide left together.
        bga, la = prev.render(tp, dx=-W * 1.05 * e)
        bgb, lb = cur.render(tc, dx=W * 1.05 * (1 - e))
        out = blend(bga, bgb, e).convert("RGBA")
        out.alpha_composite(la)
        out.alpha_composite(lb)
        place_text(out, cur, tc)
        return out
    if kind == "glow":
        # Bloom to warm light, then settle on the next shot.
        a = frame_shot(prev, tp)
        b = frame_shot(cur, tc)
        mix = blend(a, b, ease((k - 0.3) / 0.4))
        g = math.sin(math.pi * k) * 0.55
        warm = Image.new("RGBA", (W, H), (255, 214, 160, 255))
        return blend(mix, warm, g * 0.6)
    if kind == "sweep":
        # A soft band of light wipes left-to-right.
        a = frame_shot(prev, tp)
        b = frame_shot(cur, tc)
        x = np.arange(W, dtype=np.float32)
        edge = -200 + (W + 400) * e
        m = np.clip((edge - x) / 200 + 0.5, 0, 1)
        mask = Image.fromarray((np.tile(m, (H, 1)) * 255).astype(np.uint8))
        out = Image.composite(b, a, mask)
        band = np.exp(-((x - edge) / 90) ** 2) * 0.35
        bandimg = Image.fromarray((np.tile(band, (H, 1)) * 255).astype(np.uint8))
        warm = Image.new("RGBA", (W, H), (255, 225, 180, 255))
        return Image.composite(warm, out, bandimg)
    if kind == "blur":
        a = frame_shot(prev, tp)
        b = frame_shot(cur, tc)
        r = math.sin(math.pi * k) * 10
        mix = blend(a, b, e)
        return mix.filter(ImageFilter.GaussianBlur(r)) if r > 0.3 else mix
    if kind == "rise":
        # Outgoing drifts up and fades; the portrait rises into place.
        out = blend(prev.render(tp)[0], cur.render(tc)[0], e).convert("RGBA")
        _, la = prev.render(tp, dy=-140 * e)
        la.putalpha(la.getchannel("A").point(lambda v: int(v * (1 - e))))
        _, lb = cur.render(tc, dy=160 * (1 - e))
        lb.putalpha(lb.getchannel("A").point(lambda v: int(v * e)))
        out.alpha_composite(la)
        out.alpha_composite(lb)
        place_text(out, cur, tc)
        return out
    raise ValueError(kind)


def end_card(shot_last, t_last, t):
    """Closing title over the asanoha lattice, deep and dim."""
    k = ease(t / END_DUR)
    bg = affine(END_BG, 1.0 + 0.05 * k, W / 2, H / 2, (W, H)).convert("RGBA")
    c = bg.copy()
    a = ease_out((t - 0.5) / 1.0)
    if a > 0:
        txt = END_TEXT.copy()
        txt.putalpha(txt.getchannel("A").point(lambda v: int(v * a)))
        c.alpha_composite(txt, (0, int(6 * (1 - a))))
    if t < TRANS:
        prev = frame_shot(shot_last, t_last)
        c = blend(prev, c, ease(t / TRANS))
    return c


def build_end_text():
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cy = 900
    spaced(d, (W / 2, cy - 160), "NARA KASUGA", F_LOGO_EN, GOLD + (255,), 12)
    spaced(d, (W / 2, cy - 40), "奈良春日 鹿のや", F_LOGO, WHITE + (255,), 10)
    d.line([(W / 2 - 60, cy + 70), (W / 2 + 60, cy + 70)], fill=GOLD + (200,), width=1)
    spaced(d, (W / 2, cy + 150), "ロビーラウンジで、ごゆるりと", F_END, WHITE + (235,), 4)
    spaced(d, (W / 2, cy + 230), "AUBERGE  ·  5 ROOMS", F_LABEL, GOLD + (220,), 8)
    return layer


def main():
    global END_BG, END_TEXT
    shots = [Shot(*s) for s in SHOTS]
    lattice = Image.open(ROOT / "images" / "7C1A5530.jpg").convert("RGB")
    cover = max(W / lattice.width, H / lattice.height) * 1.1
    eb = lattice.resize((round(lattice.width * cover / 3), round(lattice.height * cover / 3)), Image.LANCZOS)
    eb = eb.filter(ImageFilter.GaussianBlur(5)).resize(
        (round(lattice.width * cover), round(lattice.height * cover)), Image.BICUBIC)
    END_BG = Image.eval(eb, lambda v: int(v * 0.22))
    END_TEXT = build_end_text()

    starts = np.cumsum([0] + [s.dur for s in shots])
    n = int(TOTAL * FPS)
    ff = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k", OUT,
    ], stdin=subprocess.PIPE)
    rng = np.random.default_rng(7)
    for f in range(n):
        T = f / FPS
        i = int(np.searchsorted(starts, T, side="right") - 1)
        if i >= len(shots):
            img = end_card(shots[-1], T - starts[-2], T - starts[-1])
        else:
            shot, t = shots[i], T - starts[i]
            if shot.trans == "fade_in":
                img = frame_shot(shot, t)
                if t < 1.2:
                    img = blend(Image.new("RGBA", (W, H), (0, 0, 0, 255)), img, ease(t / 1.2))
            elif t < TRANS:
                prev = shots[i - 1]
                img = transition(prev, shot, T - starts[i - 1], t, t / TRANS, shot.trans)
            else:
                img = frame_shot(shot, t)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) * VIG
        arr += rng.normal(0, 1.1, (H, W, 1)).astype(np.float32)  # fine film grain
        if T > TOTAL - 0.6:
            arr *= ease((TOTAL - T) / 0.6)
        ff.stdin.write(np.clip(arr, 0, 255).astype(np.uint8).tobytes())
        if f % 60 == 0:
            print(f"{f}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
