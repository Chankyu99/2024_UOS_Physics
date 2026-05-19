"""
Crop Fig 1, Fig 3, Fig 4, Fig 5 from extracted paper pages.

Page 200 dpi -> 1700 x 2200 px. Approx panel boxes determined visually.
Output: img/paper/fig{N}.png
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "img" / "paper"

# (page_file, box (left, top, right, bottom), out_name)
JOBS = [
    ("page2-02.png", (820, 70, 1640, 920),   "fig1.png"),     # Fig 1
    ("page6-06.png", (110, 110, 1660, 1180), "fig3.png"),     # Fig 3 (all panels)
    ("page6-06.png", (110, 110, 1660, 580),  "fig3_top.png"), # Fig 3 (a)-(h)
    ("page6-06.png", (110, 560, 1660, 1180), "fig3_bottom.png"),  # Fig 3 (i)(j)
    ("page7-07.png", (90, 130, 1660, 1170),  "fig4.png"),     # Fig 4 (a)-(j)
    ("page8-08.png", (90, 130, 1660, 1300),  "fig5.png"),     # Fig 5 (a)-(g)
    ("page8-08.png", (90, 130, 1660, 600),   "fig5_top.png"), # Fig 5 (a)-(d)
]


def main():
    for page, box, out in JOBS:
        src = PAPER / page
        im = Image.open(src)
        crop = im.crop(box)
        crop.save(PAPER / out)
        print(f"{out}: {crop.size}")


if __name__ == "__main__":
    main()
