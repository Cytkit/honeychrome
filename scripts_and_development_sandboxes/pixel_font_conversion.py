from PIL import BdfFontFile
from pathlib import Path

folder = Path('/home/ssr/Downloads/ucs-fonts')
files = list(folder.glob('*.bdf'))

for font_file_bdf in files:
    with open(font_file_bdf, 'rb') as fp:
        font = BdfFontFile.BdfFontFile(fp)
        font_file_pil = (folder / font_file_bdf.stem).with_suffix('.pil')
        font.save(font_file_pil)
