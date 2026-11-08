"""Cuts the notebook's tiles and their annotations from local copies of the datasets and caches them.

fetch_tiles(specs, data_dir, datasets) takes the locations of the datasets as a dict (README.md says where to get them):
  'tcga_slides'   TCGA-BRCA diagnostic slides, <slide id>.svs
  'bcss'          BCSS: masks/, rgbs_raw/ (region images without colour normalisation), meta/roiBounds_BaseMagnification.csv
  'nucls'         NuCLS: csv/, rgb/
  'beetle'        BEETLE: annotations.zip

Each tile is cached in data_dir as {name}.png, {name}_annotation.json and, for BCSS tiles, {name}_mask.npy, all at 448 x 448.
The annotation holds outlines in tile pixels, empty where the tile's dataset has none:
  single lobular cells   BEETLE invasive tumour outlines (label 3) of at most 200 µm² whose centre lies inside the tile
  tumour outlines        every BEETLE invasive tumour outline touching the tile, single cells and clusters
  ... nuclei             NuCLS nuclei whose centre lies inside the tile, by kind (tumour, lymphocyte, fibroblast, other)
A BCSS mask holds 0 tumour, 1 stroma, 2 immune, 3 necrosis, 4 fat, 5-8 other annotated tissue and -1 not annotated.
"""
import io, json, math, re, zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None          # BCSS region images are larger than PIL's default limit


def read_bcss(name, cx, cy, size_um, region_rgb, region_mask, bounds, fields, nucls_dir, data_dir):
    """A BCSS tile cut from its region image, its region mask, and the NuCLS nuclei inside it.
    (cx, cy) is the tile centre in region pixels (0.30 µm/px), bounds the region's level-0 box on the slide,
    fields the NuCLS fields of view of the region as (name, left, top, right, bottom) in level-0 pixels."""
    # BCSS pixel codes -> 0 tumour, 1 stroma, 2 immune, 3 necrosis, 4 fat, 5-8 other annotated tissue, -1 not annotated
    lut = np.full(256, -1)
    for code, merged in {1: 0, 2: 1, 3: 2, 10: 2, 11: 2, 4: 3, 9: 4, 13: 5, 20: 6, 6: 7, 14: 7, 18: 7, 19: 7, 5: 8, 8: 8, 12: 8, 16: 8, 17: 8, 21: 8}.items():
        lut[code] = merged
    # NuCLS super-classes -> annotation keys; all other classes are 'other nuclei'
    kinds = {'tumor_any': 'tumour nuclei', 'sTIL': 'lymphocyte nuclei', 'nonTIL_stromal': 'fibroblast nuclei'}

    half = round(size_um / 0.30 / 2, 6)
    box = tuple(round(v, 6) for v in (cx - half, cy - half, cx + half, cy + half))
    np.save(data_dir / f'{name}_mask.npy', lut[np.asarray(region_mask.resize((448, 448), Image.Resampling.NEAREST, box=box))])

    # NuCLS nuclei whose centre lies in the tile's level-0 box, outlines converted to tile pixels
    scale = region_mask.size[0] / (bounds.xmax - bounds.xmin)                 # region px per level-0 px
    x0, y0 = bounds.xmin + (cx - half) / scale, bounds.ymin + (cy - half) / scale
    x1, y1 = x0 + 2 * half / scale, y0 + 2 * half / scale
    annotation = {'single lobular cells': [], 'tumour outlines': [], 'tumour nuclei': [], 'lymphocyte nuclei': [], 'fibroblast nuclei': [], 'other nuclei': []}
    for fov, left, top, right, bottom in fields:
        if right <= x0 or left >= x1 or bottom <= y0 or top >= y1:
            continue
        k = (right - left) / Image.open(nucls_dir / 'rgb' / f'{fov}.png').size[0]   # level-0 px per field px
        for n in pd.read_csv(nucls_dir / 'csv' / f'{fov}.csv').itertuples():
            if n.type == 'rectangle':
                poly = [[n.xmin, n.ymin], [n.xmax, n.ymin], [n.xmax, n.ymax], [n.xmin, n.ymax]]
            else:
                poly = list(zip(map(float, str(n.coords_x).split(',')), map(float, str(n.coords_y).split(','))))
            pts = np.asarray(poly, float) * k + [left, top]
            centre = pts.mean(axis=0)
            if x0 <= centre[0] < x1 and y0 <= centre[1] < y1:
                in_tile = ((pts - [bounds.xmin, bounds.ymin]) * scale - [cx - half, cy - half]) * 448 / (2 * half)
                annotation[kinds.get(n.super_classification, 'other nuclei')].append(in_tile.round(1).tolist())
    (data_dir / f'{name}_annotation.json').write_text(json.dumps(annotation))
    region_rgb.resize((448, 448), Image.Resampling.BILINEAR, box=box).save(data_dir / f'{name}.png')


def read_slide_tile(slide, x, y, size_um):
    """A tile of size_um from a slide, top-left at level-0 pixel (x, y), resampled to 448 x 448."""
    side = size_um / float(slide.properties['openslide.mpp-x'])
    ix, iy = math.floor(x), math.floor(y)
    patch = slide.read_region((ix, iy), 0, (math.ceil(side + x - ix) + 1, math.ceil(side + y - iy) + 1)).convert('RGB')
    fx, fy = x - ix, y - iy
    tile = patch.transform((448, 448), Image.Transform.EXTENT, (fx, fy, fx + side, fy + side), resample=Image.Resampling.BILINEAR)
    return Image.fromarray(np.asarray(tile))           # pixels only: some slides attach a large ICC profile that PIL cannot read back


def read_beetle(name, slide_id, x, y, size_um, slides_dir, beetle_dir, data_dir):
    """A tile read from its TCGA slide, top-left at level-0 pixel (x, y), and BEETLE's invasive tumour outlines touching it."""
    slide = openslide.OpenSlide(str(slides_dir / f'{slide_id}.svs'))
    mpp = float(slide.properties['openslide.mpp-x'])
    side = size_um / mpp
    zf = zipfile.ZipFile(beetle_dir / 'annotations.zip')
    mask_w = tifffile.TiffFile(io.BytesIO(zf.read(f'annotations/masks/{slide_id}.tif'))).pages[0].shape[1]
    scale = slide.dimensions[0] / mask_w                                    # level-0 px per annotation px
    annotation = {'single lobular cells': [], 'tumour outlines': [], 'tumour nuclei': [], 'lymphocyte nuclei': [], 'fibroblast nuclei': [], 'other nuclei': []}
    for entry in json.loads(zf.read(f'annotations/jsons/{slide_id}.json')):
        poly = np.asarray(entry['coordinates'], float) * scale
        pts = (poly - [x, y]) / side * 448                                  # tile pixels
        touches = pts[:, 0].max() >= 0 and pts[:, 0].min() <= 448 and pts[:, 1].max() >= 0 and pts[:, 1].min() <= 448
        if int(entry['label']['value']) != 3 or not touches:
            continue
        pts = pts.round(1)
        area = round(float(0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))) * mpp ** 2), 1)
        centre = pts.mean(axis=0)
        annotation['tumour outlines'].append(pts.tolist())
        if area <= 200 and centre.min() >= 0 and centre.max() < 448:
            annotation['single lobular cells'].append(pts.tolist())
    (data_dir / f'{name}_annotation.json').write_text(json.dumps(annotation))
    read_slide_tile(slide, x, y, size_um).save(data_dir / f'{name}.png')


def fetch_tiles(specs, data_dir, datasets, desc='tiles'):
    """Cut every tile of specs (name -> (source, slide id, x, y, size in µm)) that is not cached in data_dir yet.
    bcss: (x, y) is the tile centre in the BCSS region image. beetle: (x, y) is the top-left level-0 pixel of the slide.
    The image is written last, so a cached image means its annotation is cached too."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted((spec[1], name) for name, spec in specs.items() if not (data_dir / f'{name}.png').exists())   # in order of slide
    if not missing:
        return

    # BCSS region bounds and NuCLS fields of view, read once: field name -> (region, left, top, right, bottom) in level-0 pixels
    bounds = pd.read_csv(datasets['bcss'] / 'meta/roiBounds_BaseMagnification.csv', index_col=0)
    fields = []
    for f in sorted((datasets['nucls'] / 'csv').glob('*.csv')):
        m = re.match(r'((TCGA-..-....-DX1)_id-.*_left-(\d+)_top-(\d+)_bottom-(\d+)_right-(\d+))\.csv$', f.name)
        if m:
            fields.append((m[1], m[2], int(m[3]), int(m[4]), int(m[6]), int(m[5])))

    region = None                       # the BCSS region open now; tiles come in order of slide, so each region is opened once
    for _, name in tqdm(missing, desc=desc, unit='tile', mininterval=5):
        source, slide_id, x, y, size_um = specs[name]
        if source == 'bcss':
            if region != f'{slide_id[:12]}-DX1':
                region = f'{slide_id[:12]}-DX1'
                mask_file = next((datasets['bcss'] / 'masks').glob(f'{region}_xmin*.png'))
                region_rgb = Image.open(datasets['bcss'] / 'rgbs_raw' / mask_file.name).convert('RGB')
                region_mask = Image.open(mask_file)
                region_fields = [(fov, left, top, right, bottom) for fov, fov_region, left, top, right, bottom in fields if fov_region == region]
            read_bcss(name, x, y, size_um, region_rgb, region_mask, bounds.loc[region], region_fields, datasets['nucls'], data_dir)
        elif source == 'beetle':
            read_beetle(name, slide_id, x, y, size_um, datasets['tcga_slides'], datasets['beetle'], data_dir)
        else:
            raise ValueError(f'{name}: unknown source {source!r}')
