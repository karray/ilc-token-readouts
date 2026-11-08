"""Reproduce the heads and every number that findings.ipynb shows as a table or quotes in its text.

Set the paths in DATA below, then run (one GPU with 8 GB, a few hours; every step is cached in CACHE):

    uv run python reproduce.py

The script reads every tile once from the public datasets, encodes it with five pathology foundation models, trains the
ductal head, the lobular head and a class-token head with five hospital folds, and reads every tile with the fold that
did not see its hospital. It writes:

    OUT/linear_heads.pt   the notebook's heads (standardisation folded into W and b) and H-optimus-0's mean token per position
    OUT/bank_tokens.csv   the notebook's bank of reliable tokens (section 6)
    OUT/examples.csv      the example tiles of section 5
    OUT/tables.md         the tables of sections 2 and 9, and the numbers quoted in the text
"""
import io
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import tifffile
import timm
import torch
from matplotlib.path import Path as Polygon
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from timm.layers import SwiGLUPacked
from tqdm import tqdm
from transformers import Dinov2Model, Dinov2WithRegistersModel

# ------------------------------------------------------------------------------------------------------------------- data
DATA = Path.home() / 'Projects/data'
TCGA_SLIDES = DATA / 'tcga_brca_hoptimus0/slides'                # TCGA-BRCA diagnostic slides, <slide id>.svs
MANIFEST = DATA / 'tcga_brca_hoptimus0/slide_manifest.csv'       # the panel's slide table: patient_id, slide_id, label
BCSS = DATA / 'bcss'                                             # BCSS: masks/, rgbs_raw/, meta/
NUCLS = DATA / 'nucls/NuCLS_dataset'                             # NuCLS: csv/, rgb/
BEETLE = DATA / 'beetle'                                         # BEETLE: annotations.zip, wsis/<Radboudumc slide>.tif
OUTLINES = DATA / 'tcga_brca_tumour_masks/coarse.zip'            # coarse tumour outlines (Banerjee et al. 2025)
CACHE, OUT = Path('reproduction/cache'), Path('reproduction')

# ----------------------------------------------------------------------------------------------------------------- design
ENCODERS = ['H-optimus-0', 'UNI2-h', 'Virchow2', 'Hibou-B', 'Phikon-v2']
TISSUES = ['tumour', 'stroma', 'immune', 'necrosis', 'fat']                       # the ductal head
S6 = ['lobular tumour', 'ductal tumour', 'stroma', 'immune', 'necrosis', 'fat']  # the lobular head
BCSS_CODES = {1: 0, 2: 1, 3: 2, 10: 2, 11: 2, 4: 3, 9: 4, 13: 5, 20: 6, 6: 7, 14: 7, 18: 7, 19: 7, 5: 8, 8: 8, 12: 8, 16: 8, 17: 8, 21: 8}   # -> 9 classes, TISSUES first
NUCLEUS_TYPES = {'tumor_nonMitotic': 'tumour', 'tumor_mitotic': 'tumour', 'lymphocyte': 'lymphocyte', 'plasma_cell': 'lymphocyte',
                 'macrophage': 'macrophage', 'nonTILnonMQ_stromal': 'fibroblast'}
# five groups of hospitals (TCGA tissue source sites); the head of fold k is trained without group k
FOLD = {'A1': 2, 'A2': 0, 'A7': 4, 'A8': 3, 'AC': 2, 'AN': 4, 'AO': 3, 'AQ': 1, 'AR': 1, 'B6': 0, 'BH': 2, 'C8': 0, 'D8': 1,
        'E2': 3, 'E9': 3, 'EW': 4, 'GI': 0, 'GM': 2, 'HN': 3, 'LL': 4, 'LQ': 0, 'OL': 1, 'S3': 0}
BEETLE_TRAIN = ['TCGA-A8-A0AB-01Z-00-DX1.103ED338-A0F9-403B-A10C-49840BD60EB8', 'TCGA-AC-A4ZE-01Z-00-DX1.A9BB4DBC-4314-4A77-AA1C-0882DBC370F1',
                'TCGA-AC-A6IX-01Z-00-DX1.45B63C4F-9DC3-4CED-9DF9-0F86E647BEA3', 'TCGA-AC-A8OS-01Z-00-DX1.3FD44846-8BD2-4A7A-9A87-8D3D29C25F60',
                'TCGA-C8-A8HR-01Z-00-DX1.8E841CA8-1CE2-4242-AC5A-3247075CF8F4']          # the lobular class is learned here
BEETLE_TCGA = ['TCGA-A2-A0EP-01Z-00-DX1.1180C406-5C18-4373-8621-1B7B70875113', 'TCGA-A2-A0YK-01Z-00-DX1.FB23FC30-B3B2-452A-A6B0-94129B333207',
               'TCGA-AC-A2FE-01Z-00-DX1.C036AE58-3C0D-4DBF-B658-296D03FD6701', 'TCGA-AC-A5XS-01Z-00-DX1.E5F2481E-9314-4B20-84B7-370EF82AF4BB',
               'TCGA-AR-A2LK-01Z-00-DX1.FBD59C38-CD4E-4C22-BC74-A57C192A9BBC']           # checked here, TCGA
BEETLE_RADBOUD = ['patient109_wsi1', 'patient111_wsi1', 'patient113_wsi1', 'patient119_wsi1', 'patient121_wsi1', 'patient287_wsi1', 'patient288_wsi1']
BEETLE_PATIENTS = {s[:12] for s in BEETLE_TRAIN + BEETLE_TCGA}   # never used to train the tissue classes
L2 = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]       # chosen for each fold by 3-fold cross-validation over its training hospitals
TOKENS_PER_CLASS = 20000                   # lobular head: BCSS tokens per class (seed 0)
OUTLINE_TILES_PER_SLIDE = 100
BANK_K = 20
DEVICE = 'cuda'
torch.set_num_threads(4)
Image.MAX_IMAGE_PIXELS = None
CACHE.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(MANIFEST, low_memory=False)
DIAGNOSIS = manifest[manifest.label.isin(['ILC', 'NST_proxy'])].drop_duplicates('patient_id').set_index('patient_id').label.map({'ILC': 'lobular', 'NST_proxy': 'ductal'})
DX1 = manifest[manifest.slide_id.str.contains('-DX1')].drop_duplicates('patient_id').set_index('patient_id').slide_id
MERGE = np.full(256, -1)
for code, k in BCSS_CODES.items():
    MERGE[code] = k


# -------------------------------------------------------------------------------------------------------------- helpers
def fold_of(hospitals):
    """The fold that left each hospital out, -1 outside TCGA (read by the mean of the five folds)."""
    return np.array([FOLD.get(h, -1) for h in hospitals])


def token_labels(mask, grid):
    """The BCSS class covering at least half of each token, else -1."""
    step = 448 // grid
    blocks = mask.reshape(grid, step, grid, step).swapaxes(1, 2).reshape(grid * grid, -1)
    labels = np.full(grid * grid, -1)
    for k in range(9):
        labels[(blocks == k).mean(axis=1) >= 0.5] = k
    return labels


def token_at(x, y, grid):
    """The token holding pixel (x, y) of a 448 px tile."""
    row = np.clip(np.asarray(y) * grid // 448, 0, grid - 1).astype(int)
    col = np.clip(np.asarray(x) * grid // 448, 0, grid - 1).astype(int)
    return row * grid + col


def token_pixels(token):
    """The centres of the 28 x 28 pixels of an H-optimus-0 token."""
    row, col = divmod(int(token), 16)
    yy, xx = np.mgrid[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] + 0.5
    return np.c_[xx.ravel(), yy.ravel()]


def nucleus_fill(px, py, token):
    """The share of a token that a nucleus covers, and the share of the nucleus inside that token."""
    nucleus = Polygon(np.c_[px, py])
    yy, xx = np.mgrid[int(min(py)):int(max(py)) + 1, int(min(px)):int(max(px)) + 1] + 0.5
    inside = nucleus.contains_points(np.c_[xx.ravel(), yy.ravel()])
    in_token = nucleus.contains_points(token_pixels(token)).sum()
    return in_token / 784, in_token / max(inside.sum(), 1)


def read_tile(slide, x, y, mpp):
    """112 µm of a slide from level 0 at (x, y), as a 448 px tile, then 224 px for the encoders."""
    side = 112 / mpp
    ix, iy = math.floor(x), math.floor(y)
    region = slide.read_region((ix, iy), 0, (math.ceil(side + x - ix), math.ceil(side + y - iy))).convert('RGB')
    tile = region.transform((448, 448), Image.Transform.EXTENT, (x - ix, y - iy, x - ix + side, y - iy + side), resample=Image.Resampling.BILINEAR)
    return np.asarray(tile.resize((224, 224), Image.Resampling.BILINEAR))


def outline_share(outline, x, y, side):
    """The share of a tile (top-left x, y and side in level-0 pixels) inside a tumour outline drawn at 1/16 of level 0."""
    crop = outline[int(y / 16):int((y + side) / 16), int(x / 16):int((x + side) / 16)]
    return crop.mean() if crop.size else 0.0


def shrink(img448):
    return np.asarray(Image.fromarray(img448).resize((224, 224), Image.Resampling.BILINEAR))


def log_odds(logits, k):
    """Logit of class k against all the other classes."""
    others = np.delete(logits, k, axis=1)
    return logits[:, k] - np.log(np.exp(others - others.max(axis=1, keepdims=True)).sum(axis=1)) - others.max(axis=1)


def markdown(df, digits=2, bold_low=()):
    """A table as markdown, the best encoder of each row in bold (lowest for the rows in bold_low)."""
    lines = ['| | ' + ' | '.join(map(str, df.columns)) + ' |', '|---|' + '---|' * len(df.columns)]
    for name, row in df.iterrows():
        best = row.idxmin() if name in bold_low else row.idxmax()
        lines.append(f'| {name} | ' + ' | '.join(f'**{v:.{digits}f}**' if c == best else f'{v:.{digits}f}' for c, v in row.items()) + ' |')
    return '\n'.join(lines)


# --------------------------------------------------------------------------------------------- 1. BCSS and NuCLS tiles
# BCSS: non-overlapping 448 px tiles of every region, at least half annotated; the tile label is the largest class if it
# covers at least half of the annotated pixels. NuCLS: one 448 px tile centred on each field, cut from the BCSS region.
bounds = pd.read_csv(BCSS / 'meta/roiBounds_BaseMagnification.csv', index_col=0)
regions = {f.name.split('_xmin')[0]: f.name for f in (BCSS / 'masks').glob('*.png')}
if not (CACHE / 'bcss.parquet').exists():
    tiles, images, masks = [], [], []
    for slide, region in tqdm(sorted(regions.items()), desc='BCSS regions'):
        mask = MERGE[np.asarray(Image.open(BCSS / 'masks' / region))]
        rgb = np.asarray(Image.open(BCSS / 'rgbs_raw' / region).convert('RGB'))
        for y in range(0, mask.shape[0] - 447, 448):
            for x in range(0, mask.shape[1] - 447, 448):
                m = mask[y:y + 448, x:x + 448]
                if (m >= 0).mean() < 0.5:
                    continue
                tiles.append({'region': region, 'x': x, 'y': y, 'patient': slide[:12]})
                images.append(shrink(rgb[y:y + 448, x:x + 448]))
                masks.append(m.astype(np.int8))
    pd.DataFrame(tiles).to_parquet(CACHE / 'bcss.parquet')
    np.save(CACHE / 'bcss_images.npy', np.stack(images))
    np.save(CACHE / 'bcss_masks.npy', np.stack(masks))
bcss = pd.read_parquet(CACHE / 'bcss.parquet')
bcss_masks = np.load(CACHE / 'bcss_masks.npy', mmap_mode='r')
bcss['hospital'] = bcss.patient.str[5:7]
bcss['dx'] = bcss.patient.map(DIAGNOSIS)
counts = np.stack([np.bincount(m[m >= 0], minlength=9) for m in bcss_masks])
bcss['label'] = np.where(counts.max(axis=1) >= 0.5 * counts.sum(axis=1), counts.argmax(axis=1), -1)

if not (CACHE / 'nucls.parquet').exists():
    tiles, images, nuclei = [], [], []
    for f in tqdm(sorted((NUCLS / 'csv').glob('*.csv')), desc='NuCLS fields'):
        slide = f.stem.split('_id-')[0]
        left, top, right, bottom = (int(re.search(k + r'-(\d+)', f.stem).group(1)) for k in ('left', 'top', 'right', 'bottom'))
        b = bounds.loc[slide]
        rgb = Image.open(BCSS / 'rgbs_raw' / regions[slide])
        sx, sy = rgb.size[0] / (b.xmax - b.xmin), rgb.size[1] / (b.ymax - b.ymin)              # region pixels per level-0 pixel
        x0, y0 = round(((left + right) / 2 - b.xmin) * sx - 224), round(((top + bottom) / 2 - b.ymin) * sy - 224)
        if x0 < 0 or y0 < 0 or x0 + 448 > rgb.size[0] or y0 + 448 > rgb.size[1]:
            continue
        field_w, field_h = Image.open(NUCLS / 'rgb' / f'{f.stem}.png').size
        fx, fy = (right - left) / field_w, (bottom - top) / field_h                                  # level-0 pixels per field pixel
        tiles.append({'field': f.stem, 'region': regions[slide], 'x': x0, 'y': y0, 'patient': slide[:12]})
        images.append(shrink(np.asarray(rgb.crop((x0, y0, x0 + 448, y0 + 448)).convert('RGB'))))
        for n in pd.read_csv(f).itertuples():
            if n.type == 'polyline':
                px, py = np.array(n.coords_x.split(','), float), np.array(n.coords_y.split(','), float)
            else:                                                                                    # a box
                px, py = np.array([n.xmin, n.xmax, n.xmax, n.xmin], float), np.array([n.ymin, n.ymin, n.ymax, n.ymax], float)
            tx, ty = (left + px * fx - b.xmin) * sx - x0, (top + py * fy - b.ymin) * sy - y0
            nuclei.append({'tile': len(tiles) - 1, 'type': NUCLEUS_TYPES.get(n.main_classification, 'other'), 'polygon': n.type == 'polyline',
                           'cx': tx.mean(), 'cy': ty.mean(), 'px': tx.tolist(), 'py': ty.tolist()})
    pd.DataFrame(tiles).to_parquet(CACHE / 'nucls.parquet')
    pd.DataFrame(nuclei).to_parquet(CACHE / 'nuclei.parquet')
    np.save(CACHE / 'nucls_images.npy', np.stack(images))
nucls = pd.read_parquet(CACHE / 'nucls.parquet')
nucls['hospital'] = nucls.patient.str[5:7]
nucls['dx'] = nucls.patient.map(DIAGNOSIS)
nuclei = pd.read_parquet(CACHE / 'nuclei.parquet')
nuclei = nuclei[nuclei.cx.between(0, 447.9) & nuclei.cy.between(0, 447.9) & (nuclei.type != 'other')].reset_index(drop=True)
nuclei['patient'] = nucls.patient.to_numpy()[nuclei.tile]
nuclei['hospital'] = nuclei.patient.str[5:7]

# --------------------------------------------------------------------------------------------------------- 2. BEETLE tiles
# Regions are BEETLE's "other" outlines (label 1). Single cells are invasive tumour outlines (label 3) of at most 200 µm²
# whose centre lies in a region. Each cell is read on the 112 µm tile of its region's grid that holds its centre, at the
# token of that centre.
if not (CACHE / 'beetle.parquet').exists():
    archive = zipfile.ZipFile(BEETLE / 'annotations.zip')
    tiles, cells = [], []
    for name in tqdm(BEETLE_TRAIN + BEETLE_TCGA + BEETLE_RADBOUD, desc='BEETLE outlines'):
        path = TCGA_SLIDES / f'{name}.svs' if name in BEETLE_TRAIN + BEETLE_TCGA else BEETLE / f'wsis/{name}.tif'
        slide = openslide.OpenSlide(str(path))
        mpp = float(slide.properties['openslide.mpp-x'])
        scale = slide.dimensions[0] / tifffile.TiffFile(io.BytesIO(archive.read(f'annotations/masks/{name}.tif'))).pages[0].shape[1]
        outlines = json.loads(archive.read(f'annotations/jsons/{name}.json'))
        region_polys = [np.asarray(o['coordinates'], float).reshape(-1, 2) * scale for o in outlines if o['label']['value'] == 1]
        cell_polys = [np.asarray(o['coordinates'], float).reshape(-1, 2) * scale for o in outlines if o['label']['value'] == 3]
        areas = np.array([0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))) * mpp ** 2 for p in cell_polys])
        centres = np.array([p.mean(axis=0) for p in cell_polys])
        side = 112 / mpp
        for region in region_polys:
            origin = region.min(axis=0)
            for j in np.flatnonzero((areas <= 200) & Polygon(region).contains_points(centres)):
                x, y = origin + (centres[j] - origin) // side * side                                 # the grid tile holding the centre
                tiles.append({'slide': name, 'x': x, 'y': y, 'mpp': mpp})
                cells.append({'slide': name, 'x': x, 'y': y, 'cx': (centres[j, 0] - x) / side * 448, 'cy': (centres[j, 1] - y) / side * 448,
                              'px': ((cell_polys[j][:, 0] - x) / side * 448).tolist(), 'py': ((cell_polys[j][:, 1] - y) / side * 448).tolist()})
    tiles = pd.DataFrame(tiles).drop_duplicates(['slide', 'x', 'y']).reset_index(drop=True)
    cells = pd.DataFrame(cells).drop_duplicates(['slide', 'cx', 'cy', 'x', 'y']).reset_index(drop=True)
    cells['tile'] = cells.merge(tiles.reset_index(), on=['slide', 'x', 'y'], how='left')['index'].to_numpy()
    images = np.zeros((len(tiles), 224, 224, 3), np.uint8)
    for name, rows in tqdm(tiles.groupby('slide'), desc='BEETLE tiles'):
        slide = openslide.OpenSlide(str(TCGA_SLIDES / f'{name}.svs' if name in BEETLE_TRAIN + BEETLE_TCGA else BEETLE / f'wsis/{name}.tif'))
        for i, r in rows.iterrows():
            images[i] = read_tile(slide, r.x, r.y, r.mpp)
    tiles.to_parquet(CACHE / 'beetle.parquet')
    cells.to_parquet(CACHE / 'cells.parquet')
    np.save(CACHE / 'beetle_images.npy', images)
beetle = pd.read_parquet(CACHE / 'beetle.parquet')
beetle['set'] = beetle.slide.map({**{s: 'train' for s in BEETLE_TRAIN}, **{s: 'TCGA' for s in BEETLE_TCGA}, **{s: 'Radboudumc' for s in BEETLE_RADBOUD}})
beetle['patient'] = np.where(beetle.set == 'Radboudumc', beetle.slide.str.split('_').str[0], beetle.slide.str[:12])
beetle['hospital'] = np.where(beetle.set == 'Radboudumc', 'Radboudumc', beetle.patient.str[5:7])
cells = pd.read_parquet(CACHE / 'cells.parquet')
cells['set'] = beetle.set.to_numpy()[cells.tile]

# ---------------------------------------------------------------------------------------------- 3. tumour-outline tiles
# Slides with a coarse tumour outline and a lobular or ductal diagnosis, outside BEETLE: a grid of 112 µm tiles over the
# outline, and up to 100 of the tiles lying at least half inside it (random, seed 0). The outline is drawn at 1/16 of level 0.
if not (CACHE / 'outline_tiles.parquet').exists():
    archive = zipfile.ZipFile(OUTLINES)
    outline_files = {Path(n).name.replace('_mask.tiff', ''): n for n in archive.namelist() if n.endswith('_mask.tiff')}
    usable = [p for p in sorted(TCGA_SLIDES.glob('*.svs')) if p.stem in outline_files and p.stem[:12] in DIAGNOSIS.index and p.stem[:12] not in BEETLE_PATIENTS]
    tiles, images = [], []
    for path in tqdm(usable, desc='outline slides'):
        outline = np.asarray(Image.open(io.BytesIO(archive.read(outline_files[path.stem])))) > 0
        slide = openslide.OpenSlide(str(path))
        mpp = float(slide.properties['openslide.mpp-x'])
        side = 112 / mpp
        rows, cols = np.nonzero(outline)
        grid = [(x, y) for y in np.arange(rows.min() * 16, (rows.max() + 1) * 16, side) for x in np.arange(cols.min() * 16, (cols.max() + 1) * 16, side)]
        inside = [xy for xy in grid if outline_share(outline, *xy, side) >= 0.5]
        for k in np.random.default_rng(0).permutation(len(inside))[:OUTLINE_TILES_PER_SLIDE]:
            x, y = inside[k]
            tiles.append({'slide': path.stem, 'x': x, 'y': y, 'patient': path.stem[:12]})
            images.append(read_tile(slide, x, y, mpp))
    pd.DataFrame(tiles).to_parquet(CACHE / 'outline_tiles.parquet')
    np.save(CACHE / 'outline_tiles.npy', np.stack(images))
outline = pd.read_parquet(CACHE / 'outline_tiles.parquet')
outline['hospital'] = outline.patient.str[5:7]
outline['dx'] = outline.patient.map(DIAGNOSIS)

# ------------------------------------------------------------------------------------------------------------ 4. encoders
# name: (model, mean, std, prefix tokens before the patches, timm or Hugging Face)
MODELS = {
    'H-optimus-0': (lambda: timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True, init_values=1e-5, dynamic_img_size=False),
                    (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517), 5, True),
    'UNI2-h': (lambda: timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True, img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                                         embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, mlp_layer=SwiGLUPacked,
                                         act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 9, True),
    'Virchow2': (lambda: timm.create_model('hf-hub:paige-ai/Virchow2', pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU),
                 (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 5, True),
    'Hibou-B': (lambda: Dinov2WithRegistersModel.from_pretrained('histai/hibou-b'), (0.7068, 0.5755, 0.722), (0.195, 0.2316, 0.1816), 5, False),
    'Phikon-v2': (lambda: Dinov2Model.from_pretrained('owkin/phikon-v2'), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1, False),
}


@torch.no_grad()
def encode(model, mean, std, prefix, is_timm, images, batch=16):
    """Final patch tokens (float16) and class tokens (float32) of 224 px images."""
    mean, std = torch.tensor(mean, device=DEVICE)[:, None, None], torch.tensor(std, device=DEVICE)[:, None, None]
    patches, classes = [], []
    for start in range(0, len(images), batch):
        x = (torch.as_tensor(np.asarray(images[start:start + batch]), device=DEVICE).permute(0, 3, 1, 2).float() / 255 - mean) / std
        out = model.forward_features(x.half()) if is_timm else model(x.half()).last_hidden_state
        patches.append(out[:, prefix:].cpu().numpy())
        classes.append(out[:, 0].float().cpu().numpy())
    return np.concatenate(patches), np.concatenate(classes)


# --------------------------------------------------------------------------------------------------------------- 5. heads
def fit(X, y, n_classes, l2, iters=200, info=None):
    """Multinomial logistic regression on standardised features, class-balanced, L2 penalty, L-BFGS on the GPU.
    Returns W and b with the standardisation folded in. If info is a dict, it receives the L2, iterations and examples."""
    X = torch.as_tensor(X, device=DEVICE).float()
    y = torch.as_tensor(y, device=DEVICE).long()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    count = torch.bincount(y, minlength=n_classes).float()
    weight = torch.where(count > 0, count.sum() / (n_classes * count.clamp(min=1)), torch.zeros_like(count))
    W = torch.zeros(Z.shape[1], n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
    optimiser = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn='strong_wolfe')

    def loss():
        optimiser.zero_grad()
        value = torch.nn.functional.cross_entropy(Z @ W + b, y, weight=weight) + l2 * (W ** 2).sum()
        value.backward()
        return value
    optimiser.step(loss)
    if info is not None:
        info.update(l2=l2, iterations=int(optimiser.state[W]['n_iter']), examples=len(y))
    return (W / sd[:, None]).detach().cpu().numpy(), (b - (mu / sd) @ W).detach().cpu().numpy()


def balanced_loss(head, X, y):
    """Cross-entropy averaged over the classes present."""
    logits = torch.as_tensor(X @ head[0] + head[1])
    p = torch.softmax(logits, 1).clamp_min(1e-7)
    return float(np.mean([-torch.log(p[y == k, k]).mean().item() for k in np.unique(y)]))


def train_folds(X, y, hospitals, n_classes, log=None):
    """Five heads. The head of fold k is trained without the hospitals of fold k, with its L2 weight chosen by 3-fold
    cross-validation over its training hospitals. If log is a list, it receives the L2, iterations and examples of each fold."""
    heads = []
    folds = fold_of(hospitals)
    for k in range(5):
        Xk, yk, hk = X[folds != k], y[folds != k], hospitals[folds != k]
        losses = {l2: 0.0 for l2 in L2}
        for inner_train, inner_val in GroupKFold(3).split(Xk, yk, hk):
            for l2 in L2:
                losses[l2] += balanced_loss(fit(Xk[inner_train], yk[inner_train], n_classes, l2), Xk[inner_val], yk[inner_val])
        info = {}
        heads.append(fit(Xk, yk, n_classes, min(L2, key=losses.get), info=info))
        if log is not None:
            log.append(info)
    return heads


def read_out(heads, X, hospitals):
    """Logits of each row by the fold that did not see its hospital (mean of the five folds outside TCGA)."""
    logits = np.stack([X @ W + b for W, b in heads])
    folds = fold_of(hospitals)
    return np.where((folds >= 0)[:, None], logits[np.clip(folds, 0, 4), np.arange(len(X))], logits.mean(axis=0))


def auroc_within(scores, positive, groups):
    """Mean over groups (tiles) of the AUROC of the positives against the negatives of the same group."""
    values = []
    for g in np.unique(groups):
        pos, neg = scores[(groups == g) & positive], scores[(groups == g) & ~positive]
        if len(pos) and len(neg):
            values.append(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())
    return float(np.mean(values))


def fg_det(hit, tile, tile_call_is_target):
    """FG Det.: the share of tiles holding the target in which a target token is predicted as the target class, over all
    such tiles and over the outvoted ones (tile predicted as another class). Also returns the two numbers of tiles."""
    per_tile = pd.Series(hit).groupby(np.asarray(tile)).any()
    outvoted = ~np.asarray(tile_call_is_target)[per_tile.index]
    return per_tile.mean(), per_tile[outvoted].mean(), len(per_tile), int(outvoted.sum())


# -------------------------------------------------------------------------------- 6. every encoder: encode, train, read
bcss_images = np.load(CACHE / 'bcss_images.npy', mmap_mode='r')
other_images = {'nucls': np.load(CACHE / 'nucls_images.npy'), 'beetle': np.load(CACHE / 'beetle_images.npy'), 'outline': np.load(CACHE / 'outline_tiles.npy')}
five = bcss.label.between(0, 4).to_numpy()
eval_cells = cells[cells.set != 'train'].reset_index(drop=True)
train_cells = cells[cells.set == 'train'].reset_index(drop=True)
card, numbers, saved, kept = [], {}, {'ductal_folds': [], 'lobular_folds': [], 'lobular_mean_folds': [], 'cls_folds': []}, {}

for model_name in ENCODERS:
    build, mean, std, prefix, is_timm = MODELS[model_name]
    model = build().eval().half().to(DEVICE)
    first = encode(model, mean, std, prefix, is_timm, bcss_images[:1])[0]
    grid = int(np.sqrt(first.shape[1]))
    bcss_tokens = np.lib.format.open_memmap(CACHE / f'bcss_tokens_{model_name}.npy', 'w+', np.float16, (len(bcss),) + first.shape[1:])
    bcss_cls = np.zeros((len(bcss), first.shape[2]), np.float32)
    for start in tqdm(range(0, len(bcss), 256), desc=f'{model_name}: BCSS tiles'):
        bcss_tokens[start:start + 256], bcss_cls[start:start + 256] = encode(model, mean, std, prefix, is_timm, bcss_images[start:start + 256])
    nucls_tokens = encode(model, mean, std, prefix, is_timm, other_images['nucls'])[0]
    beetle_tokens = encode(model, mean, std, prefix, is_timm, other_images['beetle'])[0]
    outline_mean = np.concatenate([encode(model, mean, std, prefix, is_timm, other_images['outline'][i:i + 256])[0].mean(axis=1, dtype=np.float32)
                                   for i in tqdm(range(0, len(outline), 256), desc=f'{model_name}: outline tiles')])
    model.cpu()
    del model
    torch.cuda.empty_cache()
    bcss_mean = np.concatenate([bcss_tokens[i:i + 512].mean(axis=1, dtype=np.float32) for i in range(0, len(bcss), 512)])
    bcss_labels = np.stack([token_labels(m, grid) for m in bcss_masks])

    # ductal head and class-token head: BCSS tiles of ductal patients outside BEETLE, five tissue classes
    train = (bcss.dx == 'ductal').to_numpy() & ~bcss.patient.isin(BEETLE_PATIENTS).to_numpy() & five
    ductal = train_folds(bcss_mean[train], bcss.label.to_numpy()[train], bcss.hospital.to_numpy()[train], 5)
    cls_head = train_folds(bcss_cls[train], bcss.label.to_numpy()[train], bcss.hospital.to_numpy()[train], 5)

    # lobular head: single tokens. Lobular tumour from BEETLE's training cells; the other classes from BCSS tokens covered
    # at least half by one tissue class, tumour only from ductal patients (as ductal tumour), at most 20,000 per class.
    tile_of, position = np.nonzero((bcss_labels >= 0) & (bcss_labels <= 4))
    six = bcss_labels[tile_of, position] + 1                                   # tumour 0 -> ductal tumour 1, stroma 1 -> 2, ...
    usable = ~bcss.patient.isin(BEETLE_PATIENTS).to_numpy()[tile_of] & ((six != 1) | (bcss.dx.to_numpy()[tile_of] == 'ductal'))
    rng = np.random.default_rng(0)
    picked = np.sort(np.concatenate([rng.permutation(np.flatnonzero(usable & (six == k)))[:TOKENS_PER_CLASS] for k in range(1, 6)]))
    train_positions = token_at(train_cells.cx, train_cells.cy, grid)
    X = np.concatenate([bcss_tokens[tile_of[picked], position[picked]].astype(np.float32), beetle_tokens[train_cells.tile, train_positions].astype(np.float32)])
    y = np.concatenate([six[picked], np.zeros(len(train_cells), int)])
    hospitals = np.concatenate([bcss.hospital.to_numpy()[tile_of[picked]], beetle.hospital.to_numpy()[train_cells.tile]])
    lobular_log = []
    lobular = train_folds(X, y, hospitals, 6, lobular_log)

    # the lobular head trained on mean tokens instead, for comparison: one example per tile, its mean token. BCSS tiles with one
    # of the five labels (tumour only from ductal patients, as ductal tumour), outside BEETLE's patients; every BEETLE training
    # tile (each holds at least one outlined cell) as lobular tumour.
    beetle_mean = beetle_tokens.mean(axis=1, dtype=np.float32)
    m_bcss = five & ~bcss.patient.isin(BEETLE_PATIENTS).to_numpy() & ((bcss.label != 0).to_numpy() | (bcss.dx == 'ductal').to_numpy())
    m_beetle = (beetle.set == 'train').to_numpy()
    XM = np.concatenate([bcss_mean[m_bcss], beetle_mean[m_beetle]])
    yM = np.concatenate([bcss.label.to_numpy()[m_bcss] + 1, np.zeros(m_beetle.sum(), int)])
    hM = np.concatenate([bcss.hospital.to_numpy()[m_bcss], beetle.hospital.to_numpy()[m_beetle]])
    lobular_mean_log = []
    lobular_mean = train_folds(XM, yM, hM, 6, lobular_mean_log)
    saved['ductal_folds'].append(ductal)
    saved['lobular_folds'].append(lobular)
    saved['lobular_mean_folds'].append(lobular_mean)
    saved['cls_folds'].append(cls_head)

    # readouts of every labelled BCSS token, in chunks
    token_label = bcss_labels[tile_of, position]
    token_ductal, token_lobular, token_lobular_mean, token_cls = [], [], [], []
    for start in tqdm(range(0, len(tile_of), 200000), desc=f'{model_name}: BCSS readouts'):
        chunk = bcss_tokens[tile_of[start:start + 200000], position[start:start + 200000]].astype(np.float32)
        hosp = bcss.hospital.to_numpy()[tile_of[start:start + 200000]]
        token_ductal.append(read_out(ductal, chunk, hosp).argmax(axis=1))
        token_lobular.append(read_out(lobular, chunk, hosp).argmax(axis=1))
        token_lobular_mean.append(read_out(lobular_mean, chunk, hosp).argmax(axis=1))
        token_cls.append(read_out(cls_head, chunk, hosp).argmax(axis=1))
    token_ductal, token_lobular, token_cls = np.concatenate(token_ductal), np.concatenate(token_lobular), np.concatenate(token_cls)
    token_lobular_mean = np.concatenate(token_lobular_mean)
    tile_ductal = read_out(ductal, bcss_mean, bcss.hospital).argmax(axis=1)
    tile_lobular = read_out(lobular, bcss_mean, bcss.hospital)
    tile_cls = read_out(cls_head, bcss_cls, bcss.hospital).argmax(axis=1)
    nucleus_X = nucls_tokens[nuclei.tile, token_at(nuclei.cx, nuclei.cy, grid)].astype(np.float32)
    nucleus_ductal = read_out(ductal, nucleus_X, nuclei.hospital)
    nucleus_lobular = read_out(lobular, nucleus_X, nuclei.hospital).argmax(axis=1)
    nucleus_cls = read_out(cls_head, nucleus_X, nuclei.hospital)
    nucls_mean = nucls_tokens.mean(axis=1, dtype=np.float32)
    nucls_tile = read_out(ductal, nucls_mean, nucls.hospital)
    cell_X = beetle_tokens[eval_cells.tile, token_at(eval_cells.cx, eval_cells.cy, grid)].astype(np.float32)
    cell_ductal = read_out(ductal, cell_X, beetle.hospital.to_numpy()[eval_cells.tile]).argmax(axis=1)
    cell_lobular = read_out(lobular, cell_X, beetle.hospital.to_numpy()[eval_cells.tile]).argmax(axis=1)
    beetle_tile_ductal = read_out(ductal, beetle_mean, beetle.hospital).argmax(axis=1)
    beetle_tile_lobular = read_out(lobular, beetle_mean, beetle.hospital)
    outline_ductal = read_out(ductal, outline_mean, outline.hospital).argmax(axis=1)
    outline_lobular = read_out(lobular, outline_mean, outline.hospital)

    def add(table, measure, value, n=''):
        card.append({'table': table, 'measure': measure, 'encoder': model_name, 'value': float(value), 'n': n})

    # card, ductal head on ductal tumour (BCSS, NuCLS)
    T = 'Ductal head, ductal tumour (BCSS tiles and tokens, NuCLS nuclei)'
    recall = [(tile_ductal[five & (bcss.label == k).to_numpy()] == k).mean() for k in range(5)]
    add(T, 'Tile acc., balanced over five classes', np.mean(recall))
    add(T, 'Tile acc., tumour', recall[0])
    add(T, 'Loc., tumour tokens', (token_ductal[token_label == 0] == 0).mean())
    a, b, n, m = fg_det(token_ductal[token_label == 0] == 0, tile_of[token_label == 0], tile_ductal == 0)
    add(T, 'FG Det., tumour', a, n)
    add(T, 'FG Det., tumour, outvoted tiles', b, m)
    for kind in ['tumour', 'lymphocyte', 'fibroblast']:
        add(T, f'NuCLS {kind} nuclei predicted as tumour', (nucleus_ductal[(nuclei.type == kind).to_numpy()].argmax(axis=1) == 0).mean())
    per_hospital = {}
    for h in bcss.hospital.unique():
        here = five & (bcss.hospital == h).to_numpy()
        present = [k for k in range(5) if (here & (bcss.label == k).to_numpy()).sum() >= 5]   # classes with at least 5 tiles
        if here.sum() >= 50:
            per_hospital[h] = np.mean([(tile_ductal[here & (bcss.label == k).to_numpy()] == k).mean() for k in present])
    add(T, 'worst hospital, balanced tile acc.', min(per_hospital.values()))

    # card, ductal head on lobular slides
    T = 'Ductal head on lobular slides, lobular tumour as tumour (BEETLE, tumour outlines)'
    lob, duc = (outline.dx == 'lobular').to_numpy(), (outline.dx == 'ductal').to_numpy()
    add(T, "Tile acc., lobular patients' outline tiles predicted as tumour", (outline_ductal[lob] == 0).mean(), int(lob.sum()))
    for where in ['TCGA', 'Radboudumc']:
        here = (eval_cells.set == where).to_numpy()
        add(T, f'Loc., outlined lobular cells predicted as tumour ({where})', (cell_ductal[here] == 0).mean(), int(here.sum()))
        a, b, n, m = fg_det(cell_ductal[here] == 0, eval_cells.tile.to_numpy()[here], beetle_tile_ductal == 0)
        add(T, f'FG Det. ({where})', a, n)
        add(T, f'FG Det., outvoted tiles ({where})', b, m)

    # card, lobular head on ductal and lobular tumour
    T = 'Lobular head, ductal and lobular tumour'
    merged = np.maximum(tile_lobular.argmax(axis=1) - 1, 0)                     # lobular and ductal tumour -> tumour
    add(T, 'Tile acc., balanced over five classes (tumours merged)', np.mean([(merged[five & (bcss.label == k).to_numpy()] == k).mean() for k in range(5)]))
    lod = outline_lobular[:, 0] - outline_lobular[:, 1]
    add(T, "Tile acc., ductal patients' outline tiles predicted as ductal", (outline_lobular[duc].argmax(axis=1) == 1).mean(), int(duc.sum()))
    add(T, "Tile acc., lobular patients' outline tiles predicted as lobular", (outline_lobular[lob].argmax(axis=1) == 0).mean(), int(lob.sum()))
    add(T, "Tumour type, ductal patients' outline tiles with lobular < ductal", (lod[duc] < 0).mean())
    add(T, "Tumour type, lobular patients' outline tiles with lobular > ductal", (lod[lob] > 0).mean())
    ductal_tokens = (token_label == 0) & (bcss.dx.to_numpy()[tile_of] == 'ductal')
    add(T, 'Loc., ductal: BCSS tumour tokens of ductal patients predicted as ductal', (token_lobular[ductal_tokens] == 1).mean())
    add(T, 'Loc., lobular: outlined lobular cells predicted as lobular (TCGA)', (cell_lobular[(eval_cells.set == 'TCGA').to_numpy()] == 0).mean())
    a, b, n, m = fg_det(token_lobular[ductal_tokens] == 1, tile_of[ductal_tokens], tile_lobular.argmax(axis=1) == 1)
    add(T, 'FG Det., ductal (BCSS)', a, n)
    add(T, 'FG Det., ductal, outvoted tiles', b, m)
    for where in ['TCGA', 'Radboudumc']:
        here = (eval_cells.set == where).to_numpy()
        a, b, n, m = fg_det(cell_lobular[here] == 0, eval_cells.tile.to_numpy()[here], beetle_tile_lobular.argmax(axis=1) == 0)
        add(T, f'FG Det., lobular ({where})', a, n)
        add(T, f'FG Det., lobular, outvoted tiles ({where})', b, m)
    add(T, 'Loc., lobular (Radboudumc)', (cell_lobular[(eval_cells.set == 'Radboudumc').to_numpy()] == 0).mean())
    add(T, 'BCSS stroma tokens predicted as lobular', (token_lobular[token_label == 1] == 0).mean())
    ductal_nuclei = ((nuclei.type == 'tumour') & (nuclei.patient.map(DIAGNOSIS) == 'ductal')).to_numpy()
    add(T, 'NuCLS tumour nuclei of ductal patients predicted as lobular', (nucleus_lobular[ductal_nuclei] == 0).mean(), int(ductal_nuclei.sum()))
    for kind in ['lymphocyte', 'fibroblast']:
        add(T, f'NuCLS {kind} nuclei predicted as tumour', (nucleus_lobular[(nuclei.type == kind).to_numpy()] <= 1).mean())

    # the lobular head trained on single tokens (above) or on mean tokens
    T = 'Lobular head trained on single tokens or on mean tokens'
    cell_X_hosp = beetle.hospital.to_numpy()[eval_cells.tile]
    for label, head, head_tokens, log in [('single tokens', lobular, token_lobular, lobular_log), ('mean tokens', lobular_mean, token_lobular_mean, lobular_mean_log)]:
        head_cells = read_out(head, cell_X, cell_X_hosp).argmax(axis=1)
        head_beetle_tile = read_out(head, beetle_mean, beetle.hospital).argmax(axis=1)
        head_merged = np.maximum(read_out(head, bcss_mean, bcss.hospital).argmax(axis=1) - 1, 0)
        tcga = (eval_cells.set == 'TCGA').to_numpy()
        add(T, f'Loc., lobular (TCGA), {label}', (head_cells[tcga] == 0).mean())
        add(T, f'Loc., lobular (Radboudumc), {label}', (head_cells[~tcga] == 0).mean())
        add(T, f'FG Det., lobular, outvoted tiles (TCGA), {label}', fg_det(head_cells[tcga] == 0, eval_cells.tile.to_numpy()[tcga], head_beetle_tile == 0)[1])
        add(T, f'BCSS stroma tokens predicted as lobular, {label}', (head_tokens[token_label == 1] == 0).mean())
        add(T, f'Loc., ductal, {label}', (head_tokens[ductal_tokens] == 1).mean())
        add(T, f'Tile acc., balanced over five classes (tumours merged), {label}', np.mean([(head_merged[five & (bcss.label == k).to_numpy()] == k).mean() for k in range(5)]))
        add(T, f'training examples per fold, {label}', np.mean([i['examples'] for i in log]))
        add(T, f'chosen L2, median over folds, {label}', np.median([i['l2'] for i in log]))
        add(T, f'L-BFGS iterations, mean over folds, {label}', np.mean([i['iterations'] for i in log]))

    # numbers quoted in sections 1 and 3: the class-token head against the ductal head
    tumour_nucleus = (nuclei.type == 'tumour').to_numpy()
    numbers[model_name] = {
        'ductal head, balanced tile acc.': float(np.mean(recall)),
        'class-token head, balanced tile acc.': float(np.mean([(tile_cls[five & (bcss.label == k).to_numpy()] == k).mean() for k in range(5)])),
        'ductal head, Loc. mean of five classes': float(np.mean([(token_ductal[token_label == k] == k).mean() for k in range(5)])),
        'class-token head on patch tokens, Loc. mean of five classes': float(np.mean([(token_cls[token_label == k] == k).mean() for k in range(5)])),
        'class-token head, mean of its token readouts agrees with its tile call': float((read_out(cls_head, bcss_mean, bcss.hospital).argmax(axis=1) == tile_cls).mean()),
        'ductal head, NuCLS within-tile AUROC': auroc_within(log_odds(nucleus_ductal, 0), tumour_nucleus, nuclei.tile.to_numpy()),
        'class-token head on patch tokens, NuCLS within-tile AUROC': auroc_within(log_odds(nucleus_cls, 0), tumour_nucleus, nuclei.tile.to_numpy()),
        'NuCLS nuclei scored by their tile, AUROC': float(roc_auc_score(tumour_nucleus, log_odds(nucls_tile, 0)[nuclei.tile])),
        'tissue type per hospital, median': float(np.median(list(per_hospital.values()))),
        'tissue type, worst hospital': min(per_hospital, key=per_hospital.get),
        'tissue type, hospitals': len(per_hospital),
    }
    kept[model_name] = {
                        'beetle_lod': beetle_tile_lobular[:, 0] - beetle_tile_lobular[:, 1], 'bcss_lobular': tile_lobular,
                        'nucls_lobular': read_out(lobular, nucls_mean, nucls.hospital)}

    # H-optimus-0 only: the bank of reliable tokens (section 6)
    if model_name == 'H-optimus-0':

        # the annotated error of section 7: a tumour nucleus of a ductal patient predicted as lobular tumour that fills its token
        # (covers at least 40% of the token, at least 60% of the nucleus inside it), the one with the highest lobular - ductal log-odds
        nucleus_logits = read_out(lobular, nucleus_X, nuclei.hospital)
        candidates = []
        for i in np.flatnonzero(ductal_nuclei & (nucleus_logits.argmax(axis=1) == 0)):
            n = nuclei.loc[i]
            token = int(token_at(n.cx, n.cy, 16))
            covers, inside = nucleus_fill(n.px, n.py, token)
            if covers >= 0.4 and inside >= 0.6:
                candidates.append((nucleus_logits[i, 0] - nucleus_logits[i, 1], i, token))
        lod_case, i, token = max(candidates)
        t = nucls.loc[nuclei.tile[i]]
        numbers['ductal nucleus predicted as lobular (section 7)'] = {
            'candidates': len(candidates), 'slide': DX1[t.patient], 'x': int(t.x + 224), 'y': int(t.y + 224), 'size': 134.4, 'token': token,
            'lobular - ductal': float(lod_case)}
        # the bank of section 6 (report 034): tokens that each cover one annotated structure. Every token is moved to row 8, column 8
        # of a tile cut around it (token 136), so that all bank tokens share one interior position and their full context.
        candidates = []
        # lobular cells: outlined cells on the TCGA slides that cover at least half of their token
        for c in eval_cells[eval_cells.set == 'TCGA'].itertuples():
            token = int(token_at(c.cx, c.cy, 16))
            if Polygon(np.c_[c.px, c.py]).contains_points(token_pixels(token)).mean() < 0.5:
                continue
            t = beetle.loc[c.tile]
            r, col = divmod(token, 16)
            step = 112 / t.mpp / 16                                                   # level-0 pixels per token
            candidates.append({'cls': 'lobular cell', 'source': 'beetle', 'slide': t.slide, 'x': t.x + (col - 8) * step, 'y': t.y + (r - 8) * step,
                               'mpp': t.mpp, 'patient': t.patient})
        # ductal tumour nuclei (ductal patients) and fibroblast nuclei: NuCLS outlines that cover at least 60% (ductal) or 40% (fibroblast)
        # of their token with at least 60% of the nucleus inside it; the tile is shifted in the BCSS region image
        wanted = {'tumour': 'ductal tumour nucleus', 'fibroblast': 'fibroblast nucleus'}
        min_cover = {'tumour': 0.6, 'fibroblast': 0.4}
        for n in nuclei.itertuples():
            if n.type not in wanted or not n.polygon or n.patient not in DX1.index or (n.type == 'tumour' and DIAGNOSIS.get(n.patient) != 'ductal'):
                continue
            token = int(token_at(n.cx, n.cy, 16))
            covers, inside = nucleus_fill(n.px, n.py, token)
            if covers >= min_cover[n.type] and inside >= 0.6:
                t = nucls.loc[n.tile]
                r, col = divmod(token, 16)
                candidates.append({'cls': wanted[n.type], 'source': 'bcss', 'region': t.region, 'x0': t.x + (col - 8) * 28, 'y0': t.y + (r - 8) * 28,
                                   'patient': t.patient})
        # stroma: tokens of NuCLS fields (where every nucleus is outlined) entirely inside a BCSS stroma region, touched by no
        # annotated nucleus of any type; at most 5,000 (seed 0)
        every_nucleus = pd.read_parquet(CACHE / 'nuclei.parquet')
        stroma = []
        for region, tiles in nucls[nucls.patient.isin(DX1.index)].groupby('region'):
            b = bounds.loc[region.split('_xmin')[0]]
            mask = MERGE[np.asarray(Image.open(BCSS / 'masks' / region))]
            sx, sy = mask.shape[1] / (b.xmax - b.xmin), mask.shape[0] / (b.ymax - b.ymin)
            for i, t in tiles.iterrows():
                left, top, right, bottom = (int(re.search(k + r'-(\d+)', t.field).group(1)) for k in ('left', 'top', 'right', 'bottom'))
                fx0, fy0, fx1, fy1 = (left - b.xmin) * sx - t.x, (top - b.ymin) * sy - t.y, (right - b.xmin) * sx - t.x, (bottom - b.ymin) * sy - t.y
                nucleus_pixels = np.zeros((448, 448), bool)
                for n in every_nucleus[every_nucleus.tile == i].itertuples():
                    x0, y0, x1, y1 = max(int(min(n.px)), 0), max(int(min(n.py)), 0), min(int(max(n.px)) + 1, 448), min(int(max(n.py)) + 1, 448)
                    yy, xx = np.mgrid[y0:max(y1, y0), x0:max(x1, x0)]                  # empty for a nucleus outside the tile
                    nucleus_pixels[yy, xx] |= Polygon(np.c_[n.px, n.py]).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(yy.shape)
                    vx, vy = np.asarray(n.px).astype(int), np.asarray(n.py).astype(int)
                    on_tile = (vx >= 0) & (vx < 448) & (vy >= 0) & (vy < 448)
                    nucleus_pixels[vy[on_tile], vx[on_tile]] = True
                m = mask[t.y:t.y + 448, t.x:t.x + 448]
                for token in range(256):
                    r, col = divmod(token, 16)
                    if col * 28 < fx0 or (col + 1) * 28 > fx1 or r * 28 < fy0 or (r + 1) * 28 > fy1:
                        continue
                    if (m[r * 28:(r + 1) * 28, col * 28:(col + 1) * 28] == 1).all() and not nucleus_pixels[r * 28:(r + 1) * 28, col * 28:(col + 1) * 28].any():
                        stroma.append({'cls': 'stroma', 'source': 'bcss', 'region': region, 'x0': t.x + (col - 8) * 28, 'y0': t.y + (r - 8) * 28, 'patient': t.patient})
        rng = np.random.default_rng(0)
        candidates = pd.DataFrame(candidates + [stroma[i] for i in sorted(rng.permutation(len(stroma))[:5000])])
        # two structures in the same token give the same centred tile: keep one
        candidates = candidates.drop_duplicates(['cls', 'source', 'slide', 'region', 'x', 'y', 'x0', 'y0']).reset_index(drop=True)
        candidates['hospital'] = candidates.patient.str[5:7]
        # the centred tiles: BEETLE from the slides, NuCLS from the BCSS region images (dropped where the shifted tile leaves the region)
        images = np.zeros((len(candidates), 224, 224, 3), np.uint8)
        inside_region = np.ones(len(candidates), bool)
        for slide, rows in candidates[candidates.source == 'beetle'].groupby('slide'):
            reader = openslide.OpenSlide(str(TCGA_SLIDES / f'{slide}.svs'))
            for i, c in rows.iterrows():
                images[i] = read_tile(reader, c.x, c.y, c.mpp)
        for region, rows in tqdm(candidates[candidates.source == 'bcss'].groupby('region'), desc='bank: centred BCSS tiles'):
            rgb = np.asarray(Image.open(BCSS / 'rgbs_raw' / region).convert('RGB'))
            for i, c in rows.iterrows():
                if c.x0 < 0 or c.y0 < 0 or c.x0 + 448 > rgb.shape[1] or c.y0 + 448 > rgb.shape[0]:
                    inside_region[i] = False
                    continue
                images[i] = shrink(rgb[int(c.y0):int(c.y0) + 448, int(c.x0):int(c.x0) + 448])
        candidates, images = candidates[inside_region].reset_index(drop=True), images[inside_region]
        # kept if the lobular head predicts the centred token as its class
        model = build().eval().half().to(DEVICE)
        centred = np.concatenate([encode(model, mean, std, prefix, is_timm, images[i:i + 256])[0][:, 136].astype(np.float32) for i in range(0, len(images), 256)])
        model.cpu()
        del model
        torch.cuda.empty_cache()
        target = candidates.cls.map({'lobular cell': 0, 'ductal tumour nucleus': 1, 'fibroblast nucleus': 2, 'stroma': 2}).to_numpy()
        candidates['vector'] = list(centred)
        bank = candidates[read_out(lobular, centred, candidates.hospital).argmax(axis=1) == target].copy()
        numbers['bank: candidates, predicted as their class'] = {cls: [int((candidates.cls == cls).sum()), int((bank.cls == cls).sum())]
                                                               for cls in ['lobular cell', 'ductal tumour nucleus', 'fibroblast nucleus', 'stroma']}
        bank['slide'] = np.where(bank.source == 'beetle', bank.slide, bank.patient.map(DX1))
        bank['x'] = np.where(bank.source == 'beetle', bank.x, bank.x0 + 224)          # BEETLE: top-left on the slide; BCSS: centre in the region
        bank['y'] = np.where(bank.source == 'beetle', bank.y, bank.y0 + 224)
        bank['size'] = np.where(bank.source == 'beetle', 112.0, 134.4)
        bank['token'] = 136
        # neighbour agreement: on a class-balanced pool, keep the tokens whose 20 nearest tokens from other patients are at least half
        # of their own class (cosine), then cap every class at the smallest again (chance 25%)
        smallest = bank.cls.value_counts().min()
        bank = pd.concat([g.sample(smallest, random_state=0) for _, g in bank.groupby('cls', sort=False)], ignore_index=True)
        vectors = np.stack(bank.vector).copy()
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = vectors @ vectors.T
        similarity[bank.patient.to_numpy()[:, None] == bank.patient.to_numpy()[None]] = -np.inf
        nearest = np.argsort(-similarity, axis=1)[:, :20]
        agreement = (bank.cls.to_numpy()[nearest] == bank.cls.to_numpy()[:, None]).mean(axis=1)
        numbers['bank: balanced pool, kept by neighbour agreement'] = {cls: [int((bank.cls == cls).sum()), int(((bank.cls == cls) & (agreement >= 0.5)).sum())]
                                                                       for cls in ['lobular cell', 'ductal tumour nucleus', 'fibroblast nucleus', 'stroma']}
        bank = bank[agreement >= 0.5]
        smallest = bank.cls.value_counts().min()
        bank = pd.concat([g.sample(smallest, random_state=0) for _, g in bank.groupby('cls', sort=False)], ignore_index=True)
        bank[['cls', 'source', 'slide', 'x', 'y', 'size', 'token', 'patient']].to_csv(OUT / 'bank_tokens.csv', index=False)
        # own-class share among a bank token's 20 nearest bank tokens (other patients), per class
        V = np.stack(bank.vector)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        similarity = V @ V.T
        similarity[bank.patient.to_numpy()[:, None] == bank.patient.to_numpy()[None]] = -np.inf
        own = (bank.cls.to_numpy()[np.argsort(-similarity, axis=1)[:, :20]] == bank.cls.to_numpy()[:, None]).mean(axis=1)
        numbers['bank'] = {'tokens per class': int(smallest), 'tiles': int(bank[['slide', 'x', 'y']].drop_duplicates().shape[0]),
                           'own-class share among the 20 nearest, all': round(float(own.mean()), 2),
                           **{f'own-class share among the 20 nearest, {cls}': round(float(own[(bank.cls == cls).to_numpy()].mean()), 2) for cls in bank.cls.unique()}}
    del bcss_tokens
    print(model_name, 'done', flush=True)

# ---------------------------------------------------------------------------------------------------------- 7. outputs
OUT.mkdir(exist_ok=True)
to_tensor = lambda folds: [(torch.as_tensor(W), torch.as_tensor(b)) for W, b in folds]
torch.save({'ductal_folds': [to_tensor(f) for f in saved['ductal_folds']], 'lobular_folds': [to_tensor(f) for f in saved['lobular_folds']],
            'lobular_mean_folds': [to_tensor(f) for f in saved['lobular_mean_folds']],
            'cls_folds': [to_tensor(f) for f in saved['cls_folds']]},
           OUT / 'linear_heads.pt')

lines = []
card = pd.DataFrame(card)
lower_is_better = ['NuCLS lymphocyte nuclei predicted as tumour', 'NuCLS fibroblast nuclei predicted as tumour', 'BCSS stroma tokens predicted as lobular',
                   'NuCLS tumour nuclei of ductal patients predicted as lobular', 'BCSS stroma tokens predicted as lobular, single tokens',
                   'BCSS stroma tokens predicted as lobular, mean tokens', 'L-BFGS iterations, mean over folds, single tokens', 'L-BFGS iterations, mean over folds, mean tokens']
for table, rows in card.groupby('table', sort=False):
    wide = rows.pivot_table(index='measure', columns='encoder', values='value', sort=False)[ENCODERS]
    sizes = ', '.join(f'{r.measure}: {r.n}' for r in rows[(rows.encoder == 'H-optimus-0') & (rows.n != '')].itertuples())
    lines += [f'**{table}**', '', markdown(wide, bold_low=lower_is_better), '', f'sizes: {sizes}', '']


# example tiles of section 5: the most extreme tile of each kind by the lobular head's tile call (H-optimus-0)
lod_beetle = kept['H-optimus-0']['beetle_lod']
cells_per_tile = cells[cells.set == 'TCGA'].groupby('tile').size()
beetle_candidates = cells_per_tile[cells_per_tile >= 10].index.to_numpy()
nucls_ductal = np.flatnonzero((nucls.dx == 'ductal').to_numpy())
nucls_lod = kept['H-optimus-0']['nucls_lobular'][:, 0] - kept['H-optimus-0']['nucls_lobular'][:, 1]
bcss_ductal = np.flatnonzero(((bcss.dx == 'ductal') & (bcss.label == 0)).to_numpy())
bcss_lod = kept['H-optimus-0']['bcss_lobular'][:, 0] - kept['H-optimus-0']['bcss_lobular'][:, 1]
most_lobular_beetle = beetle.loc[beetle_candidates[lod_beetle[beetle_candidates].argmax()]]
most_ductal_beetle = beetle.loc[beetle_candidates[lod_beetle[beetle_candidates].argmin()]]
most_ductal_nucls = nucls.loc[nucls_ductal[nucls_lod[nucls_ductal].argmin()]]
most_lobular_bcss = bcss.loc[bcss_ductal[bcss_lod[bcss_ductal].argmax()]]
pd.DataFrame([
    {'kind': 'lobular patient, most lobular', 'source': 'beetle', 'slide': most_lobular_beetle.slide, 'x': most_lobular_beetle.x, 'y': most_lobular_beetle.y, 'size': 112.0},
    {'kind': 'lobular patient, most ductal', 'source': 'beetle', 'slide': most_ductal_beetle.slide, 'x': most_ductal_beetle.x, 'y': most_ductal_beetle.y, 'size': 112.0},
    {'kind': 'ductal patient, most ductal', 'source': 'bcss', 'slide': DX1[most_ductal_nucls.patient], 'x': most_ductal_nucls.x + 224, 'y': most_ductal_nucls.y + 224, 'size': 134.4},
    {'kind': 'ductal patient, most lobular', 'source': 'bcss', 'slide': DX1[most_lobular_bcss.patient], 'x': most_lobular_bcss.x + 224, 'y': most_lobular_bcss.y + 224, 'size': 134.4},
]).to_csv(OUT / 'examples.csv', index=False)
numbers['data'] = {'NuCLS nuclei': len(nuclei), 'ductal head training tiles': int(train.sum()), 'lobular head training cells': len(train_cells)}
numbers['examples'] = {
    'BEETLE TCGA tiles with at least ten cells': len(beetle_candidates), 'of them with lobular > ductal': int((lod_beetle[beetle_candidates] > 0).sum()),
    'NuCLS tiles of ductal patients': len(nucls_ductal), 'of them predicted as lobular': int((kept['H-optimus-0']['nucls_lobular'][nucls_ductal].argmax(axis=1) == 0).sum()),
    'BCSS tumour tiles of ductal patients': len(bcss_ductal), 'of them predicted as lobular ': int((kept['H-optimus-0']['bcss_lobular'][bcss_ductal].argmax(axis=1) == 0).sum()),
}
lines += ['**Numbers quoted in the text**', '', '```', json.dumps(numbers, indent=1, default=float), '```']
(OUT / 'tables.md').write_text('\n'.join(lines))
print('\n'.join(lines))
