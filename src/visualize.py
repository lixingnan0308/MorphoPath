"""Slide-rendering utilities shared by the interpretability visualisations
(``visualization/``): map feature coordinates onto the slide thumbnail, size
scatter markers, and pull patch thumbnails from the raw WSI.
"""


def compute_patch_spacing(svs_path, target_patch_um=256):
    """Compute read_size_l0 from the slide MPP, matching UNI extraction:
       read_size_l0 = int(target_patch_um * (0.5 / mpp_x)).
    """
    try:
        import openslide
        s = openslide.OpenSlide(svs_path)
        mpp_x = s.properties.get('openslide.mpp-x')
        if mpp_x is None:
            res_x = s.properties.get('tiff.XResolution')
            unit = s.properties.get('tiff.ResolutionUnit')
            if res_x and unit == 'centimeter':
                mpp_x = 10000.0 / float(res_x)
            else:
                mpp_x = 0.5
        mpp_x = float(mpp_x)
        s.close()
        return int(target_patch_um * (0.5 / mpp_x))
    except Exception:
        return int(target_patch_um * (0.5 / 0.25))  # fallback: assume mpp=0.25


def get_thumbnail(svs_path, level=2):
    try:
        import openslide
        s = openslide.OpenSlide(svs_path)
        dims = s.level_dimensions
        lv = min(level, len(dims) - 1)
        img = s.read_region((0, 0), lv, dims[lv]).convert("RGB")
        return img, dims[0], s.level_downsamples[lv]
    except Exception:
        return None, None, None


def extract_patch_img(svs_path, x, y, patch_spacing):
    try:
        import openslide
        s = openslide.OpenSlide(svs_path)
        best_level = 0
        for lv in range(len(s.level_dimensions)):
            if int(patch_spacing / s.level_downsamples[lv]) >= 128:
                best_level = lv
            else:
                break
        ds = s.level_downsamples[best_level]
        rs = max(int(patch_spacing / ds), 1)
        return s.read_region((int(x), int(y)), best_level, (rs, rs)).convert("RGB").resize((256, 256))
    except Exception:
        return None


def _vis_coords(coords, slide_dims, downsample, patch_spacing):
    half = patch_spacing / 2
    ds = downsample if downsample else 1.0
    sw = slide_dims[0] if slide_dims else coords[:, 0].max() + patch_spacing
    sh = slide_dims[1] if slide_dims else coords[:, 1].max() + patch_spacing
    return coords / ds, half / ds, sw / ds, sh / ds, patch_spacing / ds


def _ms(patch_vis, fig_w, sw_vis, dpi=150):
    px = (sw_vis / fig_w) / 72.0 * (dpi / 100.0)
    return max((patch_vis / max(px, 0.1)) ** 2, 4.0)
