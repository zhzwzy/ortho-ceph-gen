# datasets_cepha.py
import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import pandas as pd

# =========================
# Configuration: crop landmarks and stable registration landmarks
# =========================
# These landmarks define the crop region.
CROP_POINT_NAMES = [
    "Ns","FH-p",
    "Co","C'", 
    "Rlar-p",
    "Rlar-a","Go","MP-a",
    "UL","LL",
    "C","Prn","Pos","Gs","Mes",
    "S","N","G","FH-a","Ba"
]
STABLE_POINT_NAMES = ["G","N","Ns","FH-p","S","FH-a","Co","C'","Rlar-p","Ba"]
RULER_POINT_NAMES = ["R1", "R2"]
BONE_CANNY_LOW = 100
BONE_CANNY_HIGH = 200
# =========================
# Landmarks used for Canny post-processing
# =========================
# External soft-tissue landmarks
MASK_SOFT_TISSUE_NAMES = [
    'G', 'Ns', 'Prn', 'Sn', 'UL', 'Stoms', 'Stomi', 'LL', 'Ms', 'Pos', 'Gs', 'Mes', 'C'
]

IMPORTANT_POINT_NAMES = [
    "Prn", "Sn", "PP-a", "A", "U1-a", "U1C", "PP-p", "UL", "U1-ta", "Stoms", 
    "Stomi", "U1-p", "L1-p", "LL", "L1C", "B", "L1-a", "Ms", "Pos", "Gs", 
    "Mes", "MP-p", "GN", "Po", "L4", "U6-a", "U6-p", "L6-a", "L6-p", 
    "Rlar-a", "Go", "MP-a"
]

def get_gender_en(gender_cn):
    """Convert a Chinese gender label to English."""
    if not isinstance(gender_cn, str): return "patient"
    if '男' in gender_cn: return "male"
    elif '女' in gender_cn: return "female"
    return "patient"

def normalize_extraction_status(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return None
    if text in ("1", "1.0", "是", "有", "yes", "Yes", "YES", "true", "True", "TRUE"):
        return "1"
    if text in ("0", "0.0", "否", "无", "no", "No", "NO", "false", "False", "FALSE"):
        return "0"
    return None

# =========================
# Basic utilities
# =========================
def read_image_gray(path: str) -> np.ndarray:
    """Read a grayscale image from a path and return a uint8 array."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = None
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return img  # uint8

def apply_clahe_and_normalize(img: np.ndarray,
                            clip_limit: float = 2.0,
                            tile_grid_size: Tuple[int,int] = (8,8)) -> np.ndarray:
    """Apply CLAHE contrast enhancement and normalize the image to [0, 1].

    The input image is expected to be uint8.
    """
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img * 255.0, 0, 255)).astype(np.uint8)
        else:
            img = (np.clip(img, 0, 255)).astype(np.uint8)

    # To enable CLAHE, uncomment the next two lines and comment out the following line.
    # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # img_enhanced = clahe.apply(img)
    img_enhanced = img # CLAHE is not applied.

    return img_enhanced.astype(np.float32) / 255.0

def rotate_image_and_points(img: np.ndarray,
                            pts: Optional[np.ndarray],
                            angle_deg_cw: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    angle = -float(angle_deg_cw)
    h, w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_r = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    if pts is None:
        return img_r, None
    
    pts = np.atleast_2d(pts).astype(np.float32)
    if pts.shape[0] == 0:
        return img_r, np.zeros((0,2), dtype=np.float32)
        
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    pts_r = (M @ pts_h.T).T
    return img_r, pts_r

def to_tensor_img(img01: np.ndarray) -> torch.Tensor:
    """Convert a grayscale image to a three-channel tensor [3, H, W] in [0, 1]."""
    if img01.ndim == 3:
        img01 = img01[:, :, 0]
        
    img3 = np.stack([img01, img01, img01], axis=0)
    return torch.from_numpy(img3.astype(np.float32))

def read_rotation_csv(csv_path: str) -> Dict[str, float]:
    rot = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if 'name' not in reader.fieldnames or 'rotation' not in reader.fieldnames:
            raise ValueError(f"{csv_path} 需要包含列: name, rotation")
        for row in reader:
            rot[row['name'].strip()] = float(row['rotation'])
    return rot

# Simplified: alias handling is disabled.
def _normalize_point_name(n: str) -> str:
    return n 

def expand_aliases(pt_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return pt_map 
# ---------------------------------

def read_landmarks_with_names(csv_path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 3: raise ValueError(f"{csv_path} 表头列数异常")
        key0 = header[0].strip()
        if key0 not in ("name", "image"): raise ValueError(f"{csv_path} 首列应为 'name' 或 'image'，实际：{key0}")

        col_pairs: List[Tuple[str,int,int]] = []
        order: List[str] = []
        i = 1
        while i < len(header):
            col = header[i].strip()
            if col.endswith("_X"):
                base = col[:-2]
                if i+1 >= len(header) or header[i+1].strip() != base + "_Y":
                    raise ValueError(f"{csv_path} 表头 {base} 的 Y 列缺失/不匹配")
                base_norm = base 
                col_pairs.append((base_norm, i, i+1))
                if base_norm not in order: order.append(base_norm)
                i += 2
            else: i += 1

        lm_dict: Dict[str, Dict[str, np.ndarray]] = {}
        for row in reader:
            img_name = row[0].strip()
            mp: Dict[str, np.ndarray] = {}
            for (p, ix, iy) in col_pairs:
                try:
                    x = float(row[ix]) if row[ix] != '' else np.nan
                    y = float(row[iy]) if row[iy] != '' else np.nan
                except Exception: x = np.nan; y = np.nan
                mp[p] = np.array([x, y], dtype=np.float32)
            lm_dict[img_name] = mp 
    return lm_dict, order

def bbox_from_points(points: np.ndarray,
                     margin: float,
                     img_size_wh: Tuple[int,int]) -> Tuple[int,int,int,int]:
    w, h = img_size_wh
    valid = points[~np.isnan(points).any(axis=1)]
    if valid.size == 0: return 0, 0, w, h
    x0, y0 = float(valid[:,0].min()), float(valid[:,1].min())
    x1, y1 = float(valid[:,0].max()), float(valid[:,1].max())
    bw, bh = x1-x0, y1-y0
    cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
    scale = max(1.0 + margin, 1.0)
    hw, hh = max(bw*scale/2, 1.0), max(bh*scale/2, 1.0)
    xl, yt = int(round(cx-hw)), int(round(cy-hh))
    xr, yb = int(round(cx+hw)), int(round(cy+hh))
    xl = max(0, xl); yt = max(0, yt); xr = min(w, xr); yb = min(h, yb)
    return xl, yt, max(1, xr-xl), max(1, yb-yt)

def scale_by_ruler(img: np.ndarray, pts_arr: np.ndarray,
                   pt_names: List[str],
                   target_mm_per_px: float, ruler_mm: float
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = img.shape[:2]
    s = 1.0

    try:
        pt_map = {name: pts_arr[i] for i, name in enumerate(pt_names)}
        p1 = pt_map.get("R1")
        p2 = pt_map.get("R2")
        
        if p1 is None or p2 is None: 
            return img, pts_arr, 1.0
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)): 
            raise ValueError("R1/R2 含 NaN")
            
        d_px = float(np.linalg.norm(p1 - p2))

        if d_px < 1.0: # A distance below one pixel indicates a likely annotation error.
            print(f"⚠️ [Scale Warning] R1-R2 像素距离过小 ({d_px:.2f} px)，跳过缩放。")
            raise ValueError("R1/R2 距离过小")
            
        px_per_mm_target = 1.0 / target_mm_per_px
        d_target_px = ruler_mm * px_per_mm_target
        
        calculated_s = d_target_px / d_px
        
        new_w = int(round(w * calculated_s))
        new_h = int(round(h * calculated_s))
        MAX_SIDE = 6000 
        if new_w > MAX_SIDE or new_h > MAX_SIDE:
            print(f"⚠️ [Scale Warning] 缩放后尺寸过大 ({new_w}x{new_h}, s={calculated_s:.2f})！可能 R1/R2 标注过近。已强制取消缩放。")
            s = 1.0
        else:
            s = calculated_s

    except Exception as e:
        s = 1.0
    if abs(s - 1.0) < 1e-6: 
        return img, pts_arr, 1.0
    img_s = cv2.resize(img, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_LINEAR)
    
    pts_s = pts_arr.copy().astype(np.float32)
    mask = ~np.isnan(pts_s).any(axis=1)
    if mask.any(): 
        pts_s[mask, 0] *= s
        pts_s[mask, 1] *= s
        
    return img_s, pts_s, s

def crop_and_resize_with_points(img: np.ndarray,
                                pts_full: Optional[np.ndarray],
                                rect: Tuple[int,int,int,int],
                                out_size: int
                                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    x, y, w, h = rect
    crop = img[y:y+h, x:x+w]
    if w <= 0 or h <= 0:
        print(f"警告: 无效裁剪窗口 rect={rect}。返回 {out_size}x{out_size} 黑图。")
        img_out = np.zeros((out_size, out_size), dtype=img.dtype)
        pts_out = pts_full.copy() if pts_full is not None else None
        if pts_out is not None: pts_out[:] = np.nan
        return img_out, pts_out
    img_out = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    if pts_full is None: return img_out, None
    pts = np.array(pts_full, dtype=np.float32, copy=True)
    mask = ~np.isnan(pts).any(axis=1)
    if mask.any():
        pts[mask, 0] = pts[mask, 0] - x
        pts[mask, 1] = pts[mask, 1] - y
        s_x = float(out_size) / float(w if w > 0 else 1.0)
        s_y = float(out_size) / float(h if h > 0 else 1.0)
        pts[mask, 0] *= s_x; pts[mask, 1] *= s_y
    return img_out, pts

def is_pre_name(fn: str) -> bool:
    stem = os.path.splitext(os.path.basename(fn))[0]
    return stem.endswith("-1") or stem.endswith("_1")

def is_post_name(fn: str) -> bool:
    stem = os.path.splitext(os.path.basename(fn))[0]
    return stem.endswith("-2") or stem.endswith("_2")

def estimate_similarity(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    s = src_pts.astype(np.float32); d = dst_pts.astype(np.float32)
    m = ~np.isnan(s).any(axis=1) & ~np.isnan(d).any(axis=1)
    s, d = s[m], d[m]
    if s.shape[0] < 2: return np.array([[1,0,0],[0,1,0]], dtype=np.float32)
    mu_s, mu_d = s.mean(axis=0), d.mean(axis=0)
    S = s - mu_s; D = d - mu_d
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[1,:] *= -1; R = Vt.T @ U.T
    var_S = (S**2).sum()/S.shape[0]
    scale = 1.0 if var_S < 1e-8 else np.trace(R.T @ H)/(var_S*S.shape[0])
    t = mu_d - scale*(R @ mu_s)
    return np.hstack([scale*R, t.reshape(2,1)]).astype(np.float32)

def warp_image_with_M(img: np.ndarray, M: np.ndarray, out_wh: Tuple[int,int],
                      border_mode=cv2.BORDER_CONSTANT, border_value=0) -> np.ndarray:
    w, h = out_wh
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value
    )

def warp_points_with_M(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0: return np.zeros((0,2), dtype=np.float32)
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    ph = np.hstack([pts.astype(np.float32), ones])
    return (M @ ph.T).T

def pick_points_by_names(pt_map: Dict[str, np.ndarray], wanted_names: List[str]) -> np.ndarray:
    pt_map_exp = pt_map 
    pts = []
    for n in wanted_names:
        pt = pt_map_exp.get(n)
        if pt is not None: pts.append(pt)
    if len(pts) == 0: return np.zeros((0,2), dtype=np.float32)
    return np.stack(pts, axis=0).astype(np.float32)

def generate_heatmap(image_size: int, points: Dict[str, np.ndarray], point_names: List[str], sigma: float = 8.0) -> np.ndarray:
    """Generate a Gaussian heatmap using an efficient patch-based method.

    The peak value at each landmark center is 1.0.
    """
    # 1. Initialize a black background.
    target = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Values outside three standard deviations are negligible and need not be computed.
    tmp_size = sigma * 3

    for name in point_names:
        pt = points.get(name)
        if pt is None or np.isnan(pt).any():
            continue
            
        mu_x, mu_y = pt[0], pt[1]
        
        # 2. Compute the patch bounds around the landmark.
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # Upper-left corner.
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # Bottom-right corner.

        # 3. Skip patches entirely outside the image.
        if ul[0] >= image_size or ul[1] >= image_size or br[0] < 0 or br[1] < 0:
            continue

        # Build the coordinate grid within the patch.
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2 # Patch center.

        # 4. Standard Gaussian function.
        # H = exp( -((x-x0)^2 + (y-y0)^2) / (2*sigma^2) )
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # 5. Find the intersection between the patch and image bounds.
        g_x = max(0, -ul[0]), min(br[0], image_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], image_size) - ul[1]
        img_x = max(0, ul[0]), min(br[0], image_size)
        img_y = max(0, ul[1]), min(br[1], image_size)

        # 6. Merge the Gaussian patch into the image.
        # Use the maximum to prevent overlapping landmarks from exceeding 1.0.
        target[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            target[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )

    return target

# =========================
# 1) Style pretraining (StylePretrainDataset)
# =========================
class StylePretrainDataset(Dataset):
    def __init__(self, root: str, size: int = 512, crop_margin: float = 0.10,
                 target_mm_per_px: float = 0.1, ruler_mm: float = 10.0):
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)
        self.img_dir = os.path.join(root, "image")
        self.lm_csv = os.path.join(root, "huaxi_correction.csv")
        if not os.path.isdir(self.img_dir): raise FileNotFoundError(self.img_dir)
        if not os.path.isfile(self.lm_csv): raise FileNotFoundError(self.lm_csv)
        self.lm_dict, _ = read_landmarks_with_names(self.lm_csv)
        if not self.lm_dict: raise ValueError(f"{self.lm_csv} 为空或读取失败")
        self.files = sorted([os.path.join(self.img_dir,f) for f in os.listdir(self.img_dir)
                             if f.lower().endswith((".png",".jpg",".jpeg",".bmp"))])
        if not self.files: raise RuntimeError("预训练集未发现图像文件")
        an_item = self.lm_dict.get(os.path.basename(self.files[0]))
        if an_item is None: raise KeyError(f"landmark.csv 缺少 {os.path.basename(self.files[0])}")
        if "R1" not in an_item or "R2" not in an_item:
            print(f"警告: {self.lm_csv} 中可能缺少 R1/R2，尺度统一将失效 (s=1.0)")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]; name = os.path.basename(path)
        pt_map = self.lm_dict.get(name)
        if pt_map is None: raise KeyError(f"landmark.csv 缺少 {name}")
        img_uint8 = read_image_gray(path)
        try:
            pt_names = list(pt_map.keys())
            pts_arr = np.stack([pt_map[n] for n in pt_names], axis=0)
        except Exception as e: raise ValueError(f"处理 {name} 的点时出错: {e}")
        img_s, pts_s, scale = scale_by_ruler(
            img_uint8, pts_arr, pt_names,
            self.target_mm_per_px, self.ruler_mm
        )
        pt_map_s = {n: p for n, p in zip(pt_names, pts_s)}
        crop_pts = pick_points_by_names(pt_map_s, CROP_POINT_NAMES)
        if crop_pts.shape[0] == 0:
            valid_pts_s = pts_s[~np.isnan(pts_s).any(axis=1)]
            if valid_pts_s.shape[0] > 0: crop_pts = valid_pts_s
            else: crop_pts = np.zeros((0,2), dtype=np.float32)
        rect = bbox_from_points(crop_pts, self.crop_margin, (img_s.shape[1], img_s.shape[0]))
        img_final_uint8, pts_final = crop_and_resize_with_points(
            img_s, pts_s, rect, self.size
        )
        img_01 = apply_clahe_and_normalize(img_final_uint8)
        style_prompt = (
            "Lateral cephalometric radiograph, raw medical image, "
            "film grain, sharp focus, X-ray contrast"
        )
        return {
            "pixel_values": to_tensor_img(img_01),
            "landmarks": torch.from_numpy(pts_final.astype(np.float32)),
            "prompt": style_prompt,
            "name": name,
            "scale_factor": float(scale)
        }

# =========================
# 2) Post-treatment prompt fine-tuning (PostPromptFinetuneDataset)
# =========================
class PostPromptFinetuneDataset(Dataset):
    exts = (".bmp",".png",".jpg",".jpeg")
    def __init__(self, root: str, size: int = 512,
                 rotation_csv: Optional[str] = None,
                 post_csv: Optional[str] = None,
                 crop_margin: float = 0.10,
                 target_mm_per_px: float = 0.1, ruler_mm: float = 10.0):
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)
        image_root = os.path.join(root, "image")
        if not os.path.isdir(image_root): raise FileNotFoundError(image_root)
        self.files = []
        for r,_,fs in os.walk(image_root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    if is_post_name(f): self.files.append(os.path.join(r,f))
        if not self.files: raise RuntimeError("治疗后微调集未发现 '-2/_2' 术后图片")
        self.files.sort()
        rot_csv = rotation_csv or os.path.join(root, "rotation.csv")
        if not os.path.isfile(rot_csv): raise FileNotFoundError(f"未找到旋转CSV：{rot_csv}")
        self.rot_map = read_rotation_csv(rot_csv)
        lm_csv = post_csv or os.path.join(root, "post.csv")
        if not os.path.isfile(lm_csv): raise FileNotFoundError(f"未找到 post.csv：{lm_csv}（用于裁剪和缩放）")
        self.post_lm_dict, _ = read_landmarks_with_names(lm_csv)
        an_item = self.post_lm_dict.get(os.path.basename(self.files[0]))
        if an_item and ("R1" not in an_item or "R2" not in an_item):
            print(f"警告: {lm_csv} 中可能缺少 R1/R2，尺度统一将失效 (s=1.0)")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]; name = os.path.basename(path)
        rot_cw = float(self.rot_map.get(name, 0.0))
        img_uint8 = read_image_gray(path)
        lm_map = self.post_lm_dict.get(name, None)
        lm_names = None; lm_arr = None
        if lm_map is not None:
            lm_names = list(lm_map.keys())
            lm_arr   = np.stack([lm_map[n] for n in lm_names], axis=0)
        if lm_arr is not None:
            img_s, lm_arr_s, scale = scale_by_ruler(
                img_uint8, lm_arr, lm_names,
                self.target_mm_per_px, self.ruler_mm
            )
        else: img_s, lm_arr_s, scale = img_uint8, None, 1.0
        
        try:
            img_rot, lm_arr_r = rotate_image_and_points(img_s, lm_arr_s, rot_cw)
        except cv2.error as e:
            print(f"\n======== 💀 OpenCV 旋转崩溃报告 ========")
            print(f"崩溃文件: {name}")
            print(f"文件路径: {path}")
            print(f"原始尺寸: {img_uint8.shape}")
            print(f"缩放倍率: {scale:.4f}")
            print(f"缩放后尺寸 (img_s): {img_s.shape}")
            print(f"尝试旋转角度: {rot_cw}")
            print(f"错误详情: {e}")
            
            # Common cause: an unusually large scaled image may result from landmarks that are too close.
            if img_s.shape[0] > 30000 or img_s.shape[1] > 30000:
                print("⚠️  分析: 缩放后的图片尺寸极大 (>30000)，通常是因为 ruler (R1/R2) 点距离过近，导致计算出的 scale 异常大。")
            
            print(f"========================================\n")
            raise e
        # ========================================================
        
        # img_rot, lm_arr_r = rotate_image_and_points(img_s, lm_arr_s, rot_cw)
        lm_map_rot = None
        if lm_arr_r is not None and lm_names is not None:
            lm_map_rot = {n: p for n, p in zip(lm_names, lm_arr_r)}
        if lm_map_rot is not None:
            crop_pts = pick_points_by_names(lm_map_rot, CROP_POINT_NAMES)
            if crop_pts.shape[0] == 0:
                valid_pts_r = lm_arr_r[~np.isnan(lm_arr_r).any(axis=1)]
                if valid_pts_r.shape[0] > 0: crop_pts = valid_pts_r
                else: crop_pts = np.zeros((0,2), dtype=np.float32)
            rect = bbox_from_points(crop_pts, self.crop_margin, (img_rot.shape[1], img_rot.shape[0]))
        else: rect = (0, 0, img_rot.shape[1], img_rot.shape[0])
        img_final_uint8, pts_final = crop_and_resize_with_points(
            img_rot, lm_arr_r, rect, self.size
        )
        img_01 = apply_clahe_and_normalize(img_final_uint8)
        return {
            "pixel_values": to_tensor_img(img_01),
            "landmarks": torch.from_numpy(pts_final.astype(np.float32)) if pts_final is not None else None,
            "prompt": "cephalometric x-ray after treatment",
            "name": name,
            "rotation_deg": float(rot_cw),
            "scale_factor": float(scale)
        }

class TreatmentAwareFinetuneDataset(Dataset):
    """Stage 2 dataset for concept-alignment fine-tuning.

    Combines pretreatment and post-treatment images with prompts describing the
    image state, gender, and treatment ages from Information.csv.
    """
    exts = (".bmp", ".png", ".jpg", ".jpeg")
    
    def __init__(self, root: str, size: int = 512,
                 rotation_csv: Optional[str] = None,
                 pre_csv: Optional[str] = None,
                 post_csv: Optional[str] = None,
                 information_csv: Optional[str] = None, # Required input.
                 crop_margin: float = 0.10,
                 target_mm_per_px: float = 0.1, ruler_mm: float = 10.0):
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)
        
        # 1. Scan pretreatment and post-treatment images.
        image_root = os.path.join(root, "image")
        if not os.path.isdir(image_root): raise FileNotFoundError(image_root)
        
        self.files = []
        for r, _, fs in os.walk(image_root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    # Classify images by suffix: -1/_1 for pre and -2/_2 for post.
                    is_pre = ('-1' in f) or ('_1' in f)
                    is_post = ('-2' in f) or ('_2' in f)
                    
                    if is_pre:
                        self.files.append({'path': os.path.join(r,f), 'type': 'pre'})
                    elif is_post:
                        self.files.append({'path': os.path.join(r,f), 'type': 'post'})
                        
        if not self.files: raise RuntimeError("Stage 2 Dataset: 未发现有效图像 (-1/-2结尾)")
        
        # 2. Load metadata (rotation and landmarks).
        rot_csv = rotation_csv or os.path.join(root, "rotation.csv")
        self.rot_map = read_rotation_csv(rot_csv) if os.path.exists(rot_csv) else {}
        
        # Load pretreatment and post-treatment landmarks into one dictionary.
        self.lm_dict = {}
        if pre_csv and os.path.exists(pre_csv):
            d, _ = read_landmarks_with_names(pre_csv)
            self.lm_dict.update(d)
        if post_csv and os.path.exists(post_csv):
            d, _ = read_landmarks_with_names(post_csv)
            self.lm_dict.update(d)
        
        # 3. Load patient information.
        info_csv_path = information_csv or os.path.join(root, "Information.csv")
        if os.path.exists(info_csv_path):
            print(f"Loading patient info from: {info_csv_path}")
            self.info_map = self._load_information_csv(info_csv_path)
        else:
            print(f"Warning: {info_csv_path} not found! Prompts will be generic.")
            self.info_map = {}

    def _load_information_csv(self, path):
        """Read sample identifiers, initial age, treatment-end age, and gender from Information.csv."""
        info_map = {}
        try:
            # Try common file encodings.
            try: df = pd.read_csv(path, encoding='utf-8-sig')
            except: df = pd.read_csv(path, encoding='gbk')
            
            # Strip whitespace from column names.
            df.columns = [c.strip() for c in df.columns]
            
            for _, row in df.iterrows():
                # Use the first column as the sample identifier.
                name = str(row.iloc[0]).strip()
                if not name: continue
                
                try:
                    # Read the required fields.
                    start_age = int(row['初诊年龄'])
                    end_age = int(row['治疗结束年龄'])
                    gender = str(row['性别']).strip()
                    
                    info_map[name] = {
                        'start_age': start_age,
                        'end_age': end_age,
                        'duration': max(0, end_age - start_age),
                        'gender': gender
                    }
                except Exception:
                    continue
            return info_map
        except Exception as e:
            print(f"Error reading info csv: {e}")
            return {}

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        item = self.files[idx]
        path = item['path']
        img_type = item['type'] # 'pre' or 'post'
        fname = os.path.basename(path)
        stem = os.path.splitext(fname)[0]
        
        # Remove the -1/-2 suffix to look up the sample in info_map.
        # Example: "SampleA-1" becomes "SampleA", and "SampleB_2" becomes "SampleB".
        sid = stem
        if stem.endswith(('-1', '_1')): sid = stem[:-2]
        elif stem.endswith(('-2', '_2')): sid = stem[:-2]
        
        # ================= Image preprocessing =================
        img_uint8 = read_image_gray(path)
        rot_cw = float(self.rot_map.get(fname, 0.0))
        lm_map = self.lm_dict.get(fname, self.lm_dict.get(stem))
        
        lm_names = list(lm_map.keys()) if lm_map else None
        lm_arr = np.stack([lm_map[n] for n in lm_names], axis=0) if lm_map else None
        
        # 1. Scale.
        if lm_arr is not None:
            img_s, lm_arr_s, scale = scale_by_ruler(img_uint8, lm_arr, lm_names, self.target_mm_per_px, self.ruler_mm)
        else:
            img_s, lm_arr_s, scale = img_uint8, None, 1.0
            
        # 2. Rotate.
        img_rot, lm_arr_r = rotate_image_and_points(img_s, lm_arr_s, rot_cw)
        
        # 3. Crop.
        lm_map_rot = {n: p for n, p in zip(lm_names, lm_arr_r)} if lm_arr_r is not None else None
        if lm_map_rot:
            crop_pts = pick_points_by_names(lm_map_rot, CROP_POINT_NAMES)
            if crop_pts.shape[0] == 0: 
                # If the designated crop landmarks are unavailable, use all valid landmarks.
                crop_pts = lm_arr_r[~np.isnan(lm_arr_r).any(axis=1)]
                if crop_pts.shape[0] == 0: crop_pts = np.zeros((0,2), dtype=np.float32)
            rect = bbox_from_points(crop_pts, self.crop_margin, (img_rot.shape[1], img_rot.shape[0]))
        else:
            rect = (0, 0, img_rot.shape[1], img_rot.shape[0])
            
        img_final_uint8, pts_final = crop_and_resize_with_points(img_rot, lm_arr_r, rect, self.size)
        
        # 4. Normalize and convert to a tensor.
        img_01 = apply_clahe_and_normalize(img_final_uint8)
        
        # ================= Prompt construction =================
        info = self.info_map.get(sid)
        
        shared_style = (
            "Lateral cephalometric radiograph, raw medical image, "
            "film grain, sharp focus, X-ray contrast"
        )
        
        if info:
            gender_en = get_gender_en(info['gender'])
            start_age = info['start_age']
            end_age = info['end_age']
            
            if img_type == 'pre':
                # ==========================================
                # Pretreatment prompt.
                # ==========================================
                # Structure: image state -> patient attributes -> style description.
                prompt = (f"Pre-treatment orthodontic patient, initial malocclusion state, "
                          f"{gender_en}, {int(start_age)} years old, "
                          f"{shared_style}")
            else:
                # ==========================================
                # Post-treatment prompt.
                # ==========================================
                # Structure: outcome -> treatment actions -> aesthetic goal -> constraints -> style.
                prompt = (f"Orthodontic treatment result, "                  
                          f"retract incisors, align dentition, "             
                          f"reduce soft tissue convexity, "                  
                          f"improve facial profile aesthetics, "             
                          f"{gender_en}, treatment from {int(start_age)} to {int(end_age)} years old, " 
                          f"{shared_style}")                                 
        else:
            # Use a generic prompt when the information CSV is unavailable.
            if img_type == 'pre': 
                prompt = f"Pre-treatment orthodontic patient, {shared_style}"
            else: 
                prompt = f"Orthodontic treatment result, {shared_style}"

        return {
            "pixel_values": to_tensor_img(img_01),
            "prompt": prompt,
            "name": fname,
            "sid": sid,
            "type": img_type
        }
        
# =========================
# 3) ControlNet training (ControlNetPairDataset)
# =========================
class ControlNetPairDataset(Dataset):
    """Dataset for ControlNet training.

    Returns the post-treatment target image, edge and landmark-heatmap conditions,
    normalized treatment metrics, the pretreatment image, a prompt, and metadata.
    """
    exts = ('.bmp', '.jpg', '.jpeg', '.png')

    def __init__(self, root, size=512, 
                 rotation_csv=None, pre_csv=None, post_csv=None,
                 movement_csv=None, information_csv=None,
                 crop_margin=0.10, target_mm_per_px=0.1, ruler_mm=10.0,
                 canny_low=25, canny_high=40,
                 mask_dilate_kernel_size=15, min_contour_area=15,
                 heatmap_sigma=8.0,
                 exclude_patients=None):
        
        self.data_root = "data"
        self.root = root
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)
        
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.mask_dilate_kernel_size = mask_dilate_kernel_size | 1
        self.min_contour_area = min_contour_area
        self.heatmap_sigma = heatmap_sigma
        self.exclude_patients = set(exclude_patients or [])
        if self.exclude_patients:
            print(f"[Dataset] Excluding patients: {', '.join(sorted(self.exclude_patients))}")

        # 1. Find paired images.
        img_root = root # Alternatively, use os.path.join(root, "image").

        pairs = {}
        for r, _, fs in os.walk(img_root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    p = os.path.join(r, f)
                    stem = os.path.splitext(f)[0]
                    if is_pre_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['pre'] = p
                    elif is_post_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['post'] = p
        
        all_items = [(sid, d['pre'], d['post']) for sid, d in pairs.items() if 'pre' in d and 'post' in d]
        if self.exclude_patients:
            self.items = [item for item in all_items if item[0] not in self.exclude_patients]
            removed = len(all_items) - len(self.items)
            print(f"[Dataset] Excluded {removed} paired samples, remaining {len(self.items)}.")
        else:
            self.items = all_items
        
        # 2. Load rotation and landmark CSV files.
        try:
            rot_csv_path = rotation_csv or os.path.join(self.data_root, "rotation.csv")
            pre_csv_path = pre_csv or os.path.join(self.data_root, "pre.csv")
            post_csv_path = post_csv or os.path.join(self.data_root, "post.csv")
            
            self.rot_map = read_rotation_csv(rot_csv_path)
            self.pre_lm_dict, _ = read_landmarks_with_names(pre_csv_path)
            self.post_lm_dict, _ = read_landmarks_with_names(post_csv_path)
        except Exception as e:
            print(f"Error loading CSV metadata: {e}")
            self.rot_map = {}
            self.pre_lm_dict = {}
            self.post_lm_dict = {}

        # 3. Load movement metrics.
        self.movement_info = None
        if movement_csv and os.path.exists(movement_csv):
            print(f"Loading movement metrics from: {movement_csv}")
            self.movement_info = self._load_movement_csv(movement_csv)
        else:
            print("Warning: No movement_csv provided or found. conditioning_metrics will be empty.")

        # 4. Load patient information for prompt construction.
        self.info_map = {}
        info_csv_path = information_csv or os.path.join(root, "Information.csv")
        if os.path.exists(info_csv_path):
            print(f"Loading patient info from: {info_csv_path}")
            self.info_map = self._load_information_csv(info_csv_path)
        else:
            print(f"Warning: {info_csv_path} not found! Prompts will be generic.")

    def _load_information_csv(self, path):
        """Read sample identifiers, initial age, treatment-end age, and gender from Information.csv."""
        info_map = {}
        try:
            # Try common file encodings.
            try: df = pd.read_csv(path, encoding='utf-8-sig')
            except: df = pd.read_csv(path, encoding='gbk')
            
            # Strip whitespace from column names.
            df.columns = [c.strip() for c in df.columns]
            
            for _, row in df.iterrows():
                # Use the first column as the sample identifier.
                name = str(row.iloc[0]).strip()
                if not name: continue
                
                try:
                    # Read the required fields by column name.
                    start_age = int(row['初诊年龄'])
                    end_age = int(row['治疗结束年龄'])
                    gender = str(row['性别']).strip()
                    ext_ortho = normalize_extraction_status(row.get('是否拔除正畸牙'))
                    ext_wisdom = normalize_extraction_status(row.get('是否拔除智齿'))
                    
                    info_map[name] = {
                        'start_age': start_age,
                        'end_age': end_age,
                        'duration': max(0, end_age - start_age),
                        'gender': gender,
                        'ext_ortho': ext_ortho,
                        'ext_wisdom': ext_wisdom
                    }
                except Exception:
                    continue
            return info_map
        except Exception as e:
            print(f"Error reading info csv: {e}")
            return {}

    def _load_movement_csv(self, path: str) -> Dict:
        """Read movement metrics from an XLSX or CSV file."""
        print(f"Loading metrics from: {path}")
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                try: df = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError: df = pd.read_csv(path, encoding='gbk')

            name_col = None
            for col in df.columns:
                if "姓名" in str(col) or "name" == str(col).lower():
                    name_col = col
                    break
            
            if name_col is None:
                name_col = df.columns[0]
                print(f"Warning: '姓名' column not found, using first column '{name_col}' as ID.")

            metric_names = []
            for col in df.columns:
                if col == name_col: continue
                c_str = str(col)
                if "Unnamed" in c_str or c_str == "": continue
                
                is_metric = False
                if "(°)" in c_str or "(mm)" in c_str: is_metric = True
                elif "Holdaway" in c_str: is_metric = True
                
                if is_metric: metric_names.append(c_str)
            
            if not metric_names:
                for col in df.columns:
                    c_str = str(col)
                    if col != name_col and "Unnamed" not in c_str and c_str.strip() != "":
                        metric_names.append(c_str)

            print(f"Found {len(metric_names)} metrics: {metric_names}")
            if len(metric_names) == 0: return None

            data_map = {}
            raw_values = []
            
            for _, row in df.iterrows():
                sid = str(row[name_col]).strip()
                if not sid or sid.lower() == 'nan': continue
                if sid in self.exclude_patients: continue

                vals = []
                for m in metric_names:
                    val = row[m]
                    try:
                        val = float(val)
                        if np.isnan(val): val = 0.0
                    except: val = 0.0
                    vals.append(val)
                
                vec = np.array(vals, dtype=np.float32)
                data_map[sid] = vec
                raw_values.append(vec)
            
            if raw_values:
                all_data = np.stack(raw_values, axis=0)
                mean_vec = np.mean(all_data, axis=0)
                std_vec = np.std(all_data, axis=0)
            else:
                mean_vec = np.zeros(len(metric_names))
                std_vec = np.ones(len(metric_names))

            return {'data': data_map, 'mean': mean_vec, 'std': std_vec, 'names': metric_names}

        except Exception as e:
            print(f"Error reading movement file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, pre_path, post_path = self.items[idx]
        pre_name = os.path.basename(pre_path)
        post_name = os.path.basename(post_path)

        # -----------------------------------------------------------
            # 1. Preprocess images: read, scale, rotate, register, and crop.
        # -----------------------------------------------------------
        img_pre_uint8 = read_image_gray(pre_path)
        img_post_uint8 = read_image_gray(post_path)

        lm_pre_map = self.pre_lm_dict.get(pre_name)
        if lm_pre_map is None: lm_pre_map = self.pre_lm_dict.get(os.path.splitext(pre_name)[0])
        
        lm_post_map = self.post_lm_dict.get(post_name)
        if lm_post_map is None: lm_post_map = self.post_lm_dict.get(os.path.splitext(post_name)[0])

        if lm_pre_map is None or lm_post_map is None:
            raise RuntimeError(f"Missing landmarks for {sid}")

        pre_names = list(lm_pre_map.keys())
        post_names = list(lm_post_map.keys())
        pre_arr = np.stack([lm_pre_map[n] for n in pre_names], axis=0)
        post_arr = np.stack([lm_post_map[n] for n in post_names], axis=0)

        # Scale.
        img_pre_s, pre_arr_s, _ = scale_by_ruler(img_pre_uint8, pre_arr, pre_names, self.target_mm_per_px, self.ruler_mm)
        img_post_s, post_arr_s, _ = scale_by_ruler(img_post_uint8, post_arr, post_names, self.target_mm_per_px, self.ruler_mm)

        # Rotate.
        rot_pre = float(self.rot_map.get(pre_name, 0.0))
        rot_post = float(self.rot_map.get(post_name, 0.0))
        img_pre_r, pre_arr_r = rotate_image_and_points(img_pre_s, pre_arr_s, rot_pre)
        img_post_r, post_arr_r = rotate_image_and_points(img_post_s, post_arr_s, rot_post)
        
        lm_pre_map_r = {n: p for n, p in zip(pre_names, pre_arr_r)}
        lm_post_map_r = {n: p for n, p in zip(post_names, post_arr_r)}

        # Register.
        pre_stable = pick_points_by_names(lm_pre_map_r, STABLE_POINT_NAMES)
        post_stable = pick_points_by_names(lm_post_map_r, STABLE_POINT_NAMES)
        if pre_stable.shape[0] < 2 or post_stable.shape[0] < 2:
            pre_stable, post_stable = pre_arr_r, post_arr_r
        M = estimate_similarity(post_stable, pre_stable)
        
        # Transform landmarks.
        post_arr_reg = warp_points_with_M(post_arr_r, M)
        # Transform the image.
        if M is not None:
            img_post_reg = cv2.warpAffine(img_post_r, M, (img_pre_r.shape[1], img_pre_r.shape[0]), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_post_reg = img_post_r

        # Compute crop parameters.
        crop_pts_pre = pick_points_by_names(lm_pre_map_r, CROP_POINT_NAMES)
        if crop_pts_pre.shape[0] == 0:
            crop_pts_pre = pre_arr_r[~np.isnan(pre_arr_r).any(axis=1)]
        h_pre_r, w_pre_r = img_pre_r.shape[:2]
        rect_syn = bbox_from_points(crop_pts_pre, self.crop_margin, (w_pre_r, h_pre_r))

        # Crop and resize.
        img_pre_cB, pre_all_B = crop_and_resize_with_points(img_pre_r, pre_arr_r, rect_syn, self.size)
        img_post_cB, post_all_B = crop_and_resize_with_points(img_post_reg, post_arr_reg, rect_syn, self.size)
        
        pt_map_B = {n: p for n, p in zip(pre_names, pre_all_B)}

        # -----------------------------------------------------------
        # 2. Generate the edge condition (combined Canny map).
        # -----------------------------------------------------------
        canny_soft_full = cv2.Canny(img_pre_cB, self.canny_low, self.canny_high)
        mask_corridor = np.zeros(img_pre_cB.shape, dtype=np.uint8)
        key_points_coords = pick_points_by_names(pt_map_B, ['Prn', 'Sn'])
        valid_key_pts = key_points_coords[~np.isnan(key_points_coords).any(axis=1)].astype(np.int32)
        for pt in valid_key_pts:
            cv2.circle(mask_corridor, tuple(pt), 7, 255, -1, lineType=cv2.LINE_AA)
        for curve_names in [MASK_SOFT_TISSUE_NAMES]:
            pts = pick_points_by_names(pt_map_B, curve_names)
            valid_pts = pts[~np.isnan(pts).any(axis=1)].astype(np.int32)
            if valid_pts.shape[0] > 1:
                cv2.polylines(mask_corridor, [valid_pts], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        k_size = self.mask_dilate_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        mask_corridor = cv2.dilate(mask_corridor, kernel, iterations=1)
        canny_soft_masked = cv2.bitwise_and(canny_soft_full, canny_soft_full, mask=mask_corridor)
        if self.min_contour_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canny_soft_masked, 8, cv2.CV_32S)
            final_soft = np.zeros_like(canny_soft_masked)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_contour_area:
                    final_soft[labels == i] = 255
        else:
            final_soft = canny_soft_masked
        combined_canny = final_soft
        
        # -----------------------------------------------------------
        # 3. Generate the landmark heatmap condition.
        # -----------------------------------------------------------
        heatmap_np = generate_heatmap(self.size, pt_map_B, IMPORTANT_POINT_NAMES, sigma=self.heatmap_sigma)
        
        post_img01 = apply_clahe_and_normalize(img_post_cB)
        target_tensor = to_tensor_img(post_img01) 

        canny_img01 = apply_clahe_and_normalize(combined_canny)
        conditioning_tensor = to_tensor_img(canny_img01) 
        
        heatmap_3c = np.stack([heatmap_np]*3, axis=-1)
        conditioning_heatmap_tensor = torch.from_numpy(heatmap_3c.transpose(2, 0, 1))

        # 4. Pretreatment image values.
        pre_img01 = apply_clahe_and_normalize(img_pre_cB)
        pre_tensor = to_tensor_img(pre_img01)

        # -----------------------------------------------------------
        # 5. Get normalized movement conditions.
        # -----------------------------------------------------------
        origin_vec = None
        movement_vec_norm = torch.zeros(0, dtype=torch.float32)
        if self.movement_info is not None:
            vec = self.movement_info['data'].get(sid)
            
            if vec is None:
                # Replace missing values with the mean, which becomes zero after normalization.
                print(f"Warning: No movement data for {sid}, using zeros.")
                num_metrics = len(self.movement_info['names'])
                vec = self.movement_info['mean'] 
            
            origin_vec = vec.copy() if isinstance(vec, np.ndarray) else vec
            vec_norm = (vec - self.movement_info['mean']) / (self.movement_info['std'] + 1e-6)
            movement_vec_norm = torch.from_numpy(vec_norm).float()

        # -----------------------------------------------------------
        # 6. Build the prompt with a structured order and style description at the end.
        # -----------------------------------------------------------
        
        # 1. Define the shared style suffix used in training and inference.
        shared_style = (
            "Lateral cephalometric radiograph, raw medical image, "
            "film grain, sharp focus, X-ray contrast"
        )

        info = self.info_map.get(sid)
        
        if info:
            gender_en = get_gender_en(info['gender'])
            start_age = info['start_age']
            end_age = info['end_age']
            
            ext_ortho = info.get('ext_ortho')
            ext_wisdom = info.get('ext_wisdom')
            if ext_ortho is None or ext_wisdom is None:
                extraction_prompt = "extraction information unavailable"
            else:
                arch_desc = "shortened dental arch" if ext_ortho == '1' else "intact dental arch"
                wisdom_desc = "missing third molars" if ext_wisdom == '1' else "third molars present"
                extraction_prompt = f"{arch_desc}, {wisdom_desc}"
            
            # 2. ControlNet maps pretreatment to post-treatment, so the prompt describes the outcome.
            # Structure: outcome -> treatment actions -> aesthetic goal -> constraints -> style.
            prompt = (f"Orthodontic treatment result, "                  # Outcome.
                      f"{extraction_prompt}, "                           # Extraction status.
                      f"retract incisors, align dentition, "             # Dental actions.
                      f"reduce soft tissue convexity, "                  # Soft-tissue action.
                      f"improve facial profile aesthetics, "             # Aesthetic goal.
                      f"{gender_en}, treatment from {int(start_age)} to {int(end_age)} years old, " # Constraints.
                      f"{shared_style}")                                 # Style suffix.
        else:
            # 3. Use a generic prompt when the information CSV is unavailable.
            prompt = f"Orthodontic treatment result, {shared_style}"
            
        # 7. Metadata
        (x, y, w, h) = rect_syn
        w_mm = w * self.target_mm_per_px
        h_mm = h * self.target_mm_per_px
        mm_per_px_x = w_mm / self.size
        mm_per_px_y = h_mm / self.size

        return {
            # Core training data.
            "pixel_values": target_tensor,                  # GT (Post Image)
            "conditioning_pixel_values": conditioning_tensor, # Canny of Pre (Source)
            "conditioning_heatmap": conditioning_heatmap_tensor, # Heatmap of Pre
            "conditioning_metrics": movement_vec_norm,      # Movement Metrics
            "metric_origin": origin_vec,
            "prompt": prompt,                               
            
            # Inputs for image-to-image processing and validation.
            "pre_pixel_values": pre_tensor,                 # Pretreatment image.
            
            # Metadata.
            "sid": sid,
            "name_pre": pre_name,
            "mm_per_px": (mm_per_px_x, mm_per_px_y),
            "gt_points": {n: p for n, p in zip(post_names, post_all_B)},
            "pre_points": {n: p for n, p in zip(pre_names, pre_all_B)}
        }

class ControlNetInpaintingDataset(Dataset):
    """Dataset for ControlNet inpainting with soft-tissue Canny edges and landmark masks."""
    exts = ('.bmp', '.jpg', '.jpeg', '.png')

    def __init__(self, root, size=512, 
                 rotation_csv=None, pre_csv=None, post_csv=None,
                 movement_csv=None, information_csv=None,
                 crop_margin=0.10, target_mm_per_px=0.1, ruler_mm=10.0,
                 canny_low=25, canny_high=40,
                 mask_dilate_kernel_size=15, min_contour_area=15):
        
        self.data_root = "data"
        self.root = root
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)
        
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.mask_dilate_kernel_size = mask_dilate_kernel_size | 1
        self.min_contour_area = min_contour_area
        
        # 1. Find paired images.
        img_root = root 
        pairs = {}
        for r, _, fs in os.walk(img_root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    p = os.path.join(r, f)
                    stem = os.path.splitext(f)[0]
                    if is_pre_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['pre'] = p
                    elif is_post_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['post'] = p
        
        self.items = [(sid, d['pre'], d['post']) for sid, d in pairs.items() if 'pre' in d and 'post' in d]
        
        # 2. Load CSV metadata.
        try:
            rot_csv_path = rotation_csv or os.path.join(self.data_root, "rotation.csv")
            pre_csv_path = pre_csv or os.path.join(self.data_root, "pre.csv")
            post_csv_path = post_csv or os.path.join(self.data_root, "post.csv")
            
            self.rot_map = read_rotation_csv(rot_csv_path)
            self.pre_lm_dict, _ = read_landmarks_with_names(pre_csv_path)
            self.post_lm_dict, _ = read_landmarks_with_names(post_csv_path)
        except Exception as e:
            print(f"Error loading CSV metadata: {e}")
            self.rot_map = {}
            self.pre_lm_dict, _ = {}, {}
            self.post_lm_dict, _ = {}, {}

        # 3. Load movement metrics.
        self.movement_info = None
        if movement_csv and os.path.exists(movement_csv):
            print(f"Loading movement metrics from: {movement_csv}")
            self.movement_info = self._load_movement_csv(movement_csv)
        else:
            print("Warning: No movement_csv provided or found.")

        # 4. Load patient information.
        self.info_map = {}
        info_csv_path = information_csv or os.path.join(root, "Information.csv")
        if os.path.exists(info_csv_path):
            print(f"Loading patient info from: {info_csv_path}")
            self.info_map = self._load_information_csv(info_csv_path)
        else:
            print(f"Warning: {info_csv_path} not found!")

    def _load_information_csv(self, path):
        info_map = {}
        try:
            try: df = pd.read_csv(path, encoding='utf-8-sig')
            except: df = pd.read_csv(path, encoding='gbk')
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                name = str(row.iloc[0]).strip()
                if not name: continue
                try:
                    start_age = int(row['初诊年龄'])
                    end_age = int(row['治疗结束年龄'])
                    gender = str(row['性别']).strip()
                    info_map[name] = {
                        'start_age': start_age,
                        'end_age': end_age,
                        'duration': max(0, end_age - start_age),
                        'gender': gender
                    }
                except Exception: continue
            return info_map
        except Exception as e:
            print(f"Error reading info csv: {e}"); return {}

    def _load_movement_csv(self, path: str) -> Dict:
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                try: df = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError: df = pd.read_csv(path, encoding='gbk')

            name_col = None
            for col in df.columns:
                if "姓名" in str(col) or "name" == str(col).lower():
                    name_col = col; break
            if name_col is None: name_col = df.columns[0]

            metric_names = []
            for col in df.columns:
                if col == name_col: continue
                c_str = str(col)
                if "Unnamed" in c_str or c_str == "": continue
                is_metric = False
                if "(°)" in c_str or "(mm)" in c_str or "Holdaway" in c_str: is_metric = True
                if is_metric: metric_names.append(c_str)
            if not metric_names:
                for col in df.columns:
                    c_str = str(col)
                    if col != name_col and "Unnamed" not in c_str and c_str.strip() != "":
                        metric_names.append(c_str)
            if len(metric_names) == 0: return None

            data_map = {}
            raw_values = []
            for _, row in df.iterrows():
                sid = str(row[name_col]).strip()
                if not sid or sid.lower() == 'nan': continue
                vals = []
                for m in metric_names:
                    val = row[m]
                    try: val = float(val); val = 0.0 if np.isnan(val) else val
                    except: val = 0.0
                    vals.append(val)
                vec = np.array(vals, dtype=np.float32)
                data_map[sid] = vec
                raw_values.append(vec)
            
            if raw_values:
                all_data = np.stack(raw_values, axis=0)
                mean_vec = np.mean(all_data, axis=0)
                std_vec = np.std(all_data, axis=0)
            else:
                mean_vec = np.zeros(len(metric_names)); std_vec = np.ones(len(metric_names))

            return {'data': data_map, 'mean': mean_vec, 'std': std_vec, 'names': metric_names}
        except Exception as e:
            print(f"Error reading movement file: {e}"); return None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, pre_path, post_path = self.items[idx]
        pre_name = os.path.basename(pre_path)
        post_name = os.path.basename(post_path)

        # -----------------------------------------------------------
        # 1. Preprocess images: scale, rotate, register, and crop.
        # -----------------------------------------------------------
        img_pre_uint8 = read_image_gray(pre_path)
        img_post_uint8 = read_image_gray(post_path)

        lm_pre_map = self.pre_lm_dict.get(pre_name, self.pre_lm_dict.get(os.path.splitext(pre_name)[0]))
        lm_post_map = self.post_lm_dict.get(post_name, self.post_lm_dict.get(os.path.splitext(post_name)[0]))

        if lm_pre_map is None or lm_post_map is None:
            raise RuntimeError(f"Missing landmarks for {sid}")

        pre_names = list(lm_pre_map.keys())
        post_names = list(lm_post_map.keys())
        pre_arr = np.stack([lm_pre_map[n] for n in pre_names], axis=0)
        post_arr = np.stack([lm_post_map[n] for n in post_names], axis=0)

        # Scale
        img_pre_s, pre_arr_s, _ = scale_by_ruler(img_pre_uint8, pre_arr, pre_names, self.target_mm_per_px, self.ruler_mm)
        img_post_s, post_arr_s, _ = scale_by_ruler(img_post_uint8, post_arr, post_names, self.target_mm_per_px, self.ruler_mm)

        # Rotate
        rot_pre = float(self.rot_map.get(pre_name, 0.0))
        rot_post = float(self.rot_map.get(post_name, 0.0))
        img_pre_r, pre_arr_r = rotate_image_and_points(img_pre_s, pre_arr_s, rot_pre)
        img_post_r, post_arr_r = rotate_image_and_points(img_post_s, post_arr_s, rot_post)
        
        lm_pre_map_r = {n: p for n, p in zip(pre_names, pre_arr_r)}
        lm_post_map_r = {n: p for n, p in zip(post_names, post_arr_r)}

        # Register
        pre_stable = pick_points_by_names(lm_pre_map_r, STABLE_POINT_NAMES)
        post_stable = pick_points_by_names(lm_post_map_r, STABLE_POINT_NAMES)
        if pre_stable.shape[0] < 2 or post_stable.shape[0] < 2:
            pre_stable, post_stable = pre_arr_r, post_arr_r
        M = estimate_similarity(post_stable, pre_stable)
        
        post_arr_reg = warp_points_with_M(post_arr_r, M)
        if M is not None:
            img_post_reg = cv2.warpAffine(img_post_r, M, (img_pre_r.shape[1], img_pre_r.shape[0]), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_post_reg = img_post_r

        # Crop
        crop_pts_pre = pick_points_by_names(lm_pre_map_r, CROP_POINT_NAMES)
        if crop_pts_pre.shape[0] == 0:
            crop_pts_pre = pre_arr_r[~np.isnan(pre_arr_r).any(axis=1)]
        h_pre_r, w_pre_r = img_pre_r.shape[:2]
        rect_syn = bbox_from_points(crop_pts_pre, self.crop_margin, (w_pre_r, h_pre_r))

        img_pre_cB, pre_all_B = crop_and_resize_with_points(img_pre_r, pre_arr_r, rect_syn, self.size)
        img_post_cB, post_all_B = crop_and_resize_with_points(img_post_reg, post_arr_reg, rect_syn, self.size)
        
        pt_map_B = {n: p for n, p in zip(pre_names, pre_all_B)}

        # -----------------------------------------------------------
        # 2. Generate the inpainting mask for repainting and edge filtering.
        # -----------------------------------------------------------
        mask_np = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Collect key soft-tissue and dental landmarks.
        important_pts_list = []
        for name in IMPORTANT_POINT_NAMES:
            pt = pt_map_B.get(name)
            if pt is not None and not np.isnan(pt).any():
                important_pts_list.append(pt)
        
        if len(important_pts_list) >= 3:
            pts_array = np.array(important_pts_list, dtype=np.int32)
            # Compute and fill the convex hull.
            hull = cv2.convexHull(pts_array)
            cv2.fillPoly(mask_np, [hull], 255)
            # Dilate the mask to cover nearby edges.
            k_size = 30 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        else:
            mask_np[:] = 255
            
        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0) 

        # -----------------------------------------------------------
        # 3. Generate the edge condition with two-stage mask filtering.
        # Step A: Apply Canny to the full image.
        canny_soft_full = cv2.Canny(img_pre_cB, self.canny_low, self.canny_high)
        
        # Step B: Build the corridor mask from Prn, Sn, and the soft-tissue curve.
        mask_corridor = np.zeros(img_pre_cB.shape, dtype=np.uint8)
        key_points_coords = pick_points_by_names(pt_map_B, ['Prn', 'Sn'])
        valid_key_pts = key_points_coords[~np.isnan(key_points_coords).any(axis=1)].astype(np.int32)
        for pt in valid_key_pts:
            cv2.circle(mask_corridor, tuple(pt), 7, 255, -1, lineType=cv2.LINE_AA)
        
        for curve_names in [MASK_SOFT_TISSUE_NAMES]:
            pts = pick_points_by_names(pt_map_B, curve_names)
            valid_pts = pts[~np.isnan(pts).any(axis=1)].astype(np.int32)
            if valid_pts.shape[0] > 1:
                cv2.polylines(mask_corridor, [valid_pts], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
        
        k_size_c = self.mask_dilate_kernel_size
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size_c, k_size_c))
        mask_corridor = cv2.dilate(mask_corridor, kernel_c, iterations=1)
        
        # Step C: Keep edges inside the soft-tissue corridor.
        canny_soft_masked = cv2.bitwise_and(canny_soft_full, canny_soft_full, mask=mask_corridor)
        
        # Step D: Remove small connected components.
        if self.min_contour_area > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canny_soft_masked, 8, cv2.CV_32S)
            final_soft = np.zeros_like(canny_soft_masked)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_contour_area:
                    final_soft[labels == i] = 255
        else:
            final_soft = canny_soft_masked
            
        # Step E: Apply the inpainting mask; retain edges inside both masks.
        combined_canny = cv2.bitwise_and(final_soft, final_soft, mask=mask_np)

        # Convert to a tensor.
        canny_img01 = apply_clahe_and_normalize(combined_canny)
        conditioning_tensor = to_tensor_img(canny_img01) 

        # -----------------------------------------------------------
        # 4. Normalize images.
        # -----------------------------------------------------------
        post_img01 = apply_clahe_and_normalize(img_post_cB)
        target_tensor = to_tensor_img(post_img01) # GT

        pre_img01 = apply_clahe_and_normalize(img_pre_cB)
        pre_tensor = to_tensor_img(pre_img01) # Inpainting base image.

        # -----------------------------------------------------------
        # 5. Metrics & Prompt
        # -----------------------------------------------------------
        origin_vec = None
        movement_vec_norm = torch.zeros(0, dtype=torch.float32)
        if self.movement_info is not None:
            vec = self.movement_info['data'].get(sid, self.movement_info['mean'])
            origin_vec = vec.copy() if isinstance(vec, np.ndarray) else vec
            vec_norm = (vec - self.movement_info['mean']) / (self.movement_info['std'] + 1e-6)
            movement_vec_norm = torch.from_numpy(vec_norm).float()

        base_style = "Lateral cephalometric radiograph, high contrast medical imaging"
        prompt = f"{base_style}, orthodontic treatment result"

        info = self.info_map.get(sid)
        if info:
            gender_en = get_gender_en(info['gender'])
            start_age = info['start_age']
            end_age = info['end_age']
            prompt = (f"{base_style}, "
                      f"simulate orthodontic treatment result, "
                      f"retract incisors and align dentition, "
                      f"reduce soft tissue convexity, "
                      f"improve facial profile aesthetics, "
                      f"{gender_en}, treatment from {int(start_age)} to {int(end_age)} years old")

        # 7. Metadata
        (x, y, w, h) = rect_syn
        w_mm = w * self.target_mm_per_px
        h_mm = h * self.target_mm_per_px
        mm_per_px_x = w_mm / self.size
        mm_per_px_y = h_mm / self.size

        return {
            "pixel_values": target_tensor,                  # GT
            "conditioning_pixel_values": conditioning_tensor, # Canny edges filtered by both masks.
            "pixel_mask": mask_tensor,                      # Inpainting Mask
            "conditioning_metrics": movement_vec_norm,
            "metric_origin": origin_vec,
            "prompt": prompt,
            "pre_pixel_values": pre_tensor,
            "sid": sid,
            "name_pre": pre_name,
            "mm_per_px": (mm_per_px_x, mm_per_px_y),
            "gt_points": {n: p for n, p in zip(post_names, post_all_B)},
            "pre_points": {n: p for n, p in zip(pre_names, pre_all_B)}
        }
        
class InstructPix2PixDataset(Dataset):
    """Dataset for InstructPix2Pix.

    Inputs include the pretreatment image, a prompt built from patient attributes,
    and numeric skeletal-change conditions. The target is the post-treatment image.
    """
    exts = ('.bmp', '.jpg', '.jpeg', '.png')

    def __init__(self, root, size=512, 
                 rotation_csv=None, pre_csv=None, post_csv=None,
                 movement_csv=None, information_csv=None, # Patient information file.
                 crop_margin=0.10, target_mm_per_px=0.1, ruler_mm=10.0):
        
        self.data_root = "data"
        self.root = root
        self.size = size
        self.crop_margin = crop_margin
        self.target_mm_per_px = float(target_mm_per_px)
        self.ruler_mm = float(ruler_mm)

        # 1. Find paired images.
        img_root = root
        pairs = {}
        for r, _, fs in os.walk(img_root):
            for f in fs:
                if f.lower().endswith(self.exts):
                    p = os.path.join(r, f)
                    stem = os.path.splitext(f)[0]
                    if is_pre_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['pre'] = p
                    elif is_post_name(f):
                        sid = stem[:-2]; pairs.setdefault(sid, {})['post'] = p
        
        self.items = [(sid, d['pre'], d['post']) for sid, d in pairs.items() if 'pre' in d and 'post' in d]
        
        # 2. Load rotation and landmark metadata.
        try:
            rot_csv_path = rotation_csv or os.path.join(self.data_root, "rotation.csv")
            pre_csv_path = pre_csv or os.path.join(self.data_root, "pre.csv")
            post_csv_path = post_csv or os.path.join(self.data_root, "post.csv")
            
            self.rot_map = read_rotation_csv(rot_csv_path)
            self.pre_lm_dict, _ = read_landmarks_with_names(pre_csv_path)
            self.post_lm_dict, _ = read_landmarks_with_names(post_csv_path)
        except Exception as e:
            print(f"Error loading metadata CSVs: {e}")
            self.rot_map, self.pre_lm_dict, self.post_lm_dict = {}, {}, {}

        # 3. Load movement metrics as numeric conditions.
        self.movement_info = None
        if movement_csv and os.path.exists(movement_csv):
            self.movement_info = self._load_movement_csv(movement_csv)
        else:
            print("Warning: movement_csv not found.")

        # 4. Load patient information for prompt construction.
        self.patient_info = {}
        if information_csv and os.path.exists(information_csv):
            print(f"Loading patient info from: {information_csv}")
            self.patient_info = self._load_information_csv(information_csv)
        else:
            print(f"Warning: information_csv not found. Default prompts will be used.")

    def _load_information_csv(self, path):
        """Read sample identifiers, initial age, treatment-end age, and gender from Information.csv."""
        info_map = {}
        try:
            # Try common file encodings.
            try: df = pd.read_csv(path, encoding='utf-8-sig')
            except: df = pd.read_csv(path, encoding='gbk')
            
            # Strip whitespace from column names.
            df.columns = [c.strip() for c in df.columns]
            
            for _, row in df.iterrows():
                # Use the first column as the sample identifier.
                name = str(row.iloc[0]).strip()
                if not name: continue
                
                try:
                    # Read the required fields.
                    start_age = int(row['初诊年龄'])
                    end_age = int(row['治疗结束年龄'])
                    gender = str(row['性别']).strip()
                    
                    info_map[name] = {
                        'start_age': start_age,
                        'end_age': end_age,
                        'duration': max(0, end_age - start_age),
                        'gender': gender
                    }
                except Exception:
                    continue
            return info_map
        except Exception as e:
            print(f"Error reading info csv: {e}")
            return {}

    def _load_movement_csv(self, path: str) -> Dict:
        """Read movement metrics from an XLSX or CSV file using pandas."""
        print(f"Loading metrics from: {path}")
        try:
            # 1. Select the reader based on the file extension.
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                # Try common encodings for CSV input.
                try:
                    df = pd.read_csv(path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df = pd.read_csv(path, encoding='gbk')

            # 2. Find the sample-identifier column.
            name_col = None
            # Prefer a column named "name" or containing the configured identifier label.
            for col in df.columns:
                if "姓名" in str(col) or "name" == str(col).lower():
                    name_col = col
                    break
            
            # If no identifier column is found, use the first column.
            if name_col is None:
                name_col = df.columns[0]
                print(f"Warning: '姓名' column not found, using first column '{name_col}' as ID.")

            # 3. Find metric columns, prioritizing columns with units.
            metric_names = []
            for col in df.columns:
                if col == name_col: continue
                c_str = str(col)
                if "Unnamed" in c_str or c_str == "":
                    continue
                # Include columns with units or recognized unitless metrics such as Holdaway.
                is_metric = False
                if "(°)" in c_str or "(mm)" in c_str:
                    is_metric = True
                elif "Holdaway" in c_str:
                    is_metric = True
                # Add other recognized unitless metrics (such as ratios or indices) here if needed.
                
                if is_metric:
                    metric_names.append(c_str)
            
            # If no unit-bearing columns are found, use all non-identifier, non-empty columns.
            if not metric_names:
                for col in df.columns:
                    c_str = str(col)
                    if col != name_col and "Unnamed" not in c_str and c_str.strip() != "":
                        metric_names.append(c_str)

            print(f"Found {len(metric_names)} metrics: {metric_names}")
            if len(metric_names) == 0:
                print("Error: No valid metric columns found.")
                return None

            # 4. Extract metric values.
            data_map = {}
            raw_values = []
            
            # Process each row.
            for _, row in df.iterrows():
                sid = str(row[name_col]).strip()
                # Skip invalid identifiers.
                if not sid or sid.lower() == 'nan': continue

                vals = []
                for m in metric_names:
                    val = row[m]
                    try:
                        val = float(val)
                        if np.isnan(val): val = 0.0
                    except:
                        val = 0.0
                    vals.append(val)
                
                vec = np.array(vals, dtype=np.float32)
                data_map[sid] = vec
                raw_values.append(vec)
            
            # 5. Compute mean and standard deviation for normalization.
            if raw_values:
                all_data = np.stack(raw_values, axis=0)
                mean_vec = np.mean(all_data, axis=0)
                std_vec = np.std(all_data, axis=0)
            else:
                mean_vec = np.zeros(len(metric_names))
                std_vec = np.ones(len(metric_names))

            return {
                'data': data_map,
                'mean': mean_vec,
                'std': std_vec,
                'names': metric_names
            }

        except Exception as e:
            print(f"Error reading movement file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def construct_prompt(self, sid):
        info = self.patient_info.get(sid)
        base_prompt = "Lateral cephalometric radiograph, high contrast medical imaging"
        
        if not info:
            return f"{base_prompt}, orthodontic treatment result."
        
        gender_cn = info['gender']
        if '男' in gender_cn: gender_en = "male"
        elif '女' in gender_cn: gender_en = "female"
        else: gender_en = "patient"
        
        prompt = (f"{base_prompt}, "                              # [1. Style anchor]
                  f"simulate orthodontic treatment result, "      # [2. Main instruction]
                  f"reduce soft tissue convexity, "               # [4. Soft-tissue action]
                  f"improve facial profile aesthetics, "          # [5. Aesthetic goal]
                  f"{gender_en}, treatment from {int(info['start_age'])} to {int(info['end_age'])} years old") # [6. Constraints]
        return prompt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, pre_path, post_path = self.items[idx]
        pre_name = os.path.basename(pre_path)
        post_name = os.path.basename(post_path)

        # -----------------------------------------------------------
        # 1. Preprocess images: scale, rotate, register, and crop.
        # -----------------------------------------------------------
        # Read images.
        img_pre_uint8 = read_image_gray(pre_path)
        img_post_uint8 = read_image_gray(post_path)

        # Load landmarks.
        lm_pre_map = self.pre_lm_dict.get(pre_name, self.pre_lm_dict.get(os.path.splitext(pre_name)[0]))
        lm_post_map = self.post_lm_dict.get(post_name, self.post_lm_dict.get(os.path.splitext(post_name)[0]))

        if lm_pre_map is None or lm_post_map is None:
            # Skip samples without landmarks; the DataLoader collate_fn must handle None.
            return None 

        pre_names = list(lm_pre_map.keys())
        post_names = list(lm_post_map.keys())
        pre_arr = np.stack([lm_pre_map[n] for n in pre_names], axis=0)
        post_arr = np.stack([lm_post_map[n] for n in post_names], axis=0)

        # A. Scale.
        img_pre_s, pre_arr_s, _ = scale_by_ruler(img_pre_uint8, pre_arr, pre_names, self.target_mm_per_px, self.ruler_mm)
        img_post_s, post_arr_s, _ = scale_by_ruler(img_post_uint8, post_arr, post_names, self.target_mm_per_px, self.ruler_mm)

        # B. Rotate.
        rot_pre = float(self.rot_map.get(pre_name, 0.0))
        rot_post = float(self.rot_map.get(post_name, 0.0))
        img_pre_r, pre_arr_r = rotate_image_and_points(img_pre_s, pre_arr_s, rot_pre)
        img_post_r, post_arr_r = rotate_image_and_points(img_post_s, post_arr_s, rot_post)
        
        lm_pre_map_r = {n: p for n, p in zip(pre_names, pre_arr_r)}
        lm_post_map_r = {n: p for n, p in zip(post_names, post_arr_r)}

        # C. Register the post-treatment image to the pretreatment image.
        pre_stable = pick_points_by_names(lm_pre_map_r, STABLE_POINT_NAMES)
        post_stable = pick_points_by_names(lm_post_map_r, STABLE_POINT_NAMES)
        
        if pre_stable.shape[0] < 2 or post_stable.shape[0] < 2:
            pre_stable, post_stable = pre_arr_r, post_arr_r
            
        M = estimate_similarity(post_stable, pre_stable)
        post_arr_reg = warp_points_with_M(post_arr_r, M)
        if M is not None:
            img_post_reg = cv2.warpAffine(img_post_r, M, (img_pre_r.shape[1], img_pre_r.shape[0]), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_post_reg = img_post_r

        # D. Crop using a bounding box derived from pretreatment landmarks.
        crop_pts_pre = pick_points_by_names(lm_pre_map_r, CROP_POINT_NAMES)
        if crop_pts_pre.shape[0] == 0: 
            crop_pts_pre = pre_arr_r[~np.isnan(pre_arr_r).any(axis=1)]
            
        h_pre_r, w_pre_r = img_pre_r.shape[:2]
        rect_syn = bbox_from_points(crop_pts_pre, self.crop_margin, (w_pre_r, h_pre_r))

        img_pre_cB, pre_all_B = crop_and_resize_with_points(img_pre_r, pre_arr_r, rect_syn, self.size)
        img_post_cB, post_all_B = crop_and_resize_with_points(img_post_reg, post_arr_reg, rect_syn, self.size)
        
        # -----------------------------------------------------------
        # 2. Convert images to tensors in InstructPix2Pix format.
        # -----------------------------------------------------------
        # Post-treatment target image.
        post_img01 = apply_clahe_and_normalize(img_post_cB)
        target_tensor = to_tensor_img(post_img01) 

        # Pretreatment conditioning image.
        pre_img01 = apply_clahe_and_normalize(img_pre_cB)
        pre_tensor = to_tensor_img(pre_img01)

        # -----------------------------------------------------------
        # 3. Extract metric conditions.
        # -----------------------------------------------------------
        movement_vec_norm = torch.zeros(0, dtype=torch.float32)
        if self.movement_info is not None:
            vec = self.movement_info['data'].get(sid)
            if vec is None: vec = self.movement_info['mean']
            
            # Z-score normalization.
            vec_norm = (vec - self.movement_info['mean']) / (self.movement_info['std'] + 1e-6)
            movement_vec_norm = torch.from_numpy(vec_norm).float()

        # -----------------------------------------------------------
        # 4. Build the prompt.
        # -----------------------------------------------------------
        prompt_str = self.construct_prompt(sid)

        # -----------------------------------------------------------
        # 5. Return the sample.
        # -----------------------------------------------------------
        # Compute mm_per_px for validation.
        (x, y, w, h) = rect_syn
        w_mm = w * self.target_mm_per_px
        h_mm = h * self.target_mm_per_px
        
        return {
            "pixel_values": target_tensor,              # Ground-truth target for loss computation.
            "conditioning_pixel_values": pre_tensor,    # Input image concatenated with the UNet input.
            "prompt": prompt_str,                       # Text condition for the text encoder.
            "conditioning_metrics": movement_vec_norm,  # Numeric condition for the Metric Projector.
            
            # Auxiliary metadata.
            "sid": sid,
            "mm_per_px": (w_mm / self.size, h_mm / self.size),
            "gt_points": {n: p for n, p in zip(post_names, post_all_B)},
            "pre_points": {n: p for n, p in zip(pre_names, pre_all_B)}
        }
