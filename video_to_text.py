#!/usr/bin/env python
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#######################################
#  SPyNet Optical Flow 모델 준비
#######################################

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"[INFO] Using device: {device}")

# --- Backwarp ---------------------------------------------------------
backwarp_grid_cache = {}

def backwarp(tenInput, tenFlow):

    key = (tenFlow.shape[2], tenFlow.shape[3], tenFlow.device)

    if key not in backwarp_grid_cache:
        H, W = tenFlow.shape[2], tenFlow.shape[3]
        tenHor = torch.linspace(-1.0, 1.0, W, device=tenFlow.device).view(
            1, 1, 1, W
        ).repeat(1, 1, H, 1)
        tenVer = torch.linspace(-1.0, 1.0, H, device=tenFlow.device).view(
            1, 1, H, 1
        ).repeat(1, 1, 1, W)
        backwarp_grid_cache[key] = torch.cat([tenHor, tenVer], 1)

    tenGrid = backwarp_grid_cache[key]

    tenFlowNorm = torch.cat(
        [
            tenFlow[:, 0:1] * (2.0 / (tenInput.shape[3] - 1.0)),
            tenFlow[:, 1:2] * (2.0 / (tenInput.shape[2] - 1.0)),
        ],
        1,
    )

    return torch.nn.functional.grid_sample(
        tenInput,
        (tenGrid + tenFlowNorm).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )

# --- SPyNet ------------------------------------------------------------
class Network(torch.nn.Module):
    def __init__(self, weight_path):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def forward(self, tenInput):
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(
                    [0.485, 0.456, 0.406],
                    dtype=tenInput.dtype,
                    device=tenInput.device,
                ).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(
                    [1 / 0.229, 1 / 0.224, 1 / 0.225],
                    dtype=tenInput.dtype,
                    device=tenInput.device,
                ).view(1, 3, 1, 1)
                return tenInput

        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(8, 32, 7, 1, 3),
                    torch.nn.ReLU(False),
                    torch.nn.Conv2d(32, 64, 7, 1, 3),
                    torch.nn.ReLU(False),
                    torch.nn.Conv2d(64, 32, 7, 1, 3),
                    torch.nn.ReLU(False),
                    torch.nn.Conv2d(32, 16, 7, 1, 3),
                    torch.nn.ReLU(False),
                    torch.nn.Conv2d(16, 2, 7, 1, 3),
                )

            def forward(self, x):
                return self.netBasic(x)

        self.netPreprocess = Preprocess()
        self.netBasic = torch.nn.ModuleList([Basic() for _ in range(6)])

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"[ERROR] SPyNet weight file not found:\n → {weight_path}"
            )

        print(f"[INFO] Loading SPyNet weights from: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        clean_dict = {k.replace("module", "net"): v for k, v in state_dict.items()}
        self.load_state_dict(clean_dict, strict=False)

    def forward(self, tenOne, tenTwo):
        tenOne_list = [self.netPreprocess(tenOne)]
        tenTwo_list = [self.netPreprocess(tenTwo)]

        for _ in range(5):
            if tenOne_list[0].shape[2] > 32:
                tenOne_list.insert(
                    0, torch.nn.functional.avg_pool2d(tenOne_list[0], 2)
                )
                tenTwo_list.insert(
                    0, torch.nn.functional.avg_pool2d(tenTwo_list[0], 2)
                )

        tenFlow = tenOne_list[0].new_zeros(
            [1, 2, tenOne_list[0].shape[2] // 2, tenOne_list[0].shape[3] // 2]
        )

        for i in range(len(tenOne_list)):
            tenUpsampled = (
                torch.nn.functional.interpolate(
                    tenFlow, scale_factor=2, mode="bilinear", align_corners=True
                )
                * 2.0
            )

            if tenUpsampled.shape[2] != tenOne_list[i].shape[2]:
                tenUpsampled = torch.nn.functional.pad(
                    tenUpsampled, [0, 0, 0, 1]
                )
            if tenUpsampled.shape[3] != tenOne_list[i].shape[3]:
                tenUpsampled = torch.nn.functional.pad(
                    tenUpsampled, [0, 1, 0, 0]
                )

            tenFlow = (
                self.netBasic[i](
                    torch.cat(
                        [
                            tenOne_list[i],
                            backwarp(tenTwo_list[i], tenUpsampled),
                            tenUpsampled,
                        ],
                        1,
                    )
                )
                + tenUpsampled
            )

        return tenFlow

netNetwork = None

def estimate_flow(tenOne, tenTwo):
    global netNetwork
    if netNetwork is None:
        netNetwork = Network("weights/spynet/network-sintel-final.pytorch").to(device).eval()

    H, W = tenOne.shape[1:]
    tenOne = tenOne.to(device).view(1, 3, H, W)
    tenTwo = tenTwo.to(device).view(1, 3, H, W)

    Hp = (H + 31) // 32 * 32
    Wp = (W + 31) // 32 * 32

    tenOne = torch.nn.functional.interpolate(tenOne, (Hp, Wp))
    tenTwo = torch.nn.functional.interpolate(tenTwo, (Hp, Wp))

    flow = torch.nn.functional.interpolate(
        netNetwork(tenOne, tenTwo), size=(H, W)
    )

    flow[:, 0] *= W / Wp
    flow[:, 1] *= H / Hp

    return flow[0].cpu()


###############################################
#  SPyNet → 스크린샷 저장 ONLY 버전
###############################################
def extract_screenshots(input_video, screenshot_dir):

    os.makedirs(screenshot_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    ret, prev = cap.read()
    prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)

    h, w = prev.shape[:2]
    scroll_acc = 0
    MIN_SCROLL = 0.3
    saved_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 중앙 crop
        prev_crop = prev_rgb[
            h // 4 : h * 3 // 4,
            w // 4 : w * 3 // 4
        ]
        curr_crop = frame_rgb[
            h // 4 : h * 3 // 4,
            w // 4 : w * 3 // 4
        ]

        prev_bgr = prev_crop[:, :, ::-1].copy()
        curr_bgr = curr_crop[:, :, ::-1].copy()

        tenOne = torch.from_numpy(prev_bgr.transpose(2, 0, 1)).float() / 255.0
        tenTwo = torch.from_numpy(curr_bgr.transpose(2, 0, 1)).float() / 255.0

        flow_crop = estimate_flow(tenOne, tenTwo).numpy().transpose(1, 2, 0)
        flow_crop[:, :, 0] = 0
        dy = np.median(flow_crop[:, :, 1])
        if abs(dy) < MIN_SCROLL:
            dy = 0

        scroll_acc += dy

        # 스크린샷 조건
        if abs(scroll_acc) > h * 0.75 or saved_count == 0:
            save_path = os.path.join(
                screenshot_dir, f"screenshot_{saved_count}.png"
            )
            cv2.imwrite(save_path, frame)
            saved_count += 1
            scroll_acc = 0

        prev_rgb = frame_rgb
        pbar.update(1)

    cap.release()

    print(f"[INFO] Saved screenshots: {saved_count}")


########################################################
#   EasyOCR 기반 카톡 OCR + 파싱 함수 불러오기
########################################################

from img_to_text import parse_chat_image, parse_kakao_dict


########################################################
#   🎯 전체 파이프라인: 영상 → OCR → 파싱 → 병합 출력
########################################################
def process_video_and_parse(video_path):

    # output dirs ---
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join("./runs", folder_name)
    screenshot_dir = os.path.join(save_root, "screenshots")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    print(f"Extracting screenshots to: {screenshot_dir}")
    extract_screenshots(video_path, screenshot_dir)

    # OCR + 파싱
    all_results = []

    print("Running OCR & parsing on all screenshots ...")

    png_list = sorted(
        [f for f in os.listdir(screenshot_dir) if f.endswith(".png")]
    )

    for img_file in png_list:
        img_path = os.path.join(screenshot_dir, img_file)

        ocr_pack = parse_chat_image(img_path)  # {'rec_texts':..., 'rec_polys':...}
        if isinstance(ocr_pack, str):
            continue

        parsed = parse_kakao_dict(ocr_pack, image_width=300)
        all_results.append(parsed)

    # 병합
    final_output = "\n".join(all_results)

    print("\n" + "=" * 60)
    print("   [최종 전체 병합 결과]")
    print("=" * 60)
    print(final_output)

    # 저장
    with open(os.path.join(save_root, "final_chat.txt"), "w") as f:
        f.write(final_output)

    print(f"결과 저장됨: {save_root}/final_chat.txt")


##########################################
#  MAIN
##########################################
if __name__ == "__main__":
    video = "Test_Dataset/slow_version.mp4"
    process_video_and_parse(video)
