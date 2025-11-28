import easyocr
import os
import shutil
import cv2
import numpy as np
import re
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import easyocr
import os




# =========================================================
# 0. EasyOCR 모델 세팅
# =========================================================

MODEL_DIR = ".weights/easyocr"
DETECTOR_SRC = "weights/easyocr/craft_mlt_25k.pth"
RECOG_SRC = "weights/easyocr/korean_g2.pth"

BASE_DIR = os.path.expanduser("~/.EasyOCR")
DETECTOR_DST = os.path.join(BASE_DIR, "model", "craft_mlt_25k.pth")
RECOG_DST = os.path.join(BASE_DIR, "model", "korean_g2.pth")

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

def safe_copy(src, dst):
    if not os.path.exists(dst):
        # print(f"모델 복사: {src} → {dst}")
        shutil.copy(src, dst)
    else:
        pass
        # print(f"✔ 이미 존재함: {dst}")

safe_copy(DETECTOR_SRC, DETECTOR_DST)
safe_copy(RECOG_SRC, RECOG_DST)

print("EasyOCR 모델 로딩 중...")
ocr = easyocr.Reader(
    ['ko'],
    gpu=False,
    download_enabled=False
)
print("로컬 EasyOCR 로딩 완료!")



# =========================================================
# 1. OCR 유틸
# =========================================================

def clean_text(text):
    text = re.sub(r"[|│]", "", text)
    return text.strip()

def is_noise(text):
    if not text:
        return True

    system_patterns = [
        r".*님이 들어왔습니다\.$",
        r".*님이 나갔습니다\.$",
        r".*님이 .*님을 초대했습니다\.$",
        r"^삭제된 메시지입니다\.$",
        r".*기프티콘을 보냈습니다.*",
        r".*송금했습니다.*",
        r"^톡게시판.*",
    ]
    for p in system_patterns:
        if re.match(p, text):
            return True

    ui_keywords = {
        "사진","동영상","음성메시지","보이스톡","페이스톡","라이브톡",
        "선물하기","송금","정산하기","프로필 보기","공지 등록","좋아요","공감",
        "안읽음","MY","채팅방","메뉴","전송","읽음"
    }
    if text in ui_keywords:
        return True

    # 숫자 '1', '2' 같은 것은 제거
    if text.replace(':','').isdigit() and len(text) < 3:
        return True

    return False

def format_time(ts_str):
    ts_str = ts_str.replace(" ", "")
    final_time = ts_str

    try:
        is_pm = "오후" in ts_str
        is_am = "오전" in ts_str

        m = re.search(r"(\d{1,2})[:\.,]?(\d{2})", ts_str)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            if is_pm and hour != 12:
                hour += 12
            if is_am and hour == 12:
                hour = 0
            final_time = f"{hour:02}:{minute:02}"
    except:
        pass

    return final_time



# =========================================================
# 2. OCR 이미지 → EasyOCR 결과 → PaddleOCR 형식으로 변환
# =========================================================

def parse_chat_image(img_path):

    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return "Error: 이미지 없음"

    h, w = img.shape[:2]

    # EasyOCR 정식 호출 방식
    result = ocr.readtext(img, detail=1)

    # EasyOCR 결과 → paddleOCR형 구조로 변환
    rec_texts = []
    rec_polys = []

    for box, text, conf in result:
        rec_texts.append(text)
        rec_polys.append(box)

    return {
        "rec_texts": rec_texts,
        "rec_polys": rec_polys,
        "width": w
    }



# =========================================================
# 3. KakaoTalk 파싱
# =========================================================

def parse_kakao_dict(ocr_result, image_width):

    rec_texts = ocr_result["rec_texts"]
    rec_polys = ocr_result["rec_polys"]

    center_x = image_width / 2

    time_regexs = [
        re.compile(r".*\d{1,2}:\d{2}$"),
        re.compile(r".*\d{1,2}\.\d{2}$"),
        re.compile(r".*\d{1,2},\d{2}$"),
        re.compile(r".*오[전후]\d{3,4}$")
    ]
    date_regex = re.compile(r"^20\d{2}[.년]\s*\d{1,2}[.월]\s*\d{1,2}")

    processed = []

    for text, poly in zip(rec_texts, rec_polys):
        text = clean_text(text)
        if not text or is_noise(text):
            continue

        poly_np = np.array(poly)
        y_center = (poly_np[:,1].min() + poly_np[:,1].max()) / 2
        x_left = poly_np[:,0].min()
        x_right = poly_np[:,0].max()

        processed.append({
            "text": text,
            "y_center": y_center,
            "x_left": x_left,
            "x_right": x_right,
            "is_timestamp": any(r.match(text) for r in time_regexs),
            "is_date": bool(date_regex.match(text))
        })

    if not processed:
        return "텍스트 없음"

    # Y 축 정렬
    processed.sort(key=lambda x: x["y_center"])

    grouped = []
    curr = [processed[0]]
    curr_y = processed[0]["y_center"]

    Y_TOL = 50

    # Y 기준 그룹
    for item in processed[1:]:
        if abs(item["y_center"] - curr_y) < Y_TOL:
            curr.append(item)
        else:
            _process_and_save_groups(grouped, curr)
            curr = [item]
            curr_y = item["y_center"]

    _process_and_save_groups(grouped, curr)

    # 날짜 찾기
    detected_dates = []
    for g in grouped:
        if g["type"] == "date":
            line_text = " ".join([i["text"] for i in g["items"]])
            m = re.search(r"(\d{4})[.년]\s*(\d+)[.월]\s*(\d+)", line_text)
            if m:
                y,m2,d = map(int,m.groups())
                detected_dates.append(datetime(y,m2,d))

    if detected_dates:
        current_date = detected_dates[0].strftime("%Y. %-m. %-d.")
    else:
        current_date = "2000. 1. 1."

    final_chat = []
    pending = []

    for g in grouped:

        if g["type"] == "date":
            raw = " ".join([i["text"] for i in g["items"]])
            m = re.search(r"(\d{4})[.년]\s*(\d+)[.월]\s*(\d+)", raw)
            if m:
                current_date = f"{m.group(1)}. {m.group(2)}. {m.group(3)}."

        elif g["type"] == "text":
            pending.append(g)

        elif g["type"] == "timestamp":

            if not pending:
                continue

            raw_ts = " ".join([i["text"] for i in g["items"]])
            ts = format_time(raw_ts)
            full_ts = f"{current_date} {ts}"

            first = pending[0]
            center = (min(i["x_left"] for i in first["items"]) +
                      max(i["x_right"] for i in first["items"])) / 2

            if center > center_x:
                speaker = "나"
                messages = [" ".join(i["text"] for i in line["items"]) for line in pending]
            else:
                speaker = " ".join(i["text"] for i in first["items"])
                messages = [" ".join(i["text"] for i in line["items"]) for line in pending[1:]]

            msg = " ".join(messages)
            final_chat.append(f"{full_ts}, {speaker} : {msg}")
            pending = []

    return "\n".join(final_chat)


def _process_and_save_groups(grouped, items):

    items.sort(key=lambda x: x["y_center"])

    sub = [items[0]]
    for it in items[1:]:
        if it["y_center"] - sub[-1]["y_center"] > 12:
            _finalize(grouped, sub)
            sub = [it]
        else:
            sub.append(it)

    _finalize(grouped, sub)


def _finalize(grouped, items):

    items.sort(key=lambda x: x["x_left"])

    texts = [i for i in items if not i["is_timestamp"] and not i["is_date"]]
    times = [i for i in items if i["is_timestamp"]]
    dates = [i for i in items if i["is_date"]]

    if texts: grouped.append({"type": "text", "items": texts})
    if times: grouped.append({"type": "timestamp", "items": times})
    if dates: grouped.append({"type": "date", "items": dates})



# =========================================================
# 4. 실행
# =========================================================

if __name__ == "__main__":

    image_file = "Test_Dataset/Test_image.jpg"

    print("OCR + 파싱 시작:", image_file)
    ocr_pack = parse_chat_image(image_file)

    parsed_log = parse_kakao_dict(ocr_pack, ocr_pack["width"])

    print("\n====================================")
    print(" 최종 변환 결과")
    print("====================================")
    print(parsed_log)
