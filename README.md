# GDGoC_Yonsei_oTP — ToText Sub-Project  
이미지 및 스크롤 녹화 영상 → 텍스트 변환 자동화 도구

---

## Introduction

이 저장소는 **GDGoC_Yonsei OTP 프로젝트**를 위해 제작된 **서브 프로젝트(Sub Project)**입니다.  
카카오톡 기반 채팅 화면을 이미지 또는 스크롤 녹화 영상 형태로 입력하면, 이를 **자동으로 텍스트 로그로 변환하는 기능**을 제공합니다.

이 프로젝트는 메인 프로젝트의 부가기능으로,  
- Chat Screenshot → Text 변환  
- Scrolling Recording Video → 자동 스크린샷 분리 → OCR → 텍스트 파싱  
을 담당합니다.

> **즉, 이 저장소는 독립적인 앱이 아니라  
>  “GDGoC_Yonsei OTP”에 포함되는 데이터 전처리/텍스트 변환 도구입니다.**

---

## 기능 Overview

### 1) 이미지 → 텍스트 변환 (`img_to_text.py`)
- EasyOCR(한글 모델) 기반  
- 로컬 모델만 사용 (다운로드 없음)  
- 카카오톡 채팅 포맷에 맞춰 텍스트 파싱  
- 날짜/시간/화자(나/상대) 자동 구분  
- OCR 노이즈 제거 & 문장 재구성

---

### 2) 스크롤 화면 녹화 영상 → 텍스트 (`video_to_text.py`)
- SPyNet 기반 Optical Flow를 이용해 **스크롤 발생 구간 자동 감지**
- 자동 스크린샷 저장
- 저장된 모든 이미지 → OCR → 파싱 → **챗로그 전체 병합**
- 최종 결과: `runs/<timestamp>/final_chat.txt`

