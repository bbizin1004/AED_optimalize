# geocode_aed.py
# -*- coding: utf-8 -*-
import os, time, re, math, json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT  = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

# 입력: 너의 통합 AED 엑셀(천안 시트)
AED_ALL_XLSX = DATA / "자동심장충격기(AED)_통합.xlsx"
SHEET_HINT = "천안"   # 시트명에 '천안' 포함

# 카카오 REST 키 (.env에 KAKAO_REST_API_KEY=xxxxx)
load_dotenv()
KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
if not KAKAO_KEY:
    raise RuntimeError("환경변수 KAKAO_REST_API_KEY 가 없습니다. .env에 넣어주세요.")

def pick_col(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, str(c)):
                return c
    return None

def build_address(row, cols_priority):
    parts = []
    for c in cols_priority:
        if c and pd.notna(row.get(c)) and str(row.get(c)).strip():
            parts.append(str(row[c]))
    return " ".join(parts) if parts else None

def kakao_geocode(query:str):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    r = requests.get(url, params={"query": query}, headers=headers, timeout=7)
    r.raise_for_status()
    data = r.json().get("documents", [])
    if not data:
        return None
    # 도로명 > 지번 우선
    doc = next((d for d in data if d.get("address_type")=="ROAD_ADDR"), data[0])
    x = float(doc["x"]); y = float(doc["y"])
    return y, x  # (lat, lon)

def main():
    xls = pd.ExcelFile(AED_ALL_XLSX)
    sheet = next((n for n in xls.sheet_names if SHEET_HINT in str(n)), None)
    if not sheet:
        raise RuntimeError(f"'{SHEET_HINT}' 포함 시트를 찾지 못함: {xls.sheet_names}")

    raw = pd.read_excel(AED_ALL_XLSX, sheet_name=sheet, dtype=str)
    cols = list(raw.columns)

    sido_col   = pick_col(cols, [r"^시도$", r"시\s*도"])
    sgg_col    = pick_col(cols, [r"^시군구$", r"시\s*군\s*구", r"시군구명"])
    emd_col    = pick_col(cols, [r"^읍면동$", r"읍\s*면\s*동", r"행정동", r"법정동"])
    road_col   = pick_col(cols, [r"도로명주소"])
    jibun_col  = pick_col(cols, [r"지번주소"])
    addr_col   = pick_col(cols, [r"^주소$", r"상세주소"])
    facility_col = pick_col(cols, [r"설치소|시설명|시설|장소|기관|설치장소"])

    df = pd.DataFrame()
    if facility_col: df["facility_name"] = raw[facility_col]
    for c in ["시도","sgg","emd","road","jibun","addr_any"]:
        df[c] = None

    df["시도"] = raw.get(sido_col)
    df["sgg"]  = raw.get(sgg_col)
    df["emd"]  = raw.get(emd_col)
    df["road"] = raw.get(road_col)
    df["jibun"]= raw.get(jibun_col)
    df["addr_any"] = raw.get(addr_col)

    # 지오코딩용 주소 후보: 도로명 > 지번 > 조합
    results = []
    for i, r in df.iterrows():
        cand = None
        for addr in [r["road"], r["jibun"]]:
            if isinstance(addr, str) and addr.strip():
                cand = addr.strip(); break
        if not cand:
            cand = build_address(r, ["시도","sgg","emd","addr_any"])
        if not cand or not isinstance(cand, str) or not cand.strip():
            results.append((np.nan, np.nan, None)); continue

        try:
            geo = kakao_geocode(cand)
            if geo:
                results.append((geo[0], geo[1], cand))
            else:
                results.append((np.nan, np.nan, cand))
        except requests.HTTPError as e:
            results.append((np.nan, np.nan, cand))
        except Exception:
            results.append((np.nan, np.nan, cand))
        time.sleep(0.12)  # rate-limit

    df["lat"], df["lon"], df["addr_used"] = zip(*results)

    out_path = OUT / "aed_geocoded.csv"
    keep_cols = ["facility_name","시도","sgg","emd","road","jibun","addr_used","lat","lon"]
    df[keep_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 저장: {out_path} (지오코딩 성공률={df['lat'].notna().mean():.0%})")

if __name__ == "__main__":
    main()
