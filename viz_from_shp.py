# viz_from_shp.py
# SGIS Shapefile(SHP: ADM_NM/ADM_CD만 있는 경우 포함) -> AED 결과 결합 -> folium 지도 + 그래프 생성

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)

# main.py 산출물
FINAL_CSV  = OUT / "cheonan_eupmyeondong_aed_pop.csv"
RANK_OVER  = OUT / "aed_over_alloc_top10_nb.csv"      # NB가 기본. 없으면 poisson 파일을 참고하도록 개선 가능
RANK_UNDER = OUT / "aed_under_alloc_top10_nb.csv"

# -------------------------------------------------
# 1) Shapefile 자동 탐색 (.shp만 올려놔도 세트 확인)
# -------------------------------------------------
SHP_FILES = list(DATA.glob("bnd_dong_*_*.shp"))
if not SHP_FILES:
    raise SystemExit("[에러] data 폴더에서 Shapefile(.shp)을 찾지 못했습니다. 예) data/bnd_dong_34011_2024_2Q.shp")

print("[정보] 사용 Shapefile들:")
for p in SHP_FILES:
    print(" -", p.name)

# 필수 세트 검사(.shp/.shx/.dbf)
required_ext = {".shp", ".shx", ".dbf"}
for shp in SHP_FILES:
    base = shp.with_suffix("")
    missing = [ext for ext in required_ext if not (base.with_suffix(ext)).exists()]
    if missing:
        raise SystemExit(f"[에러] {shp.name} 세트에 {missing} 파일이 없습니다. (.shp/.shx/.dbf 필수)")

# -------------------------------------------------
# 2) 유틸
# -------------------------------------------------
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s.replace("·", "")

def first_key(cols, candidates):
    for k in candidates:
        if k in cols:
            return k
    return None

def map_gu_name_from_sigcd(sigcd):
    # 천안시 동남구/서북구
    m = {"44130": "천안시 동남구", "44131": "천안시 서북구"}
    return m.get(str(sigcd), None)

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# -------------------------------------------------
# 3) AED 결과 로드
# -------------------------------------------------
df = pd.read_csv(FINAL_CSV, dtype={"adm_cd": str})
df["old_ratio"] = np.where(df["pop_total"] > 0, df["pop_65p"] / df["pop_total"], np.nan)
df["_emd_key"]  = df["emd_nm"].map(norm)
df["_gu_key"]   = df["gu"].map(norm)

rank_over  = safe_read_csv(RANK_OVER)
rank_under = safe_read_csv(RANK_UNDER)

# -------------------------------------------------
# 4) SHP 읽기 & 좌표계
# -------------------------------------------------
gdfs = []
for shp in SHP_FILES:
    try:
        g = gpd.read_file(shp)
    except UnicodeDecodeError:
        g = gpd.read_file(shp, encoding="euc-kr")
    gdfs.append(g)

gdf = pd.concat(gdfs, ignore_index=True)

# folium 용 WGS84로 변환
try:
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
except Exception:
    try:
        gdf = gdf.set_crs(epsg=5179).to_crs(epsg=4326)  # 한국 TM 추정
    except Exception:
        print("[경고] 좌표계 변환 실패. 지도 위치가 어긋날 수 있습니다.")

cols = list(gdf.columns)
print("[정보] SHP 컬럼들:", cols)

# -------------------------------------------------
# 5) 컬럼 자동 인식 (당신 SHP: ['BASE_DATE','ADM_NM','ADM_CD','geometry'])
# -------------------------------------------------
emd_name_col = first_key(cols, ["EMD_KOR_NM", "EMD_NM", "EMD_NAME", "ADM_NM", "adm_nm",
                                "법정동", "법정동명", "행정동명"])
emd_code_col = first_key(cols, ["EMD_CD", "EMD_CD10", "ADM_CD", "adm_cd", "법정동코드", "HCODE"])
sig_name_col = first_key(cols, ["SIG_KOR_NM", "SIGUNGU_NM", "SIG_KOR", "SIG_NM", "SIG_NAME", "시군구명"])
sig_code_col = first_key(cols, ["SIG_CD", "sig_cd", "SIGCODE", "시군구코드"])

if not emd_name_col:
    raise SystemExit(f"[에러] 동명 컬럼(예: ADM_NM/EMD_NM)을 찾지 못했습니다. 컬럼들: {cols}")

# SIG 이름이 없으면 코드로부터 구이름 복원 (ADM_CD 앞 5자리 = SIG_CD 가정)
if not sig_name_col:
    if sig_code_col is None and emd_code_col:
        gdf["_SIG_CD_FROM_ADM"] = gdf[emd_code_col].astype(str).str[:5]
        sig_code_col = "_SIG_CD_FROM_ADM"
    if sig_code_col:
        gdf["_SIG_NAME_FROM_CD"] = gdf[sig_code_col].map(map_gu_name_from_sigcd)
        sig_name_col = "_SIG_NAME_FROM_CD"

# 조인 키 생성
gdf["_emd_nm"]  = gdf[emd_name_col].astype(str)
gdf["_emd_key"] = gdf["_emd_nm"].map(norm)

if sig_name_col:
    gdf["_gu_nm"]  = gdf[sig_name_col].astype(str)
    gdf["_gu_key"] = gdf["_gu_nm"].map(norm)
else:
    gdf["_gu_nm"]  = None
    gdf["_gu_key"] = None

# -------------------------------------------------
# 6) 병합: 코드 우선, 실패 시 이름+구
# -------------------------------------------------
merged = None
if emd_code_col and "adm_cd" in df.columns:
    gdf["_emd_cd"] = gdf[emd_code_col].astype(str).str[:10]
    merged = gdf.merge(df, left_on="_emd_cd", right_on="adm_cd", how="left")

if merged is None or merged["aed_count"].isna().all():
    merged = gdf.merge(df, left_on=["_emd_key", "_gu_key"], right_on=["_emd_key", "_gu_key"], how="left")

ok = merged["aed_count"].notna().sum()
print(f"[정보] SHP-결과 병합 매칭 성공: {ok} / {len(merged)}")

# -------------------------------------------------
# 7) Folium 지도 생성
# -------------------------------------------------
def make_map(value_col: str, title: str, outname: str):
    if value_col not in merged.columns:
        raise SystemExit(f"[에러] {value_col} 컬럼이 없습니다. 먼저 main.py를 실행해 결과 CSV를 생성하세요.")

    m = folium.Map(location=[36.815, 127.146], zoom_start=11, tiles="cartodbpositron")

    series = pd.to_numeric(merged[value_col], errors="coerce")
    qs = series.dropna().quantile([0.2, 0.4, 0.6, 0.8]).values if series.notna().any() else None

    gjson = json.loads(merged.to_json())

    def style_fn(feat):
        p = feat["properties"]
        val = p.get(value_col, None)
        fill_opacity = 0.7 if val is not None else 0.0
        color = "#f7fbff"
        if val is not None and qs is not None:
            if val <= qs[0]: color = "#deebf7"
            elif val <= qs[1]: color = "#c6dbef"
            elif val <= qs[2]: color = "#9ecae1"
            elif val <= qs[3]: color = "#6baed6"
            else: color = "#3182bd"
        return {"fillColor": color, "color": "#555555", "weight": 0.6, "fillOpacity": fill_opacity}

    layer = folium.GeoJson(
        gjson,
        name=title,
        style_function=style_fn,
        tooltip=GeoJsonTooltip(
            fields=[c for c in ["_gu_nm", "_emd_nm", "aed_count", "pop_total", "pop_65p", value_col] if c in merged.columns],
            aliases=["구", "읍면동", "AED 수", "총인구", "65세+", "지표"],
            sticky=True
        ),
    )
    layer.add_to(m)
    folium.LayerControl().add_to(m)
    out_html = OUT / outname
    m.save(str(out_html))
    print(f"[OK] 지도 저장: {out_html}")

make_map("aed_per_10k",    "AED per 10k population",   "map_from_shp_aed_per_10k.html")
make_map("aed_per_1k_65p", "AED per 1k seniors (65+)", "map_from_shp_aed_per_1k_65p.html")

# -------------------------------------------------
# 8) 그래프: 랭킹 TOP10 막대 + 산점도 2종
# -------------------------------------------------
def bar_top10(csv_path, title, outname, value_col="std_resid"):
    if csv_path is None or not csv_path.exists():
        print(f("[SKIP] {title}: {csv_path} 없음"))
        return

    tdf = pd.read_csv(csv_path)

    def pick_col(df, bases):
        # bases 안의 후보들에 대해 '', '_x', '_y' 순으로 존재하는 첫 컬럼 반환
        suffixes = ["", "_x", "_y"]
        for b in bases:
            for suf in suffixes:
                col = b + suf
                if col in df.columns:
                    return col
        return None

    # 읍면동/구 컬럼 자동 탐색 (접미사 대응)
    label_col = pick_col(tdf, ["emd_nm", "ADM_NM", "읍면동", "읍면동명", "행정동명", "법정동명"])
    gu_col    = pick_col(tdf, ["gu", "구", "sigungu", "SIG_KOR_NM"])

    if label_col is None:
        raise SystemExit(f"[에러] 읍면동 이름 컬럼을 찾지 못했습니다. CSV 컬럼: {tdf.columns.tolist()}")
    if value_col not in tdf.columns:
        raise SystemExit(f"[에러] '{value_col}' 컬럼이 없습니다. CSV 컬럼: {tdf.columns.tolist()}")

    # TOP10
    if len(tdf) > 10:
        tdf = tdf.head(10)

    # 라벨: "구 읍면동" (구가 없으면 읍면동만)
    if gu_col and gu_col in tdf.columns:
        tdf["_label"] = (tdf[gu_col].astype(str).fillna("") + " " + tdf[label_col].astype(str).fillna("")).str.strip()
    else:
        tdf["_label"] = tdf[label_col].astype(str)

    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    y = np.arange(len(tdf))
    plt.barh(y, tdf[value_col].values)   # 색 지정 X (규칙)
    plt.yticks(y, tdf["_label"].values)
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    out_png = OUT / outname
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 그래프 저장: {out_png}")


bar_top10(RANK_OVER,  "과다 배치 TOP10 (표준화 잔차)", "bar_from_shp_over_top10.png")
bar_top10(RANK_UNDER, "과소 배치 TOP10 (표준화 잔차)", "bar_from_shp_under_top10.png")

# 산점도 1: 총인구 vs AED 수 (점크기 = 고령비)
plt.figure(figsize=(6.5, 5))
sizes = (df["old_ratio"].fillna(0.0) * 800) + 20
plt.scatter(df["pop_total"], df["aed_count"], s=sizes)  # 색 지정 X
plt.xlabel("총인구")
plt.ylabel("AED 수")
plt.title("총인구 vs AED 수 (점 크기 = 고령비)")
plt.tight_layout()
plt.savefig(OUT / "scatter_from_shp_pop_vs_aed.png", dpi=150)
plt.close()
print(f"[OK] 그래프 저장: {OUT / 'scatter_from_shp_pop_vs_aed.png'}")

# 산점도 2: 고령비 vs 1만명당 AED
plt.figure(figsize=(6.5, 5))
plt.scatter(df["old_ratio"], df["aed_per_10k"])        # 색 지정 X
plt.xlabel("고령비 (65+/총인구)")
plt.ylabel("1만명당 AED")
plt.title("고령비 vs AED 밀도(1만명당)")
plt.tight_layout()
plt.savefig(OUT / "scatter_from_shp_oldratio_vs_aedper10k.png", dpi=150)
plt.close()
print(f"[OK] 그래프 저장: {OUT / 'scatter_from_shp_oldratio_vs_aedper10k.png'}")
