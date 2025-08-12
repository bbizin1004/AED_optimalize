# viz_cheonan.py
# 천안시 AED 분석 결과 시각화 (지도 + 그래프)
# 생성물: output/ 아래 HTML 지도 2개, PNG 그래프 4개

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip

# -------------------------
# 경로
# -------------------------
BASE   = Path(__file__).resolve().parent
OUT    = BASE / "output"
OUT.mkdir(exist_ok=True)

FINAL_CSV   = OUT / "cheonan_eupmyeondong_aed_pop.csv"
RANK_OVER   = OUT / "aed_over_alloc_top10_nb.csv"   # main.py에서 이미 생성됨(NB가 기본)
RANK_UNDER  = OUT / "aed_under_alloc_top10_nb.csv"  # 없으면 포아송 버전으로 교체해도 됨

# ✅ 여기를 여러분 PC의 지오JSON 경로로 바꿔주세요!
# 천안시 읍·면·동 경계 GeoJSON (행정동 단위). 파일 안의 속성키는 자동 추정합니다.
GEOJSON_PATH = BASE / "data" / "cheonan_emd.geojson"   # 예시 경로

# -------------------------
# 데이터 로드
# -------------------------
df = pd.read_csv(FINAL_CSV, dtype={"adm_cd":str})
df["old_ratio"] = np.where(df["pop_total"]>0, df["pop_65p"]/df["pop_total"], np.nan)

# 랭킹(없으면 건너뛰도록)
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

rank_over  = safe_read_csv(RANK_OVER)
rank_under = safe_read_csv(RANK_UNDER)

# -------------------------
# GeoJSON 로드 & 이름 키 추출
# -------------------------
def norm(s):
    if s is None or (isinstance(s,float) and np.isnan(s)): return None
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s.replace("·","")

def first_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None

def load_geojson(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    if not feats:
        raise RuntimeError("GeoJSON features가 없습니다.")
    # 속성 키 후보들(지자체마다 다름)
    # 동명
    emd_key = first_key(feats[0]["properties"], [
        "EMD_KOR_NM","EMD_NM","EMD_NAME","adm_nm","EMDENG_NM","법정동명","행정동명"
    ])
    # 구/시군구
    gu_key  = first_key(feats[0]["properties"], [
        "SIG_KOR_NM","SIG_KOR","SIGUNGU_NM","SIG_NM","SIG_KOR_NM2","시군구명","SIG_NAME"
    ])
    # 코드(있으면 보조)
    code_key = first_key(feats[0]["properties"], [
        "EMD_CD","EMD_CD10","adm_cd","EMD_CODE","법정동코드","행정동코드"
    ])
    if not emd_key:
        raise RuntimeError("동명(EMD) 속성 키를 찾지 못했습니다. GeoJSON 속성 키를 확인하세요.")
    return gj, emd_key, gu_key, code_key

gj, gj_emd_key, gj_gu_key, gj_code_key = load_geojson(GEOJSON_PATH)

# GeoJSON feature별 join용 key 만들기
for f in gj["features"]:
    props = f["properties"]
    props["_emd_nm_raw"] = props.get(gj_emd_key)
    props["_emd_key"]    = norm(props.get(gj_emd_key))
    props["_gu_key"]     = norm(props.get(gj_gu_key)) if gj_gu_key else None

# 분석 DF도 키 생성
df["_emd_key"] = df["emd_nm"].map(norm)
df["_gu_key"]  = df["gu"].map(norm)

# -------------------------
# 지도 만들기(Choropleth)
# -------------------------
def make_choropleth(value_col: str, title: str, outname: str):
    # 지도 중심(천안시 대략 중심)
    m = folium.Map(location=[36.815, 127.146], zoom_start=11, tiles="cartodbpositron")

    # 데이터 매핑 dict: (gu_key, emd_key) -> 값
    val_map = {}
    for _, r in df.iterrows():
        val_map[(r["_gu_key"], r["_emd_key"])] = r.get(value_col)

    # 툴팁용: AED 수, 인구, 값
    tooltip_fields = ["emd_nm","gu","aed_count","pop_total","pop_65p", value_col]
    tooltip_alias  = ["읍면동","구","AED 수","총인구","65세+","지표"]

    def style_function(feat):
        p = feat["properties"]
        key = (p.get("_gu_key"), p.get("_emd_key"))
        val = val_map.get(key, None)
        # 값에 따라 채우기(단, folium.LinearColormap를 안 쓰고 간단 구간화)
        fill_opacity = 0.7 if val is not None else 0.0
        # 색상은 folium 기본 colormap 대신 고정 팔레트(점진톤) 간단 사용
        # (요구사항은 matplotlib에만 색 제약, folium은 지도 색상 OK)
        color = "#f7fbff"
        if val is not None:
            # 구간화(사분위)
            q = np.nanquantile(df[value_col], [0.2,0.4,0.6,0.8])
            if val <= q[0]: color = "#deebf7"
            elif val <= q[1]: color = "#c6dbef"
            elif val <= q[2]: color = "#9ecae1"
            elif val <= q[3]: color = "#6baed6"
            else: color = "#3182bd"
        return {"fillColor": color, "color": "#555555", "weight": 0.6, "fillOpacity": fill_opacity}

    gj_layer = folium.GeoJson(
        gj,
        name=title,
        style_function=style_function,
        tooltip=GeoJsonTooltip(
            fields=[gj_emd_key] + ([gj_gu_key] if gj_gu_key else []),
            aliases=["읍면동","구"] if gj_gu_key else ["읍면동"],
            sticky=True
        )
    )

    # 팝업: 합쳐서 보여주기
    for f in gj_layer.data["features"]:
        p = f["properties"]
        key = (p.get("_gu_key"), p.get("_emd_key"))
        val = val_map.get(key, None)
        # df에서 레코드 찾기
        row = df[(df["_gu_key"]==key[0]) & (df["_emd_key"]==key[1])]
        if len(row):
            r = row.iloc[0]
            html = f"""
            <b>{r['gu']} {r['emd_nm']}</b><br>
            AED: {int(r['aed_count'])}<br>
            총인구: {int(r['pop_total']):,}<br>
            65세+: {int(r['pop_65p']) :,}<br>
            {value_col}: {'' if pd.isna(r[value_col]) else round(float(r[value_col]), 3)}
            """
        else:
            html = f"<b>{p.get(gj_emd_key)}</b><br>데이터 없음"
        folium.Popup(html, max_width=300).add_to(f)

    gj_layer.add_to(m)
    folium.LayerControl().add_to(m)
    out_html = OUT / outname
    m.save(str(out_html))
    print(f"[OK] 지도 저장: {out_html}")

# 1) 1만명당 AED
make_choropleth("aed_per_10k", "AED per 10k population", "map_aed_per_10k.html")
# 2) 65세 1천명당 AED
make_choropleth("aed_per_1k_65p", "AED per 1k seniors (65+)", "map_aed_per_1k_65p.html")

# -------------------------
# 그래프 1: 과다/과소 TOP10 막대
# (요구사항: matplotlib 사용, 한 그림당 하나의 차트, 색 지정하지 않음)
# -------------------------
def bar_top10(csv_path, title, outname, value_col="std_resid", label_col="emd_nm"):
    if csv_path is None or not csv_path.exists():
        print(f"[SKIP] {title}: {csv_path} 없음")
        return
    tdf = pd.read_csv(csv_path)
    # 상위 10개만 (이미 파일이 10개일 것)
    if len(tdf) > 10:
        tdf = tdf.head(10)
    # 라벨 만들기: 구+동
    if "gu" in tdf.columns and label_col in tdf.columns:
        tdf["_label"] = tdf["gu"].fillna("") + " " + tdf[label_col].fillna("")
    else:
        tdf["_label"] = tdf[label_col].astype(str)

    plt.figure(figsize=(8, 4.5))
    y = np.arange(len(tdf))
    plt.barh(y, tdf[value_col].values)
    plt.yticks(y, tdf["_label"].values)
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    out_png = OUT / outname
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] 그래프 저장: {out_png}")

bar_top10(RANK_OVER,  "과다 배치 TOP10 (표준화 잔차)", "bar_over_top10.png")
bar_top10(RANK_UNDER, "과소 배치 TOP10 (표준화 잔차)", "bar_under_top10.png")

# -------------------------
# 그래프 2: 산점도 (총인구 vs AED 수)
# -------------------------
plt.figure(figsize=(6.5, 5))
sizes = (df["old_ratio"].fillna(0.0) * 800) + 20  # 고령비가 클수록 점 크게
plt.scatter(df["pop_total"], df["aed_count"], s=sizes)
plt.xlabel("총인구")
plt.ylabel("AED 수")
plt.title("총인구 vs AED 수 (점 크기 = 고령비)")
plt.tight_layout()
out_png = OUT / "scatter_pop_vs_aed.png"
plt.savefig(out_png, dpi=150)
plt.close()
print(f"[OK] 그래프 저장: {out_png}")

# -------------------------
# 그래프 3: 산점도 (고령비 vs 1만명당 AED)
# -------------------------
plt.figure(figsize=(6.5, 5))
plt.scatter(df["old_ratio"], df["aed_per_10k"])
plt.xlabel("고령비 (65+/총인구)")
plt.ylabel("1만명당 AED")
plt.title("고령비 vs AED 밀도(1만명당)")
plt.tight_layout()
out_png = OUT / "scatter_oldratio_vs_aedper10k.png"
plt.savefig(out_png, dpi=150)
plt.close()
print(f"[OK] 그래프 저장: {out_png}")
