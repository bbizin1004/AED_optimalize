# app.py
# 천안시 AED 배치 분석 Streamlit 대시보드
# - main.py 산출물(csv) + SGIS Shapefile(SHP) 결합
# - 지도(folium), 랭킹, 산점도 시각화

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt


# 한글 폰트 설정 (NanumGothic)
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False) 


# ------------------------------
# 경로 세팅
# ------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT  = BASE / "output"

FINAL_CSV = OUT / "cheonan_eupmyeondong_aed_pop.csv"
RANK_NB_OVER  = next((OUT.glob("aed_over_alloc_top10_nb.csv")), None)
RANK_NB_UNDER = next((OUT.glob("aed_under_alloc_top10_nb.csv")), None)
RANK_P_OVER   = next((OUT.glob("aed_over_alloc_top10_poisson.csv")), None)
RANK_P_UNDER  = next((OUT.glob("aed_under_alloc_top10_poisson.csv")), None)

# 기본 Shapefile 패턴(동남+서북)
SHP_FILES = list(DATA.glob("bnd_dong_*_*.shp"))


# 유틸
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return None
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s.replace("·", "")

def map_gu_name_from_sigcd(sigcd):
    m = {"44130":"천안시 동남구", "44131":"천안시 서북구"}
    return m.get(str(sigcd), None)

def pick_col(cols, bases):
    suffixes = ["", "_x", "_y"]
    for b in bases:
        for suf in suffixes:
            c = b + suf
            if c in cols:
                return c
    return None

@st.cache_data(show_spinner=False)
def load_final_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"adm_cd":str})
    df["old_ratio"] = np.where(df["pop_total"]>0, df["pop_65p"]/df["pop_total"], np.nan)
    df["_emd_key"] = df["emd_nm"].map(norm)
    df["_gu_key"]  = df["gu"].map(norm)
    return df

@st.cache_resource(show_spinner=False)
def load_shp(files: list[Path]) -> gpd.GeoDataFrame:
    if not files:
        return gpd.GeoDataFrame()
    gdfs = []
    for shp in files:
        try:
            g = gpd.read_file(shp)
        except UnicodeDecodeError:
            g = gpd.read_file(shp, encoding="euc-kr")
        gdfs.append(g)
    gdf = pd.concat(gdfs, ignore_index=True)
    # 좌표계 변환 (folium: WGS84)
    try:
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        try:
            gdf = gdf.set_crs(epsg=5179).to_crs(epsg=4326)
        except Exception:
            pass
    return gdf

def auto_identify_cols(gdf: gpd.GeoDataFrame):
    cols = list(gdf.columns)
    emd_name_col = None
    for c in ["EMD_KOR_NM","EMD_NM","EMD_NAME","ADM_NM","adm_nm","법정동","법정동명","행정동명"]:
        if c in cols:
            emd_name_col = c; break
    emd_code_col = None
    for c in ["EMD_CD","EMD_CD10","ADM_CD","adm_cd","법정동코드","HCODE"]:
        if c in cols:
            emd_code_col = c; break
    sig_name_col = None
    for c in ["SIG_KOR_NM","SIGUNGU_NM","SIG_KOR","SIG_NM","SIG_NAME","시군구명"]:
        if c in cols:
            sig_name_col = c; break
    sig_code_col = None
    for c in ["SIG_CD","sig_cd","SIGCODE","시군구코드"]:
        if c in cols:
            sig_code_col = c; break
    return emd_name_col, emd_code_col, sig_name_col, sig_code_col

def merge_geo(gdf: gpd.GeoDataFrame, df: pd.DataFrame) -> gpd.GeoDataFrame:
    emd_name_col, emd_code_col, sig_name_col, sig_code_col = auto_identify_cols(gdf)
    if not emd_name_col:
        raise ValueError(f"SHP 동명 컬럼을 찾지 못함. 컬럼: {list(gdf.columns)}")

    # 구명 없으면 코드에서 복원(ADM_CD 앞 5자리)
    if not sig_name_col:
        if not sig_code_col and emd_code_col:
            gdf["_SIG_CD_FROM_ADM"] = gdf[emd_code_col].astype(str).str[:5]
            sig_code_col = "_SIG_CD_FROM_ADM"
        if sig_code_col:
            gdf["_SIG_NAME_FROM_CD"] = gdf[sig_code_col].map(map_gu_name_from_sigcd)
            sig_name_col = "_SIG_NAME_FROM_CD"

    gdf = gdf.copy()
    gdf["_emd_nm"]  = gdf[emd_name_col].astype(str)
    gdf["_emd_key"] = gdf["_emd_nm"].map(norm)
    if sig_name_col:
        gdf["_gu_nm"]  = gdf[sig_name_col].astype(str)
        gdf["_gu_key"] = gdf["_gu_nm"].map(norm)
    else:
        gdf["_gu_nm"] = None; gdf["_gu_key"] = None

    merged = None
    if emd_code_col and "adm_cd" in df.columns:
        gdf["_emd_cd"] = gdf[emd_code_col].astype(str).str[:10]
        merged = gdf.merge(df, left_on="_emd_cd", right_on="adm_cd", how="left")
    if merged is None or merged["aed_count"].isna().all():
        merged = gdf.merge(df, left_on=["_emd_key","_gu_key"], right_on=["_emd_key","_gu_key"], how="left")
    return merged

def load_rank_csvs() -> tuple[pd.DataFrame|None, pd.DataFrame|None]:
    # NB 우선, 없으면 Poisson
    over = pd.read_csv(RANK_NB_OVER) if RANK_NB_OVER and RANK_NB_OVER.exists() else \
           (pd.read_csv(RANK_P_OVER) if RANK_P_OVER and RANK_P_OVER.exists() else None)
    under = pd.read_csv(RANK_NB_UNDER) if RANK_NB_UNDER and RANK_NB_UNDER.exists() else \
            (pd.read_csv(RANK_P_UNDER) if RANK_P_UNDER and RANK_P_UNDER.exists() else None)
    return over, under


# UI
st.set_page_config(page_title="천안시 AED 분석", layout="wide")
st.title("천안시 AED 배치 분석 대시보드")

with st.sidebar:
    st.header("데이터 선택")
    final_csv_path = st.text_input("결과 CSV 경로", str(FINAL_CSV))
    shp_info = st.write("SHP 자동 탐색:", ", ".join(p.name for p in SHP_FILES) or "없음")
    metric = st.selectbox("지도 지표", ["aed_per_10k", "aed_per_1k_65p"], format_func=lambda x: "1만명당 AED" if x=="aed_per_10k" else "65세 1천명당 AED")
    quantiles = st.slider("색상 분위 경계(20~80%)", 20, 80, (20,80))
    show_labels = st.checkbox("툴팁에 인구/AED 표시", True)

# 데이터 로드
try:
    df_final = load_final_csv(Path(final_csv_path))
except Exception as e:
    st.error(f"결과 CSV 로드 실패: {e}")
    st.stop()

if not SHP_FILES:
    st.warning("data/ 폴더에 Shapefile(.shp/.shx/.dbf...)을 넣어주세요.")
    st.stop()

gdf_raw = load_shp(SHP_FILES)
if gdf_raw.empty:
    st.error("Shapefile 로드 실패")
    st.stop()

try:
    gdf = merge_geo(gdf_raw, df_final)
except Exception as e:
    st.error(f"Shapefile-결과 병합 실패: {e}")
    st.stop()

# KPI
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("행정동 수", int(gdf.shape[0]))
with c2:
    st.metric("총 AED 수(집계)", int(np.nansum(df_final["aed_count"])))
with c3:
    st.metric("총 인구", f"{int(np.nansum(df_final['pop_total'])):,}")
with c4:
    st.metric("65세 이상 인구", f"{int(np.nansum(df_final['pop_65p'])):,}")


# 탭
tab1, tab2, tab3 = st.tabs(["🗺 지도", "📊 랭킹", "🔬 산점도"])

#TAB1: 지도 
with tab1:
    st.subheader("행정동별 AED 밀도 지도")
    # 분위 경계
    col_vals = pd.to_numeric(gdf[metric], errors="coerce")
    if col_vals.notna().any():
        q_low  = col_vals.quantile(quantiles[0]/100)
        q_mid1 = col_vals.quantile(0.4)
        q_mid2 = col_vals.quantile(0.6)
        q_high = col_vals.quantile(quantiles[1]/100)
        qs = [q_low, q_mid1, q_mid2, q_high]
    else:
        qs = None

    m = folium.Map(location=[36.815, 127.146], zoom_start=11, tiles="cartodbpositron")
    gjson = json.loads(gdf.to_json())

    def style_fn(feat):
        p = feat["properties"]
        val = p.get(metric, None)
        fill_opacity = 0.7 if val is not None else 0.0
        color = "#f7fbff"
        if val is not None and qs is not None:
            if val <= qs[0]: color = "#deebf7"
            elif val <= qs[1]: color = "#c6dbef"
            elif val <= qs[2]: color = "#9ecae1"
            elif val <= qs[3]: color = "#6baed6"
            else: color = "#3182bd"
        return {"fillColor": color, "color": "#555555", "weight": 0.6, "fillOpacity": fill_opacity}

    tooltip_fields = ["_gu_nm", "_emd_nm"]
    tooltip_alias  = ["구", "읍면동"]
    if show_labels:
        for c in ["aed_count","pop_total","pop_65p", metric]:
            if c in gdf.columns:
                tooltip_fields.append(c)
                tooltip_alias.append({"aed_count":"AED 수","pop_total":"총인구","pop_65p":"65세+","aed_per_10k":"1만명당 AED","aed_per_1k_65p":"65세 1천명당 AED"}\
                                      .get(c, c))

    folium.GeoJson(
        gjson,
        name="AED Choropleth",
        style_function=style_fn,
        tooltip=folium.features.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_alias, sticky=True)
    ).add_to(m)

    st_folium(m, height=620, use_container_width=True)

#TAB2: 랭킹
with tab2:
    st.subheader("예상 대비 과다/과소 배치 TOP10 (표준화 잔차)")
    over_df, under_df = load_rank_csvs()
    if (over_df is None) or (under_df is None):
        st.info("랭킹 CSV 파일이 없습니다. main.py를 먼저 실행해 주세요.")
    else:
        # 컬럼 자동 선택
        label_col = pick_col(over_df.columns, ["emd_nm","ADM_NM","읍면동","읍면동명","행정동명","법정동명"]) or "emd_nm"
        gu_col    = pick_col(over_df.columns, ["gu","구","sigungu","SIG_KOR_NM"])
        val_col   = pick_col(over_df.columns, ["std_resid"]) or "std_resid"

        col1, col2 = st.columns(2)
        for title, df_rank, container in [
            ("과다 배치 TOP 10", over_df, col1),
            ("과소 배치 TOP 10", under_df, col2)
        ]:
            with container:
                tdf = df_rank.copy()
                if len(tdf) > 10: tdf = tdf.head(10)
                # 라벨
                if gu_col and gu_col in tdf.columns:
                    tdf["_label"] = (tdf[gu_col].astype(str).fillna("") + " " + tdf[label_col].astype(str).fillna("")).str.strip()
                else:
                    tdf["_label"] = tdf[label_col].astype(str)

                st.dataframe(tdf[[label_col, gu_col, val_col]].rename(columns={
                    label_col:"읍면동", gu_col:"구", val_col:"표준화잔차"
                }), use_container_width=True, hide_index=True)

                # 막대 그래프 (matplotlib, 색 미지정, 한 그림 한 차트)
                plt.figure(figsize=(6, 4))
                y = np.arange(len(tdf))
                plt.barh(y, tdf[val_col].values)
                plt.yticks(y, tdf["_label"].values)
                plt.title(title)
                plt.xlabel("표준화 잔차")
                plt.tight_layout()
                st.pyplot(plt.gcf())
                plt.close()

#TAB3: 산점도
with tab3:
    st.subheader("관계 탐색")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("총인구 vs AED 수 (점 크기=고령비)")
        plt.figure(figsize=(6, 4.5))
        sizes = (df_final["old_ratio"].fillna(0.0) * 800) + 20
        plt.scatter(df_final["pop_total"], df_final["aed_count"], s=sizes)  # 색 지정 X
        plt.xlabel("총인구")
        plt.ylabel("AED 수")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    with c2:
        st.caption("고령비 vs 1만명당 AED")
        plt.figure(figsize=(6, 4.5))
        plt.scatter(df_final["old_ratio"], df_final["aed_per_10k"])         # 색 지정 X
        plt.xlabel("고령비 (65+/총인구)")
        plt.ylabel("1만명당 AED")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

st.success("대시보드 준비 완료!")
