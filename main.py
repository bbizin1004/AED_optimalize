# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from pathlib import Path

# 통계 패키지
from scipy import stats
import statsmodels.api as sm

# =========================================
# 0) 경로/입력 파일
# =========================================
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT  = BASE / "output"
OUT.mkdir(parents=True, exist_ok=True)

CODES_XLSX   = DATA / "센서스 공간정보 지역 코드.xlsx"
AED_ALL_XLSX = DATA / "자동심장충격기(AED)_통합.xlsx"              # 통합본(시트 중 '천안')
AGE5_XLSX    = DATA / "행정구역_읍면동_별_5세별_주민등록인구_2011년__20250813010009.xlsx"  # 천안시 전용

# =========================================
# 유틸
# =========================================
def norm(s):
    if pd.isna(s): return None
    s = str(s)
    s = re.sub(r"\s+", "", s)
    return s.replace("·", "")

def pick_col(cols, patterns):
    for p in patterns:
        for c in cols:
            if re.search(p, str(c)):
                return c
    return None

def strip_full(s):
    if pd.isna(s): return None
    return re.sub(r"\u3000+", "", str(s)).strip()

# =========================================
# 1) 행정구역 코드 (천안시만)
# =========================================
codes = pd.read_excel(CODES_XLSX, header=1, dtype=str)
codes["시도코드"]   = codes["시도코드"].str.zfill(2)
codes["시군구코드"] = codes["시군구코드"].str.zfill(3)
codes["읍면동코드"] = codes["읍면동코드"].str.zfill(3)
codes["adm_cd"]     = codes["시도코드"] + codes["시군구코드"] + codes["읍면동코드"]
codes = codes.rename(columns={"시도명칭":"sido_nm","시군구명칭":"sigungu_nm","읍면동명칭":"emd_nm"})

codes_cheonan = codes[codes["sigungu_nm"].str.contains("천안시", na=False)].copy()
codes_cheonan["gu"]      = codes_cheonan["sigungu_nm"].str.extract(r"(동남구|서북구)")
codes_cheonan["emd_key"] = codes_cheonan["emd_nm"].map(norm)
codes_cheonan["gu_key"]  = codes_cheonan["gu"].map(norm)
codes_cheonan = codes_cheonan[["adm_cd","gu","emd_nm","emd_key","gu_key"]].drop_duplicates()

# =========================================
# 2) AED (통합본의 '천안' 시트만) → 지번 기반 매핑(+ 분동 자동 분해 + 전역 fallback)
# =========================================
xls = pd.ExcelFile(AED_ALL_XLSX)
sheet_name = next((n for n in xls.sheet_names if "천안" in str(n)), None)
if sheet_name is None:
    raise RuntimeError(f"통합본에서 '천안'이 포함된 시트를 찾지 못했습니다. 시트들: {xls.sheet_names}")

aed_raw = pd.read_excel(AED_ALL_XLSX, sheet_name=sheet_name, dtype=str)
cols = list(aed_raw.columns)

# 주소/시설명 컬럼 탐색
sido_col   = pick_col(cols, [r"^시도$", r"시\s*도"])
sgg_col    = pick_col(cols, [r"^시군구$", r"시\s*군\s*구", r"시군구명"])
emd_col    = pick_col(cols, [r"^읍면동$", r"읍\s*면\s*동", r"행정동", r"법정동"])
road_col   = pick_col(cols, [r"도로명주소"])
jibun_col  = pick_col(cols, [r"지번주소"])
addr_col_generic = pick_col(cols, [r"^주소$", r"상세주소"])
facility_col = pick_col(cols, [r"설치소|시설명|시설|장소|기관|설치장소"])

aed = pd.DataFrame()
if facility_col and facility_col in aed_raw:
    aed["facility_name"] = aed_raw[facility_col]

# 지번만 별도로 보존(매칭 핵심)
aed["jibun"] = aed_raw[jibun_col] if jibun_col and jibun_col in aed_raw else pd.NA

# (로그/보정표용) 전체 address는 유지
def build_address(r):
    parts = []
    for c in [sido_col, sgg_col, emd_col, jibun_col, road_col, addr_col_generic]:
        if c and pd.notna(r.get(c)) and str(r.get(c)).strip():
            parts.append(str(r[c]))
    return " ".join(parts) if parts else pd.NA

aed["address"] = aed_raw.apply(build_address, axis=1)

# 지번에서 '구'와 '동/읍/면/리' 힌트 추출
gu_re   = re.compile(r"(동남구|서북구)")
emd_re  = re.compile(r"([가-힣0-9]+(?:동|읍|면|리))")

def extract_from_jibun(jibun: str):
    if not isinstance(jibun, str):
        return pd.NA, pd.NA
    parts = re.split(r"[,\u3001/;]+", jibun)
    best_gu, best_emd = pd.NA, pd.NA
    for p in parts:
        g = gu_re.search(p)
        e = emd_re.search(p)
        if g and pd.isna(best_gu): best_gu = g.group(1)
        if e and pd.isna(best_emd): best_emd = e.group(1)
        if not pd.isna(best_gu) and not pd.isna(best_emd):
            break
    return best_gu, best_emd

aed[["gu_from_jibun","emd_from_jibun"]] = aed["jibun"].apply(lambda x: pd.Series(extract_from_jibun(x)))

# gu는 지번에서 뽑은 값으로 우선 덮어쓰기
def infer_gu_from_all(row):
    if isinstance(row.get("gu_from_jibun"), str) and row["gu_from_jibun"].strip():
        return row["gu_from_jibun"]
    if sgg_col and sgg_col in aed_raw:
        s = aed_raw.loc[row.name, sgg_col]
        if isinstance(s, str):
            if "동남구" in s: return "동남구"
            if "서북구" in s: return "서북구"
    a = row.get("address")
    if isinstance(a, str):
        if "동남구" in a: return "동남구"
        if "서북구" in a: return "서북구"
    return np.nan

aed["gu"]     = aed.apply(infer_gu_from_all, axis=1)
aed["gu_key"] = aed["gu"].map(norm)

# 중복 제거(선택)
if "facility_name" in aed.columns and "address" in aed.columns:
    aed = aed.drop_duplicates(subset=["facility_name","address"])

# 주소 → 읍/면/동 매핑 준비
emd_list = codes_cheonan[["adm_cd","gu","emd_nm"]].copy()
emd_list["emd_key"]  = emd_list["emd_nm"].apply(lambda s: re.sub(r"\s+","", s).replace("·",""))
emd_list["root"]     = emd_list["emd_nm"].str.replace(r"(동|읍|면|리)$", "", regex=True)
emd_list["root_key"] = emd_list["root"].apply(lambda s: re.sub(r"\s+","", s) if isinstance(s,str) else s)

# 정규식/도움함수
emd_paren_re       = re.compile(r"\(([^)]+)\)")
emd_word_re        = re.compile(r"([가-힣0-9]+(?:동|읍|면|리))")
only_digits_hyphen = re.compile(r"^\d+(-\d+)?$")

def get_pools(gu_val):
    pool_city = emd_list
    pool_gu = emd_list[emd_list["gu"]==gu_val] if pd.notna(gu_val) else emd_list.iloc[0:0]
    return pool_city, pool_gu

def match_key_in_pool(key, pool):
    if not key: return None
    key = re.sub(r"\s+","", str(key)).replace("·","")
    hit = pool[pool["emd_key"]==key]
    return hit.iloc[0] if len(hit) else None

def _extract_tokens(text: str) -> list[str]:
    toks = []
    if not isinstance(text, str) or not text.strip():
        return toks
    for inside in emd_paren_re.findall(text):
        m = emd_word_re.search(inside)
        if m: toks.append(m.group(1))
    words = [w for w in re.split(r"\s+", text) if w]
    words = [w for w in words if not only_digits_hyphen.match(w)]
    for w in words:
        m = emd_word_re.search(w)
        if m: toks.append(m.group(1))
    toks = sorted(set(toks), key=len, reverse=True)
    return toks

def get_root(name: str):
    if not isinstance(name, str): return None
    name = name.strip()
    name = re.sub(r"(\D)(\d+)동$", r"\1동", name)  # 성정1동 → 성정동
    name = name.replace("·","")
    return name

def pick_best_candidate(cands_df, addr_full: str, facility: str):
    if cands_df is None or len(cands_df)==0:
        return None
    if len(cands_df)==1:
        return cands_df.iloc[0]
    text = " ".join([t for t in [addr_full, facility] if isinstance(t,str)]).replace("·","")
    text_nospace = re.sub(r"\s+","", text)
    scores = []
    for idx, r in cands_df.iterrows():
        emd = str(r["emd_nm"])
        cnt = text.count(emd)  # 정확 명칭 등장 횟수
        k = r["emd_key"]
        longest = 0
        if isinstance(k,str) and k and k in text_nospace:
            longest = len(k)
        score = cnt * 10 + longest  # 등장 횟수 크게 가중, 길이 보조
        scores.append((score, idx))
    scores.sort(reverse=True)
    best_idx = scores[0][1]
    return cands_df.loc[best_idx]

def try_match_on_pool(row, pool):
    """한 개의 pool(구 또는 전역)에 대해: 정확키 → 루트후보(분동) → 지번 부분문자열 → address 보조 → 시설명 보조 순 시도."""
    addr_full = row.get("address")
    fn = row.get("facility_name")
    jibun = row.get("jibun")

    # a) 입력 파일의 행정동/법정동 직접 컬럼
    if emd_col and emd_col in aed_raw:
        raw_val = aed_raw.loc[row.name, emd_col]
        if pd.notna(raw_val) and str(raw_val).strip():
            hit = match_key_in_pool(raw_val, pool)
            if hit is not None:
                return (hit["emd_nm"], hit["adm_cd"])

    # b) 지번에서 추출한 emd 힌트
    emd_hint = row.get("emd_from_jibun")
    if isinstance(emd_hint, str) and emd_hint.strip():
        hit = match_key_in_pool(emd_hint, pool)
        if hit is not None:
            return (hit["emd_nm"], hit["adm_cd"])
        root = get_root(emd_hint)
        if root:
            root_key = norm(root)
            cand = pool[pool["root_key"] == root_key]
            if len(cand) >= 1:
                best = pick_best_candidate(cand, addr_full, fn)
                if best is not None:
                    return (best["emd_nm"], best["adm_cd"])

    # c) 지번 텍스트 전체 토큰 → 정확키 → 루트 후보 → 부분문자열
    if isinstance(jibun, str) and jibun.strip():
        tokens = _extract_tokens(jibun)
        for t in tokens:
            hit = match_key_in_pool(t, pool)
            if hit is not None:
                return (hit["emd_nm"], hit["adm_cd"])
        for t in tokens:
            root = get_root(t)
            if root:
                root_key = norm(root)
                cand = pool[pool["root_key"] == root_key]
                if len(cand) >= 1:
                    best = pick_best_candidate(cand, addr_full, fn)
                    if best is not None:
                        return (best["emd_nm"], best["adm_cd"])
        text = re.sub(r"\s+","", jibun).replace("·","")
        best = None; best_len = 0
        for _, r in pool.iterrows():
            k = r["emd_key"]
            if k and k in text and len(k) > best_len:
                best, best_len = r, len(k)
        if best is not None:
            return (best["emd_nm"], best["adm_cd"])

    # d) address 보조(정확키 → 루트)
    if isinstance(addr_full, str) and addr_full.strip():
        text = re.sub(r"\s+","", addr_full).replace("·","")
        for _, r in pool.iterrows():
            if r["emd_key"] and r["emd_key"] in text:
                return (r["emd_nm"], r["adm_cd"])
        roots = set(pool["root_key"].dropna().unique().tolist())
        for rk in sorted(roots, key=len, reverse=True):
            if rk and rk in text:
                cand = pool[pool["root_key"] == rk]
                best = pick_best_candidate(cand, addr_full, fn)
                if best is not None:
                    return (best["emd_nm"], best["adm_cd"])

    # e) 시설명 어근(루트) 보조
    if isinstance(fn, str) and fn.strip():
        text = re.sub(r"\s+","", fn).replace("·","")
        cand = pool[pool["root_key"].apply(lambda rk: isinstance(rk,str) and rk and rk in text)]
        if len(cand):
            best = pick_best_candidate(cand, addr_full, fn)
            if best is not None:
                return (best["emd_nm"], best["adm_cd"])

    return None

def find_emd_smart(row):
    """구 풀 → 실패 시 전역 풀(fallback) 순으로 시도."""
    gu_val = row.get("gu")
    pool_city, pool_gu = get_pools(gu_val)

    # 1) 구 풀에서 시도
    hit = try_match_on_pool(row, pool_gu)
    if hit is not None:
        return pd.Series(hit)

    # 2) 전역 풀에서 재시도
    hit = try_match_on_pool(row, pool_city)
    if hit is not None:
        return pd.Series(hit)

    # 최종 실패
    return pd.Series([pd.NA, pd.NA])

aed[["emd_nm","adm_cd"]] = aed.apply(find_emd_smart, axis=1)

# 매핑 실패 목록 저장(보정용)
unmatched = aed[aed["adm_cd"].isna()][["facility_name","address","gu"]]
unmatched_path = None
if len(unmatched) > 0:
    unmatched_path = OUT / "aed_unmatched_for_fix.csv"
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

# 수기 보정 반영(옵션) — aed_unmatched_for_fix.csv에 emd_nm_fix 추가 시 자동 반영
fix_path = OUT / "aed_unmatched_for_fix.csv"
if fix_path.exists():
    fix = pd.read_csv(fix_path, dtype=str)
    if "emd_nm_fix" in fix.columns:
        fix2 = fix.dropna(subset=["emd_nm_fix"]).copy()
        fix2 = fix2.merge(codes_cheonan[["adm_cd","gu","emd_nm"]],
                          left_on=["gu","emd_nm_fix"], right_on=["gu","emd_nm"], how="left")
        aed = aed.merge(fix2[["facility_name","address","adm_cd"]],
                        on=["facility_name","address"], how="left", suffixes=("","_fix"))
        aed["adm_cd"] = aed["adm_cd"].fillna(aed["adm_cd_fix"])
        aed = aed.drop(columns=["adm_cd_fix"])

# AED 집계
aed_counts = (aed.dropna(subset=["adm_cd"])
                .groupby("adm_cd", as_index=False).size()
                .rename(columns={"size":"aed_count"}))

# =========================================
# 3) 인구 (천안시 전용 5세 연령대 → 최신월 자동)
# =========================================
age = pd.read_excel(AGE5_XLSX, sheet_name="데이터", dtype=str)
age["name"] = age["행정구역(동읍면)별"].apply(strip_full)
age2 = age[age["항목"].astype(str).eq("총인구수 (명)")].copy()

age2["gu"] = np.where(age2["name"].str.endswith("구"), age2["name"], np.nan)
age2["gu"] = age2["gu"].ffill()
age2 = age2[age2["name"].str.endswith(("동","읍","면"), na=False)].copy()

ym_cols = [c for c in age2.columns if re.fullmatch(r"\d{4}\.\d{2}(\.\d+)?", str(c))]
ym_prefixes = sorted(set([re.match(r"(\d{4}\.\d{2})", c).group(1) for c in ym_cols]))
if not ym_prefixes:
    raise RuntimeError("연월(YYYY.MM) 형식의 컬럼을 찾지 못했습니다.")
target_ym = ym_prefixes[-1]
target_cols = [c for c in ym_cols if c.startswith(target_ym)]
labels_map = dict(zip(target_cols, age.loc[0, target_cols].tolist()))

def to_num(x):
    try: return int(str(x).replace(",",""))
    except: return np.nan
for c in target_cols:
    age2[c] = age2[c].apply(to_num)

# '계' 라벨 안전 확인
pop_total_candidates = [c for c,l in labels_map.items() if l == "계"]
if not pop_total_candidates:
    raise RuntimeError(f"'{target_ym}' 구간에서 '계' 라벨을 찾지 못했습니다. labels={set(labels_map.values())}")
pop_total_col = pop_total_candidates[0]

senior_labels = ['65 - 69세','70 - 74세','75 - 79세','80 - 84세','85 - 89세','90 - 94세','95 - 99세','100+']
senior_cols = [c for c,l in labels_map.items() if l in senior_labels]
if not senior_cols:
    raise RuntimeError(f"'{target_ym}' 구간에서 65+ 라벨 컬럼을 찾지 못했습니다. labels={set(labels_map.values())}")

age2 = age2.rename(columns={"name":"emd_nm"})
age2["pop_total"] = age2[pop_total_col]
age2["pop_65p"]   = age2[senior_cols].sum(axis=1, min_count=1)

pop = (age2.merge(codes_cheonan[["adm_cd","gu","emd_nm"]],
                  on=["gu","emd_nm"], how="left")
            .dropna(subset=["adm_cd"]))
pop = pop[["adm_cd","gu","emd_nm","pop_total","pop_65p"]].drop_duplicates()

# =========================================
# 4) 병합/지표
# =========================================
final = (codes_cheonan[["adm_cd","gu","emd_nm"]]
         .merge(pop, on=["adm_cd","gu","emd_nm"], how="left")
         .merge(aed_counts, on="adm_cd", how="left"))

final["pop_total"]   = final["pop_total"].fillna(0).astype(int)
final["pop_65p"]     = final["pop_65p"].fillna(0).astype(int)
final["aed_count"]   = final["aed_count"].fillna(0).astype(int)
final["aed_per_10k"] = np.where(final["pop_total"] > 0, final["aed_count"] / final["pop_total"] * 10000, np.nan)
final["aed_per_1k_65p"] = np.where(final["pop_65p"] > 0, final["aed_count"] / final["pop_65p"] * 1000, np.nan)

# 저장
final_csv  = OUT / "cheonan_eupmyeondong_aed_pop.csv"
final_xlsx = OUT / "cheonan_eupmyeondong_aed_pop.xlsx"
final.sort_values(["gu","emd_nm"]).to_csv(final_csv, index=False, encoding="utf-8-sig")
final.sort_values(["gu","emd_nm"]).to_excel(final_xlsx, index=False)

# =========================================
# 5) 통계 검정: 독립성/연관성
# =========================================
df = final.copy()
df = df[(df["pop_total"]>0) | (df["pop_65p"]>0)].copy()

# 관측치
obs = df["aed_count"].astype(int).to_numpy()
total_aed = int(obs.sum())

# 기대치(총인구 비례, 65+ 비례)
den_pop = float(df["pop_total"].sum())
if den_pop > 0:
    pop_share = df["pop_total"] / den_pop
else:
    pop_share = pd.Series(np.nan, index=df.index)

den_old = float(df["pop_65p"].sum())
if den_old > 0:
    old_share = df["pop_65p"] / den_old
else:
    old_share = pd.Series(np.nan, index=df.index)

exp_pop = (total_aed * pop_share).to_numpy() if den_pop > 0 else np.full_like(obs, np.nan, dtype=float)
exp_old = (total_aed * old_share).to_numpy() if den_old > 0 else np.full_like(obs, np.nan, dtype=float)

def chisq_result(observed, expected, label):
    expected = np.asarray(expected, dtype=float)
    mask = np.isfinite(expected) & (expected > 0)
    if mask.sum() <= 1:
        return {"test": f"Chi-square ({label})", "chi2": np.nan, "p_value": np.nan, "df": np.nan,
                "note": "유효 기대빈도 부족"}
    chi2, p = stats.chisquare(f_obs=observed[mask], f_exp=expected[mask])
    small_exp_ratio = float((expected[mask] < 5).sum()) / mask.sum()
    return {"test": f"Chi-square ({label})", "chi2": chi2, "p_value": p, "df": mask.sum()-1,
            "small_exp_ratio(<5)": round(small_exp_ratio, 3)}

results = []
results.append(chisq_result(obs, exp_pop, "인구 비례"))
if np.isfinite(exp_old).any() and np.nansum(exp_old) > 0:
    results.append(chisq_result(obs, exp_old, "65세이상 비례"))

# 상관검정(안정 버전)
def corr_result(x, y, label):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    out = {"test": f"Correlation ({label})"}
    if len(x) < 3 or x.nunique() <= 1 or y.nunique() <= 1:
        out.update({"spearman_rho": np.nan, "spearman_p": np.nan,
                    "pearson_r": np.nan,  "pearson_p": np.nan,
                    "note": "표본/분산 부족"})
        return out
    rho_s, p_s = stats.spearmanr(x, y)
    r_p,  p_p  = stats.pearsonr(x, y)
    out.update({"spearman_rho": rho_s, "spearman_p": p_s,
                "pearson_r": r_p,  "pearson_p": p_p})
    return out

results.append(corr_result(df["pop_total"], df["aed_count"], "AED~총인구"))
results.append(corr_result(df["pop_65p"],   df["aed_count"], "AED~65세이상"))

# 포아송 회귀: offset=log(총인구), 설명변수=고령비(old_ratio)
glm_df = df[(df["pop_total"] > 0)].copy()
glm_df["old_ratio"] = np.where(glm_df["pop_total"]>0, glm_df["pop_65p"]/glm_df["pop_total"], 0.0)

y = glm_df["aed_count"].astype(int).to_numpy()
X = sm.add_constant(glm_df[["old_ratio"]].astype(float))
offset = np.log(glm_df["pop_total"].astype(float))

poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
poisson_res = poisson_model.fit()

overdisp_ratio = poisson_res.deviance / poisson_res.df_resid if poisson_res.df_resid>0 else np.nan

# 필요 시 Negative Binomial (과산포 완화)
nb_res = None
if np.isfinite(overdisp_ratio) and overdisp_ratio > 2:
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
    nb_res = nb_model.fit()

# 진단 저장
glm_out = glm_df.copy()
glm_out["pred_poisson"] = poisson_res.predict()
glm_out["resid_poisson"] = glm_out["aed_count"] - glm_out["pred_poisson"]
if nb_res is not None:
    glm_out["pred_nb"] = nb_res.predict()
    glm_out["resid_nb"] = glm_out["aed_count"] - glm_out["pred_nb"]
glm_out.to_csv(OUT / "aed_glm_diagnostics.csv", index=False, encoding="utf-8-sig")

# 검정 결과 저장
tests_df = pd.DataFrame(results)
tests_df.to_csv(OUT / "aed_tests_summary.csv", index=False, encoding="utf-8-sig")

# === 추가 산출물: ‘과다/과소 배치’ 랭킹 ===
rank_df = glm_out.merge(df[["adm_cd","gu","emd_nm"]], left_index=True, right_index=True, how="left")
if "resid_nb" in rank_df.columns:
    rank_df["resid_used"] = rank_df["resid_nb"]
    rank_df["pred_used"]  = rank_df["pred_nb"]
    resid_label = "nb"
else:
    rank_df["resid_used"] = rank_df["resid_poisson"]
    rank_df["pred_used"]  = rank_df["pred_poisson"]
    resid_label = "poisson"

rank_df["std_resid"] = (rank_df["resid_used"] / np.sqrt(rank_df["pred_used"].clip(lower=1e-9)))
over_top  = rank_df.sort_values("std_resid", ascending=False).head(10)
under_top = rank_df.sort_values("std_resid", ascending=True).head(10)
over_top.to_csv(OUT / f"aed_over_alloc_top10_{resid_label}.csv",  index=False, encoding="utf-8-sig")
under_top.to_csv(OUT / f"aed_under_alloc_top10_{resid_label}.csv", index=False, encoding="utf-8-sig")

# =========================================
# 출력 로그
# =========================================
print(f"[OK] 저장: {final_csv}")
print(f"[OK] 저장: {final_xlsx}")
print("행 수:", len(final))
print("AED 매핑 성공 건수:", int(aed["adm_cd"].notna().sum()), "/", len(aed))
print("선택된 AED 시트명:", sheet_name)
print("인구 기준 월:", target_ym)
if unmatched_path:
    print(f"매핑 실패 AED 목록(보정용): {unmatched_path}")

print("\n=== [카이제곱 적합도 검정: 비례 가정] ===")
for r in results:
    if "Chi-square" in r["test"]:
        msg = f"{r['test']}: chi2={r['chi2']:.2f}, df={r['df']}, p={r['p_value']:.4g}"
        if "small_exp_ratio(<5)" in r:
            msg += f", 기대빈도<5 비율={r['small_exp_ratio(<5)']}"
        if r.get("note"): msg += f" ({r['note']})"
        print(msg)

print("\n=== [상관 검정] ===")
for r in results:
    if "Correlation" in r["test"]:
        base = (f"{r['test']}: Spearman rho={r['spearman_rho'] if pd.notna(r['spearman_rho']) else 'NA'} "
                f"(p={r['spearman_p'] if pd.notna(r['spearman_p']) else 'NA'}), "
                f"Pearson r={r['pearson_r'] if pd.notna(r['pearson_r']) else 'NA'} "
                f"(p={r['pearson_p'] if pd.notna(r['pearson_p']) else 'NA'})")
        if r.get("note"): base += f" ({r['note']})"
        print(base)

print("\n=== [포아송 회귀: AED ~ old_ratio; offset=log(총인구)] ===")
print(poisson_res.summary())
print(f"\n과산포(대략) deviance/df={overdisp_ratio:.2f} "
      f"{'→ 과산포 큼: NegativeBinomial 결과 아래 참고' if (np.isfinite(overdisp_ratio) and overdisp_ratio>2) else ''}")

if nb_res is not None:
    print("\n=== [Negative Binomial 회귀 결과] ===")
    print(nb_res.summary())

print("\n[해석 가이드]")
print("- 카이제곱: p<0.05면 ‘인구(또는 65+) 비례’ 가정에서 유의하게 벗어남 → 인구와 무관(독립) 가설은 지지되지 않음.")
print("- 상관: p<0.05 & 계수 크면 AED 수와 인구 사이 연관성 시사.")
print("- GLM: old_ratio(고령비) 계수가 유의하고 양(+)이면 고령비가 높을수록 인구규모 통제하에서도 AED가 많음.")
print("- 과산포가 크면 NB 결과를 우선 참고.")
print(f"[랭킹 저장] 과다: {OUT / f'aed_over_alloc_top10_{resid_label}.csv'}")
print(f"[랭킹 저장] 과소: {OUT / f'aed_under_alloc_top10_{resid_label}.csv'}")
