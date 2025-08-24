from contextlib import contextmanager
from datetime import datetime
import pandas as pd
import numpy as np
import re, os
from pathlib import Path

# 통계 패키지
from scipy import stats
import statsmodels.api as sm

# =========================================
# 포맷터(범용 숫자/NaN 대응)
# =========================================
def fmt_p(x, nd=3):
    """숫자/NaN 안전 포맷터.
    - 유효한 실수면 소수 nd자리
    - 절대값이 매우 작으면 지수표기(1e-4 미만)
    - None/NaN/Inf이면 'NA'
    """
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(xf):
        return "NA"
    if xf == 0.0:
        return "0"
    if abs(xf) < 1e-4:
        return f"{xf:.2e}"
    return f"{xf:.{nd}f}"

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

# 리포터 유틸
class HypoReporter:
    def __init__(self, to_file: bool = True, out_dir: str = "./output", filename: str | None = None):
        self.to_file = to_file
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hypothesis_report_{ts}.txt"
        self.path = os.path.join(out_dir, filename)
        if self.to_file:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("천안시 AED 최적화 - 가설 검증 리포트\n")
                f.write("="*78 + "\n\n")

    def _emit(self, text: str = ""):
        print(text)
        if self.to_file:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(text + ("\n" if not text.endswith("\n") else ""))

    @contextmanager
    def section(self, num: int, title: str, evidence: str | None = None):
        hdr = f"가설 {num}. {title}"
        sep = "-" * len(hdr)
        self._emit(hdr)
        self._emit(sep)
        if evidence:
            self._emit(f"- 근거: {evidence}")
        yield
        self._emit("")  # 공백 줄

    def bullet(self, text: str):
        self._emit(f"- {text}")

    def kv(self, key: str, value: str):
        self._emit(f"  · {key}: {value}")

    def table(self, df: pd.DataFrame, caption: str = "요약 통계(상위 5행)", max_rows: int = 5):
        if df is None or len(df) == 0:
            self._emit(f"- {caption}: (빈 데이터)")
            return
        preview = df.head(max_rows)
        self._emit(f"- {caption}:")
        self._emit(preview.to_string())
        self._emit("")

    def stat_line(self, name: str, **stats):
        # 숫자형은 fmt_p로 자동 정규화
        parts = [name + ":"]
        for k, v in stats.items():
            try:
                vv = float(v)
                parts.append(f"{k}={fmt_p(vv)}")
            except Exception:
                parts.append(f"{k}={v}")
        self._emit("- 검정: " + ", ".join(parts))

    def saved(self, what: str, path: str):
        self._emit(f"- 저장: {what} → {path}")

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
# 6) 가설4(리네이밍: 가설1): 도심(동) vs 읍/면 불균형 검정
# =========================================
grp_df = final.copy()

# 행정구역 유형: 동 / 읍면
def classify_emd(name: str):
    if not isinstance(name, str): return np.nan
    if name.endswith("동"): return "dong"
    if name.endswith("읍") or name.endswith("면"): return "eupmyeon"
    return "other"

grp_df["emd_type"] = grp_df["emd_nm"].apply(classify_emd)
grp_df = grp_df[grp_df["emd_type"].isin(["dong","eupmyeon"])].copy()

# 지표 준비
grp_df["aed_per_10k"]    = np.where(grp_df["pop_total"]>0, grp_df["aed_count"]/grp_df["pop_total"]*10000, np.nan)
grp_df["aed_per_1k_65p"] = np.where(grp_df["pop_65p"]>0,   grp_df["aed_count"]/grp_df["pop_65p"]*1000,   np.nan)

# 그룹별 요약
summary = (grp_df
           .groupby("emd_type")[["aed_count","pop_total","pop_65p","aed_per_10k","aed_per_1k_65p"]]
           .agg(["count","mean","std","median"]))

# 검정 함수
def two_group_tests(series_name):
    a = grp_df.loc[grp_df["emd_type"]=="dong", series_name].dropna()
    b = grp_df.loc[grp_df["emd_type"]=="eupmyeon", series_name].dropna()
    out = {"metric": series_name, "n_dong": len(a), "n_eupmyeon": len(b)}
    if len(a)>=3 and len(b)>=3:
        # Mann-Whitney (비모수)
        U, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out.update({"mw_U": U, "mw_p": p_u})
        # t-test (정규성 가정 참고용)
        t, p_t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        out.update({"t_stat": t, "t_p": p_t})
    else:
        out.update({"mw_U": np.nan, "mw_p": np.nan, "t_stat": np.nan, "t_p": np.nan, "note":"표본 부족"})
    return out

tests_g4 = []
for col in ["aed_per_10k","aed_per_1k_65p","aed_count"]:
    tests_g4.append(two_group_tests(col))

g4_out = pd.DataFrame(tests_g4)
g4_path = OUT / "aed_hypo4_group_compare.csv"
g4_out.to_csv(g4_path, index=False, encoding="utf-8-sig")

# =========================================
# 7) 접근성 가설(H5): 동별 최소거리 & 인구 커버리지
# =========================================
from math import radians, sin, cos, asin, sqrt
import warnings

AED_GEO_CSV = OUT / "aed_geocoded.csv"
aed_geo = None
if not AED_GEO_CSV.exists():
    warnings.warn(f"접근성 계산을 건너뜁니다. {AED_GEO_CSV} 가 없습니다. 우선 geocode_aed.py 를 실행하세요.")
else:
    aed_geo = pd.read_csv(AED_GEO_CSV, dtype={"lat":"float64","lon":"float64"})
    aed_geo = aed_geo.dropna(subset=["lat","lon"]).copy()
    if len(aed_geo)==0:
        warnings.warn("지오코딩된 AED가 없습니다."); aed_geo=None

if aed_geo is not None:
    # 7-1) 읍면동 중심점(centroid) 좌표 가져오기
    emd_cent = None
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        SHP1 = DATA / "bnd_dong_34011_2024_2Q.shp"
        SHP2 = DATA / "bnd_dong_34012_2024_2Q.shp"
        g1 = gpd.read_file(SHP1)
        g2 = gpd.read_file(SHP2)
        g = pd.concat([g1,g2], ignore_index=True)
        g = g.rename(columns={"ADM_CD":"adm_cd","ADM_NM":"emd_nm"})
        g = g.merge(codes_cheonan[["adm_cd","gu","emd_nm"]], on=["adm_cd","emd_nm"], how="inner")
        g_proj = g.to_crs(epsg=5179)
        centroids_proj = g_proj.geometry.centroid
        centroids_wgs  = centroids_proj.to_crs(epsg=4326)
        g = g_proj.to_crs(epsg=4326)
        g = g.assign(centroid=centroids_wgs)
        emd_cent = g[["adm_cd","gu","emd_nm","centroid"]].copy()
        emd_cent["lat"] = emd_cent["centroid"].y
        emd_cent["lon"] = emd_cent["centroid"].x
        emd_cent = pd.DataFrame(emd_cent.drop(columns=["centroid"]))
    except Exception as e:
        warnings.warn(f"geopandas 경로로 centroids 생성 실패: {e}. 센트로이드 근사 생략.")

    # 7-2) Haversine
    def hav_km(lat1, lon1, lat2, lon2):
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        c = 2*np.arcsin(np.sqrt(a))
        return 6371.0088 * c  # km

    # 7-3) 동별 최소거리
    access_df = final[["adm_cd","gu","emd_nm","pop_total","pop_65p"]].copy()
    if emd_cent is not None:
        access_df = access_df.merge(emd_cent, on=["adm_cd","gu","emd_nm"], how="left")
        a_lat = aed_geo["lat"].to_numpy()
        a_lon = aed_geo["lon"].to_numpy()
        mins = []
        for _, r in access_df.iterrows():
            if pd.isna(r["lat"]) or pd.isna(r["lon"]) or len(a_lat)==0:
                mins.append(np.nan); continue
            d_km = hav_km(r["lat"], r["lon"], a_lat, a_lon)  # 브로드캐스트
            mins.append(float(np.nanmin(d_km)))
        access_df["min_dist_m"] = np.array(mins) * 1000.0
    else:
        access_df["min_dist_m"] = np.nan

    # 7-4) 커버리지(200m/300m)
    access_df["cov200_pop_share"] = np.nan
    access_df["cov300_pop_share"] = np.nan
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        g1 = gpd.read_file(DATA / "bnd_dong_34011_2024_2Q.shp")
        g2 = gpd.read_file(DATA / "bnd_dong_34012_2024_2Q.shp")
        g = pd.concat([g1,g2], ignore_index=True)
        g = g.rename(columns={"ADM_CD":"adm_cd","ADM_NM":"emd_nm"})
        g = g.merge(codes_cheonan[["adm_cd","gu","emd_nm"]], on=["adm_cd","emd_nm"], how="inner")
        emd_gdf = g.to_crs(epsg=5179)
        aed_gdf = gpd.GeoDataFrame(aed_geo.dropna(subset=["lat","lon"]).copy(),
                                   geometry=gpd.points_from_xy(aed_geo["lon"], aed_geo["lat"]), crs="EPSG:4326").to_crs(5179)

        def coverage_share(buffer_m):
            aed_buf = aed_gdf.geometry.buffer(buffer_m)
            union_geom = aed_buf.union_all()
            inter_area = emd_gdf.geometry.intersection(union_geom).area
            share = (inter_area / emd_gdf.geometry.area).fillna(0.0).clip(0,1)
            return pd.Series(share.values, index=emd_gdf["adm_cd"].values)

        cov200 = coverage_share(200.0)
        cov300 = coverage_share(300.0)
        access_df = access_df.merge(cov200.rename("cov200_area_share"), left_on="adm_cd", right_index=True, how="left")
        access_df = access_df.merge(cov300.rename("cov300_area_share"), left_on="adm_cd", right_index=True, how="left")
        access_df["cov200_pop_share"] = access_df["cov200_area_share"]
        access_df["cov300_pop_share"] = access_df["cov300_area_share"]
    except Exception as e:
        warnings.warn(f"정밀 커버리지(면적) 계산 실패: {e}. 센트로이드 근사로 대체합니다.")
        access_df["cov200_pop_share"] = np.where(access_df["min_dist_m"]<=200, 1.0, 0.0)
        access_df["cov300_pop_share"] = np.where(access_df["min_dist_m"]<=300, 1.0, 0.0)

    # 저장
    acc_out = OUT / "aed_accessibility_by_emd.csv"
    access_df.to_csv(acc_out, index=False, encoding="utf-8-sig")

    # 접근성 요약/검정(도심 vs 읍면)
    def classify_emd(name: str):
        if isinstance(name, str) and name.endswith("동"): return "dong"
        if isinstance(name, str) and (name.endswith("읍") or name.endswith("면")): return "eupmyeon"
        return "other"

    access_df["emd_type"] = access_df["emd_nm"].apply(classify_emd)
    acc_sum = (access_df
               .groupby("emd_type")[["min_dist_m","cov200_pop_share","cov300_pop_share"]]
               .agg(["count","mean","std","median"]))
else:
    access_df, acc_sum = None, None

# =========================================
# 8) 리포팅 블록: 가설 N. 형식으로 깔끔 출력
# =========================================

report = HypoReporter(to_file=True, out_dir=str(OUT))

# 편의 함수: GLM 한 줄 요약
def _glm_one_liner(res, label):
    if res is None:
        return f"[{label}] 결과 없음"
    try:
        coef = res.params.get("old_ratio", np.nan)
        se   = res.bse.get("old_ratio", np.nan)
        p    = res.pvalues.get("old_ratio", np.nan)
        return f"[{label}] old_ratio: coef={fmt_p(coef,3)} (SE={fmt_p(se,3)}), p={fmt_p(p)}"
    except Exception:
        return f"[{label}] (요약 생성 오류)"

# 공통 데이터/산출물 안내
with report.section(
    num=0,
    title="데이터/산출물 생성 현황",
    evidence="천안시 읍·면·동 단위 AED·인구 집계 완료, GLM 진단/랭킹 포함."
):
    report.kv("행 수", str(len(final)))
    report.kv("AED 매핑 성공 건수", f"{int(aed['adm_cd'].notna().sum())} / {len(aed)}")
    report.kv("선택된 AED 시트명", str(sheet_name))
    report.kv("인구 기준 월", str(target_ym))
    if unmatched_path:
        report.saved("매핑 실패 AED 목록(보정용)", str(unmatched_path))
    report.saved("AED-인구 집계(CSV)", str(final_csv))
    report.saved("AED-인구 집계(XLSX)", str(final_xlsx))

# 가설 1. 도심(동) vs 읍·면 불균형
with report.section(
    num=1,
    title="도심(동) vs 읍·면 AED 불균형 검정",
    evidence="국내외에서 도심/비도심 간 설치 밀도·접근성 격차가 보고됨."
):
    report.table(summary, caption="읍·면·동 그룹별 AED/인구 지표 요약")
    for _, r in g4_out.iterrows():
        m = r["metric"]
        report.stat_line(f"{m} - Mann-Whitney", U=r.get("mw_U"), p=r.get("mw_p"))
        report.stat_line(f"{m} - t-test", t=r.get("t_stat"), p=r.get("t_p"))
    report.saved("그룹 비교 테이블", str(g4_path))

# 가설 2. 접근성 격차(동 vs 읍·면)
with report.section(
    num=2,
    title="접근성 격차(동 vs 읍·면): 최소거리·200/300m 커버리지 차이",
    evidence="비도심 지역은 시설 간격·밀도 문제로 AED 실접근성이 낮다는 보고."
):
    try:
        report.table(acc_sum, caption="접근성 지표 요약")
    except Exception:
        report.bullet("접근성 요약: (생성 실패/스킵)")
    try:
        if access_df is not None:
            from scipy import stats as _st
            a = access_df.loc[access_df["emd_type"]=="dong", "min_dist_m"].dropna()
            b = access_df.loc[access_df["emd_type"]=="eupmyeon", "min_dist_m"].dropna()
            if len(a)>=3 and len(b)>=3:
                U, p = _st.mannwhitneyu(a, b, alternative="two-sided")
                report.stat_line("Mann–Whitney (최소거리, m)", U=U, p=p)
            else:
                report.bullet("Mann–Whitney (최소거리, m): 표본 부족으로 생략")
            report.saved("접근성 지표 집계", str(OUT / "aed_accessibility_by_emd.csv"))
        else:
            report.bullet("접근성 데이터 없음: aed_geocoded.csv가 필요")
    except Exception as e:
        report.bullet(f"접근성 검정 스킵: {e}")

# 가설 3. AED 수가 인구/65+ 비례 가정을 따르는가?
with report.section(
    num=3,
    title="AED 수가 인구(또는 65+) 비례 가정을 따르는가",
    evidence="비례 가설에서 크게 벗어나면 수요 대비 과/소배치 가능성."
):
    for r in results:
        if "Chi-square" in r["test"]:
            chi2_str = fmt_p(r.get("chi2"), 2)
            df_val   = r.get("df")
            df_str   = str(int(df_val)) if (df_val is not None and pd.notna(df_val)) else "NA"
            p_str    = fmt_p(r.get("p_value"))
            msg = f"{r['test']}: chi2={chi2_str}, df={df_str}, p={p_str}"
            if "small_exp_ratio(<5)" in r:
                msg += f", 기대빈도<5 비율={r['small_exp_ratio(<5)']}"
            if r.get("note"):
                msg += f" ({r['note']})"
            report.bullet(msg)
    report.saved("검정 요약 CSV", str(OUT / "aed_tests_summary.csv"))

# 가설 4. AED 수와 인구(총/65+)의 상관관계
with report.section(
    num=4,
    title="AED 수와 총인구·고령인구의 상관관계",
    evidence="단순 상관으로 1차 연관성 점검(인과 아님)."
):
    for r in results:
        if "Correlation" in r["test"]:
            rho = r.get("spearman_rho")
            pr  = r.get("spearman_p")
            rr  = r.get("pearson_r")
            pp  = r.get("pearson_p")
            msg = (f"{r['test']}: "
                   f"Spearman rho={fmt_p(rho,3)} (p={fmt_p(pr)}), "
                   f"Pearson r={fmt_p(rr,3)} (p={fmt_p(pp)})")
            if r.get("note"):
                msg += f" ({r['note']})"
            report.bullet(msg)

# 가설 5. 고령비(old_ratio) → AED 수 영향 (오프셋=log 인구)
with report.section(
    num=5,
    title="고령비(old_ratio)가 AED 수에 미치는 영향 (오프셋=log 인구)",
    evidence="인구 규모 통제 후 고령 구조가 설치량과 연관되는지 GLM으로 확인."
):
    report.bullet(_glm_one_liner(poisson_res, "Poisson"))
    if np.isfinite(overdisp_ratio):
        report.kv("과산포(대략) deviance/df", f"{fmt_p(overdisp_ratio,2)}" + (" → 과산포 큼: NB 해석 권장" if overdisp_ratio>2 else ""))
    report.bullet(_glm_one_liner(nb_res, "NegativeBinomial"))
    report.saved("GLM 진단(예측/잔차)", str(OUT / "aed_glm_diagnostics.csv"))
    report.saved("과다 배치 Top10", str(OUT / f"aed_over_alloc_top10_{resid_label}.csv"))
    report.saved("과소 배치 Top10", str(OUT / f"aed_under_alloc_top10_{resid_label}.csv"))




# ============================
# H-UTIL: 공통 로더/공간도구
# ============================
import warnings
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

def _ensure_crs_wgs(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if g.crs is None:
        # 한국 TM일 가능성 높음
        try: g = g.set_crs(epsg=5179).to_crs(epsg=4326)
        except Exception: pass
    elif g.crs.to_epsg() != 4326:
        g = g.to_crs(epsg=4326)
    return g

def _load_shp_emds():
    """천안(동남/서북) 동경도형 shp → WGS84 + 면적(km2) 산출"""
    SHP1 = DATA / "bnd_dong_34011_2024_2Q.shp"  # 동남구
    SHP2 = DATA / "bnd_dong_34012_2024_2Q.shp"  # 서북구
    if not SHP1.exists() or not SHP2.exists():
        warnings.warn("SHP가 없어 공간 기반 가설(H1/H3/H5 일부)을 건너뜁니다.")
        return None
    g1 = gpd.read_file(SHP1); g2 = gpd.read_file(SHP2)
    g  = pd.concat([g1,g2], ignore_index=True)
    g  = g.rename(columns={"ADM_CD":"adm_cd","ADM_NM":"emd_nm"})
    g  = g.merge(codes_cheonan[["adm_cd","gu","emd_nm"]], on=["adm_cd","emd_nm"], how="inner")
    # 면적 계산은 투영좌표에서
    g_proj = g.to_crs(epsg=5179)
    g["area_km2"] = (g_proj.geometry.area / 1e6).astype(float)
    return _ensure_crs_wgs(g)

def _load_aed_points():
    """geocode_aed.py 산출물 사용 (필수: lat/lon)"""
    path = OUT / "aed_geocoded.csv"
    if not path.exists():
        warnings.warn("aed_geocoded.csv 없음 → AED 포인트가 필요한 가설(H1/H3/H5 등)을 건너뜁니다.")
        return None
    d = pd.read_csv(path)
    d = d.dropna(subset=["lat","lon"])
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["lon"], d["lat"]), crs="EPSG:4326")
    return g

def _load_events():
    """
    사건(심정지) 데이터 로더(열 자동 인식).
    - 지원 열(한국어 흔한 표기들): 위도/경도 또는 주소, '목격여부', '장소유형', '생존여부', '구급대 제세동', '소방서/구급대', '발생일시'
    - 좌표 없고 주소만 있으면: TODO(지오코딩) → 현재는 좌표 없으면 공간분석 제외
    """
    # === 요구 데이터 ===
    # /mnt/data/급성심장정지_발생_YYYYMM...xlsx 같은 원본을 data/에 복사해 이름을 고정하세요.
    cand = [DATA / "천안_심정지사건.xlsx", DATA / "급성심장정지_발생.xlsx"]
    src = next((p for p in cand if p.exists()), None)
    if src is None:
        warnings.warn("사건 엑셀을 찾지 못했어요(예: data/천안_심정지사건.xlsx). H3/H6/H7/H8/H10 일부 건너뜀.")
        return None

    ev = pd.read_excel(src, dtype=str)
    C = ev.columns.astype(str)

    def pick_cs(patterns):
        for p in patterns:
            hit = [c for c in C if re.search(p, c)]
            if hit: return hit[0]
        return None

    lat_c = pick_cs([r"위도", r"lat|LAT"])
    lon_c = pick_cs([r"경도", r"lon|LON"])
    addr_c= pick_cs([r"주소", r"지번|도로명"])
    surv_c= pick_cs([r"생존", r"퇴원|결과|CPC"])
    wit_c = pick_cs([r"목격", r"목격여부"])
    loc_c = pick_cs([r"장소", r"장소유형|발생장소"])
    ems_d= pick_cs([r"구급대.*제세동|EMS.*제세동|제세동.*구급"])
    statn= pick_cs([r"소방서|구급대|센터|센터명"])
    dt_c = pick_cs([r"발생일시|시간|date|datetime"])

    # 정규화 열
    out = pd.DataFrame()
    out["lat"]  = pd.to_numeric(ev.get(lat_c), errors="coerce") if lat_c else np.nan
    out["lon"]  = pd.to_numeric(ev.get(lon_c), errors="coerce") if lon_c else np.nan
    out["addr"] = ev.get(addr_c)
    out["witnessed"] = ev.get(wit_c)
    out["place_type"] = ev.get(loc_c)
    out["survival"] = ev.get(surv_c)
    out["ems_defib"] = ev.get(ems_d)
    out["ems_station"] = ev.get(statn)
    out["occurred_at"] = pd.to_datetime(ev.get(dt_c), errors="coerce") if dt_c else pd.NaT

    # 생존 이진화(대충 규칙)
    def _to_bin_surv(x):
        if not isinstance(x,str): return np.nan
        if re.search(r"생존|퇴원|good|CPC\s*1|CPC\s*2", x, re.I): return 1
        if re.search(r"사망|사체|CPC\s*3|CPC\s*4|CPC\s*5", x, re.I): return 0
        return np.nan
    out["survival_bin"] = out["survival"].apply(_to_bin_surv)

    # 목격 이진화
    out["witness_bin"] = out["witnessed"].astype(str).str.contains(r"유|있|Yes|Y", case=False, na=False).astype(int)

    # 장소 공공/비공공 라벨
    def _place_pub(x):
        if not isinstance(x,str): return np.nan
        if re.search(r"공원|거리|지하철|역|학교|관공서|상업|시장|공공|공중", x): return "public"
        if re.search(r"주거|자택|집|아파트|빌라|원룸", x): return "non_public"
        return "unknown"
    out["place_pub"] = out["place_type"].apply(_place_pub)

    # EMS 제세동 이진화
    out["ems_defib_bin"] = out["ems_defib"].astype(str).str.contains(r"시행|사용|Yes|Y|제세동", case=False, na=False).astype(int)

    # 좌표 → 점
    has_xy = out["lat"].notna() & out["lon"].notna()
    if has_xy.any():
        g = gpd.GeoDataFrame(out[has_xy].copy(),
                             geometry=gpd.points_from_xy(out.loc[has_xy,"lon"], out.loc[has_xy,"lat"]),
                             crs="EPSG:4326")
        return g, out  # (지오, 원본)
    else:
        warnings.warn("사건 좌표가 없어 공간거리 기반 가설은 제외됩니다(지오코딩 필요).")
        return None

# =========================================
# H1. 인구밀도 vs AED 커버리지(100m 버퍼)
# =========================================
def run_H1(reporter: HypoReporter):
    with reporter.section(1, "인구밀도 vs AED 100m 커버리지", "읍·면·동 면적/인구 + AED 좌표 + SHP"):
        emd_g = _load_shp_emds()
        aed_g = _load_aed_points()
        if emd_g is None or aed_g is None:
            reporter.bullet("SHP 또는 AED 포인트가 없어 분석 건너뜀.")
            return

        # 인구 병합/밀도
        emd = emd_g.merge(final[["adm_cd","pop_total"]], on="adm_cd", how="left")
        emd["pop_total"] = emd["pop_total"].fillna(0).astype(int)
        emd["pop_density"] = np.where(emd["area_km2"]>0, emd["pop_total"]/emd["area_km2"], np.nan)

        # 100m 버퍼: 거리 계산은 투영좌표로
        aed_proj = aed_g.to_crs(epsg=5179)
        aed_proj["buffer100"] = aed_proj.geometry.buffer(100.0)
        cov = gpd.GeoDataFrame(geometry=[unary_union(aed_proj["buffer100"].values)], crs="EPSG:5179").to_crs(epsg=4326)

        # 커버리지 비율 ≈ 면적비(areal weighting)
        emd_proj = emd.to_crs(epsg=5179)
        cov_proj = cov.to_crs(epsg=5179)
        inter_area = emd_proj.geometry.intersection(cov_proj.geometry.iloc[0]).area
        emd_proj["cover_rate_area"] = np.where(emd_proj["area_km2"]>0, (inter_area/1e6)/emd_proj["area_km2"], 0.0)
        emd_proj["covered_pop_est"] = emd_proj["pop_total"] * emd_proj["cover_rate_area"].clip(0,1)

        # 상관/회귀
        df_ = pd.DataFrame({
            "pop_density": emd_proj["pop_density"],
            "cov_rate": emd_proj["cover_rate_area"]
        }).replace([np.inf,-np.inf], np.nan).dropna()
        if len(df_)>=3 and df_["pop_density"].nunique()>1 and df_["cov_rate"].nunique()>1:
            r_s, p_s = stats.spearmanr(df_["pop_density"], df_["cov_rate"])
            r_p, p_p = stats.pearsonr(df_["pop_density"], df_["cov_rate"])
            reporter.stat_line("상관(인구밀도~커버리지)", spearman_rho=r_s, spearman_p=p_s, pearson_r=r_p, pearson_p=p_p)
            # 단순 OLS
            X = sm.add_constant(df_[["pop_density"]].astype(float))
            y = df_["cov_rate"].astype(float)
            ols = sm.OLS(y, X).fit()
            out_path = OUT / "H1_popdens_cov_ols.txt"
            with open(out_path,"w",encoding="utf-8") as f: f.write(ols.summary().as_text())
            reporter.saved("H1 OLS 결과", str(out_path))
        else:
            reporter.bullet("표본/분산 부족으로 상관/회귀 생략.")

# =========================================
# H2. 고령인구 비율 vs AED 수(불일치) → 카이제곱
# =========================================
def run_H2(reporter: HypoReporter):
    with reporter.section(2, "고령비율과 AED 수요 불일치(카이제곱)", "65세 비율 vs 인구1천명당 AED"):
        df = final.copy()
        df["old_ratio"] = np.where(df["pop_total"]>0, df["pop_65p"]/df["pop_total"], np.nan)
        df["aed_per_1k"] = np.where(df["pop_total"]>0, df["aed_count"]/df["pop_total"]*1000, np.nan)
        X = df[["old_ratio","aed_per_1k"]].replace([np.inf,-np.inf], np.nan).dropna()
        if len(X)<6:
            reporter.bullet("표본 부족.")
            return
        # 사분위로 2x2 테이블(상/하)
        q_old = X["old_ratio"].median()
        q_aed = X["aed_per_1k"].median()
        ct = pd.crosstab(X["old_ratio"]>=q_old, X["aed_per_1k"]>=q_aed)
        chi2, p, dof, _ = stats.chi2_contingency(ct.values)
        reporter.stat_line("카이제곱(고령↑ vs AED밀도↑)", chi2=chi2, p_value=p, df=dof)
        out = OUT / "H2_table.csv"; ct.to_csv(out, encoding="utf-8-sig"); reporter.saved("H2 교차표", str(out))

# =========================================
# H3. 실제 심정지 발생지 vs AED 거리 / 100m 이내 비율
# =========================================
def run_H3(reporter: HypoReporter):
    with reporter.section(3, "사건-최근접 AED 거리 및 100m 이내 비율", "사건 좌표 + AED 좌표"):
        aed = _load_aed_points()
        ev = _load_events()
        if aed is None or ev is None:
            reporter.bullet("필요 데이터 없음.")
            return
        ev_g, ev_raw = ev
        # 거리 계산(미터) - 투영
        A = aed.to_crs(epsg=5179); E = ev_g.to_crs(epsg=5179)
        aed_tree = A.sindex
        dists = []
        for geom in E.geometry:
            idx = list(aed_tree.nearest(geom.bounds, 1))[0]
            d = geom.distance(A.geometry.iloc[idx])
            dists.append(d)
        E["dist_m"] = dists
        within100 = (E["dist_m"]<=100).mean()
        reporter.kv("사건 수", str(len(E)))
        reporter.kv("100m 이내 사건 비율", f"{within100*100:.1f}%")
        E[["lat","lon","dist_m"]].to_csv(OUT / "H3_event_nearest_distance.csv", index=False, encoding="utf-8-sig")

# =========================================
# H4. 아파트 단지 내 설치 비효율성(아파트 vs 비아파트, 사건 근접/사용대리)
# =========================================
def run_H4(reporter: HypoReporter):
    with reporter.section(4, "아파트 단지 내 AED 설치 효율성", "시설명 분류 + 사건 근접도"):
        aed = _load_aed_points()
        ev = _load_events()
        if aed is None or ev is None:
            reporter.bullet("필요 데이터 없음.")
            return
        aed["is_apartment"] = aed.get("facility_name","").astype(str).str.contains(r"아파트|APT|APT\.", case=False, regex=True)
        # 사건과의 거리 요약
        A = aed.to_crs(epsg=5179); E = ev[0].to_crs(epsg=5179)
        idxs = []
        for geom in E.geometry:
            nearest_idx = list(A.sindex.nearest(geom.bounds, 1))[0]
            idxs.append(nearest_idx)
        near = A.iloc[idxs].copy()
        near["from_event_dist_m"] = E.geometry.reset_index(drop=True).distance(near.geometry)
        # 아파트/비아파트 간 사건근접도 비교(비모수)
        a = near.loc[near["is_apartment"], "from_event_dist_m"].dropna()
        b = near.loc[~near["is_apartment"], "from_event_dist_m"].dropna()
        if len(a)>=3 and len(b)>=3:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            reporter.stat_line("Mann-Whitney(거리: 아파트 vs 비아파트)", U=U, p_value=p, n_apartment=len(a), n_nonap=len(b))
        else:
            reporter.bullet("표본 부족.")

# =========================================
# H5. 야간·주말 접근성 격차(시설유형 분류 → 커버리지 비교)
# =========================================
def run_H5(reporter: HypoReporter):
    with reporter.section(5, "야간·주말 접근성 격차", "시설유형 라벨링(24시간/제한) + 커버리지"):
        aed = _load_aed_points()
        emd_g = _load_shp_emds()
        if aed is None or emd_g is None:
            reporter.bullet("필요 데이터 없음.")
            return
        # 간단 라벨: 편의점/병원/약국/소방/경찰/터미널/호텔 등은 24H 가능성 높음, 학교/관공서/체육관 등은 제한
        def _open_cat(name: str):
            if not isinstance(name,str): return "unknown"
            if re.search(r"편의점|CU|GS25|세븐일레븐|병원|응급실|약국(24|심야)|소방|경찰|터미널|호텔|모텔|숙소|역", name): return "open24"
            if re.search(r"학교|초등|중학교|고등학교|대학교|관공서|주민센터|체육관|문화센터|도서관", name): return "limited"
            return "unknown"
        aed["open_cat"] = aed.get("facility_name","").apply(_open_cat)

        # 100m 버퍼 union by group
        def _cov_union(sub: gpd.GeoDataFrame):
            proj = sub.to_crs(epsg=5179)
            buf = unary_union(proj.geometry.buffer(100.0).values)
            return gpd.GeoSeries([buf], crs=proj.crs).to_crs(epsg=4326).iloc[0]

        cats = ["open24","limited"]
        covs = {}
        for c in cats:
            gsub = aed[aed["open_cat"]==c]
            if len(gsub)==0: continue
            covs[c] = _cov_union(gsub)

        if not covs:
            reporter.bullet("라벨링 결과 유효 카테고리 없음.")
            return

        emd_proj = emd_g.to_crs(epsg=5179)
        res = []
        for c, cov_geom in covs.items():
            cov_p = gpd.GeoSeries([cov_geom], crs="EPSG:4326").to_crs(epsg=5179).iloc[0]
            inter = emd_proj.geometry.intersection(cov_p).area
            rate = (inter/1e6) / emd_proj["area_km2"]
            res.append(rate.rename(c))
        cov_df = pd.concat(res, axis=1).clip(lower=0, upper=1)
        cov_df = cov_df.join(final.set_index("adm_cd")[["pop_total"]], how="left")
        # 단순 비교(면적가중 X): 평균 커버리지
        out_summary = cov_df[cats].mean().to_frame("mean_area_cov")
        out_path = OUT / "H5_access_coverage.csv"
        out_summary.to_csv(out_path, encoding="utf-8-sig")
        reporter.saved("H5 커버리지 요약", str(out_path))

# =========================================
# H6. 사회경제적 수준과 생존율(로지스틱)
# =========================================
def run_H6(reporter: HypoReporter):
    with reporter.section(6, "사회경제적 수준과 생존율(로지스틱)", "사건 생존여부 + (읍면동/구 레벨) 재정자료"):
        ev = _load_events()
        if ev is None:
            reporter.bullet("사건 데이터 없음.")
            return
        ev_g, ev_raw = ev
        # 필요: emd 매칭(포인트를 SHP 폴리곤에 spatial join)
        emd = _load_shp_emds()
        if emd is None:
            reporter.bullet("SHP 없음 → 공간 조인 불가.")
            return
        j = gpd.sjoin(ev_g.to_crs(emd.crs), emd[["adm_cd","gu","emd_nm","geometry"]], how="left", predicate="within")
        # 사회경제 지표 CSV(예: data/cheonan_finance.csv, 열: adm_cd 또는 gu, 재정자립도/복지비 등)
        econ_path = DATA / "cheonan_finance.csv"
        if not econ_path.exists():
            reporter.bullet("재정 지표 파일 없음(cheonan_finance.csv). 로지스틱 건너뜀.")
            return
        econ = pd.read_csv(econ_path, dtype=str)
        # adm_cd 우선, 없으면 gu 매칭
        if "adm_cd" in econ.columns:
            jj = j.merge(econ, on="adm_cd", how="left")
        else:
            jj = j.merge(econ, on="gu", how="left")
        # 생존 이진 + 연속/범주 설명변수 구성 (예: finance_ratio)
        y = jj["survival_bin"].astype(float)
        xcols = [c for c in econ.columns if c not in ["adm_cd","gu","emd_nm"]]
        if not xcols:
            reporter.bullet("설명변수 없음.")
            return
        X = sm.add_constant(jj[xcols].apply(pd.to_numeric, errors="coerce"))
        mask = y.notna() & np.isfinite(X).all(axis=1)
        if mask.sum()<30:
            reporter.bullet("표본 부족.")
            return
        logit = sm.Logit(y[mask], X[mask]).fit(disp=False)
        with open(OUT / "H6_logit_summary.txt","w",encoding="utf-8") as f:
            f.write(logit.summary().as_text())
        reporter.saved("H6 로지스틱 결과", str(OUT / "H6_logit_summary.txt"))

# =========================================
# H7. 공공장소 vs 비공공장소 발생 격차(생존률/AED 커버율)
# =========================================
def run_H7(reporter: HypoReporter):
    with reporter.section(7, "공공 vs 비공공 발생 격차", "사건 장소유형 + 생존/커버"):
        ev = _load_events()
        if ev is None:
            reporter.bullet("사건 데이터 없음.")
            return
        _, ev_raw = ev
        # 생존률 교차
        sub = ev_raw.dropna(subset=["place_pub","survival_bin"])
        ct = pd.crosstab(sub["place_pub"], sub["survival_bin"])
        if ct.shape[0]>=2 and ct.shape[1]>=2:
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
            reporter.stat_line("카이제곱(장소유형~생존)", chi2=chi2, p_value=p, df=dof)
        else:
            reporter.bullet("장소/생존 표본 부족.")

# =========================================
# H8. 목격 여부 효과(생존률 비교)
# =========================================
def run_H8(reporter: HypoReporter):
    with reporter.section(8, "목격 여부 효과(생존률)", "사건 목격여부 + 생존"):
        ev = _load_events()
        if ev is None:
            reporter.bullet("사건 데이터 없음.")
            return
        _, ev_raw = ev
        sub = ev_raw.dropna(subset=["witness_bin","survival_bin"])
        ct = pd.crosstab(sub["witness_bin"], sub["survival_bin"])
        if ct.shape==(2,2):
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
            reporter.stat_line("카이제곱(목격~생존)", chi2=chi2, p_value=p, df=dof)
            # OR
            a,b,c,d = ct.values.flatten()  # [[no_surv, surv]…] 일 수 있어 안전 처리:
            # 재정렬(0/1 x 0/1 가정: witness=0,1 / survival=0,1)
            try:
                a = ct.loc[0,0]; b = ct.loc[0,1]; c = ct.loc[1,0]; d = ct.loc[1,1]
            except Exception:
                pass
            OR = (d * a) / max(b * c, 1e-9)
            reporter.kv("오즈비(목격→생존)", fmt_p(OR))
        else:
            reporter.bullet("2x2 표 아님.")

# =========================================
# H9. 일반인 CPR 교육 수준과 생존율(상관/회귀; 외부지표 필요)
# =========================================
def run_H9(reporter: HypoReporter):
    with reporter.section(9, "CPR 교육 수준과 생존율", "읍면동/구별 교육률 CSV 필요"):
        # 예시 스키마: data/cheonan_cpr_edu.csv [adm_cd(or gu), cpr_rate]
        path = DATA / "cheonan_cpr_edu.csv"
        ev = _load_events()
        if not path.exists() or ev is None:
            reporter.bullet("교육률 파일 혹은 사건 데이터 없음.")
            return
        ev_g, _ = ev
        emd = _load_shp_emds()
        if emd is None:
            reporter.bullet("SHP 없음.")
            return
        j = gpd.sjoin(ev_g.to_crs(emd.crs), emd[["adm_cd","gu","geometry"]], how="left", predicate="within")
        edu = pd.read_csv(path, dtype=str)
        if "adm_cd" in edu.columns:
            jj = j.merge(edu, on="adm_cd", how="left")
        else:
            jj = j.merge(edu, on="gu", how="left")
        # 집계: emd/구별 생존률
        surv = jj.groupby("adm_cd", as_index=False)["survival_bin"].mean().rename(columns={"survival_bin":"survival_rate"})
        merged = surv.merge(edu, on="adm_cd", how="left")
        merged["cpr_rate"] = pd.to_numeric(merged["cpr_rate"], errors="coerce")
        merged = merged.dropna(subset=["cpr_rate","survival_rate"])
        if len(merged)<5:
            reporter.bullet("표본 부족.")
            return
        r, p = stats.pearsonr(merged["cpr_rate"], merged["survival_rate"])
        reporter.stat_line("상관(CPR 교육률~생존률)", pearson_r=r, pearson_p=p)
        merged.to_csv(OUT / "H9_cpr_vs_survival.csv", index=False, encoding="utf-8-sig")
        reporter.saved("H9 테이블", str(OUT / "H9_cpr_vs_survival.csv"))

# =========================================
# H10. 119 구급대 제세동 활용 지역격차(ANOVA)
# =========================================
def run_H10(reporter: HypoReporter):
    with reporter.section(10, "119 구급대 제세동 활용 지역격차(ANOVA)", "사건별 EMS 제세동 + 소방서/구급대 코드"):
        ev = _load_events()
        if ev is None:
            reporter.bullet("사건 데이터 없음.")
            return
        _, raw = ev
        sub = raw.dropna(subset=["ems_station"])
        if sub.empty:
            reporter.bullet("소방서/구급대 열 없음.")
            return
        grp = sub.groupby("ems_station")["ems_defib_bin"].mean().dropna()
        if grp.index.nunique() < 2:
            reporter.bullet("집단 수 부족.")
            return
        # 일원분산분석
        samples = [g["ems_defib_bin"].dropna().values for _, g in raw.groupby("ems_station")]
        samples = [s for s in samples if len(s)>=3]
        if len(samples) < 2:
            reporter.bullet("표본 부족.")
            return
        F, p = stats.f_oneway(*samples)
        reporter.stat_line("ANOVA(EMS 제세동률 ~ 소방서)", F=F, p_value=p)
        grp.sort_values(ascending=False).to_csv(OUT / "H10_ems_defib_rate_by_station.csv", encoding="utf-8-sig")
        reporter.saved("H10 소방서별 제세동률", str(OUT / "H10_ems_defib_rate_by_station.csv"))

# ============================
# MAIN: 실행 스위치
# ============================
if __name__ == "__main__":
    rep = HypoReporter(to_file=True, out_dir=str(OUT), filename="hypothesis_run.txt")
    # 이미 기존 코드에서 final/각종 산출물을 만든 다음 아래 호출이 실행되도록 배치하세요.
    run_H1(rep)
    run_H2(rep)
    run_H3(rep)
    run_H4(rep)
    run_H5(rep)
    run_H6(rep)
    run_H7(rep)
    run_H8(rep)
    run_H9(rep)
    run_H10(rep)
    print(f"[완료] 리포트: {rep.path}")



# 간단 가이드
with report.section(num=6, title="해석 가이드(짧게)", evidence=None):
    report.bullet("카이제곱: p<0.05 → ‘인구(또는 65+) 비례’ 가정에서 유의하게 벗어남")
    report.bullet("상관: p<0.05 & 계수 크면 AED 수와 인구 사이 연관성 시사(인과 아님)")
    report.bullet("GLM: old_ratio 계수 유의·양(+) → 인구 통제하 고령비 ↑ 지역에 AED ↑ 경향")
