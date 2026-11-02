"""
Tests for src/concordance. Run from the repo root:
    python -m unittest tests.test_concordance        (or: python -m pytest tests/test_concordance.py)
Uses only synthetic data; no MIMIC access needed.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.concordance import checks, confirmed, criteria, icd, kdigo, stats
from src.concordance.config import Config
from src.concordance.run import run
from src.concordance.stats import Suppressor


# ---------------------------------------------------------------------------
# Synthetic MIMIC-IV
# ---------------------------------------------------------------------------

def make_synthetic_mimic(root: Path, n_patients: int = 1500, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    (root / "hosp").mkdir(parents=True)
    (root / "icu").mkdir(parents=True)
    base = pd.Timestamp("2150-01-01")
    pats, adms, dxs, icus, labs = [], [], [], [], []
    outs, wts, procs_ev, procs_icd = [], [], [], []
    hadm = 20000000
    stay = 30000000
    lab_id = 0
    for sid in range(1000, 1000 + n_patients):
        female = rng.random() < 0.5
        pats.append(dict(subject_id=sid, gender="F" if female else "M",
                         anchor_age=int(rng.integers(20, 90)), anchor_year=2150,
                         anchor_year_group=["2008 - 2010", "2011 - 2013", "2014 - 2016"][int(rng.integers(0, 3))]))
        b = float(np.exp(rng.normal(np.log(0.9), 0.25)))
        t0 = base + pd.Timedelta(days=int(rng.integers(0, 200)))
        for k in range(int(rng.integers(1, 4))):
            hadm += 1
            admit = t0 + pd.Timedelta(days=150 * k)
            los = int(rng.integers(2, 15))
            disch = admit + pd.Timedelta(days=los)
            adms.append(dict(subject_id=sid, hadm_id=hadm, admittime=admit, dischtime=disch))
            # outpatient labs before admission (null hadm_id)
            for _ in range(int(rng.integers(0, 3))):
                lab_id += 1
                labs.append((lab_id, sid, np.nan, 50912, admit - pd.Timedelta(days=int(rng.integers(1, 200)), minutes=int(rng.integers(0, 600))), b * rng.normal(1, 0.05)))
            aki = rng.random() < 0.35
            has_icu = rng.random() < 0.55
            if has_icu:
                stay += 1
                intime = admit + pd.Timedelta(hours=6)
                outtime = min(intime + pd.Timedelta(days=int(rng.integers(1, 5))), disch)
                icus.append(dict(subject_id=sid, hadm_id=hadm, stay_id=stay, intime=intime, outtime=outtime))
                if rng.random() > 0.1:                              # some stays have no recorded weight
                    wts.append(dict(stay_id=stay, patientweight=float(rng.normal(80, 12))))
                low_uo = aki or rng.random() < 0.15                 # 15% of non-AKI stays: oliguria without a creatinine rise
                hours = int((outtime - intime).total_seconds() // 3600)
                for hr in range(hours):
                    rate = (6.0 if low_uo and hr >= 2 else 80.0) * rng.uniform(0.8, 1.2)
                    outs.append(dict(subject_id=sid, hadm_id=hadm, stay_id=stay,
                                     charttime=intime + pd.Timedelta(hours=hr, minutes=30), itemid=226559, value=rate))
                if aki and rng.random() < 0.08:
                    procs_ev.append(dict(stay_id=stay, hadm_id=hadm, starttime=intime + pd.Timedelta(hours=10), itemid=225441))
            peak_day = float(rng.uniform(0.5, los))
            mult = float(rng.choice([1.35, 1.6, 2.2, 3.4])) if aki else 1.0
            for d in np.arange(0.1, los, 0.5 + rng.random() * 0.6):
                lab_id += 1
                bump = 1 + (mult - 1) * np.exp(-((d - peak_day) ** 2) / 2.0)
                v = round(b * bump * rng.normal(1, 0.04), 2)
                labs.append((lab_id, sid, hadm, 50912, admit + pd.Timedelta(days=float(d), seconds=int(rng.integers(0, 60))), v))
            if rng.random() < 0.1:   # urine creatinine must be ignored
                lab_id += 1
                labs.append((lab_id, sid, hadm, 51082, admit + pd.Timedelta(hours=5), 80.0))
            v10 = rng.random() < 0.5
            pick = lambda a, b_: b_ if v10 else a  # noqa: E731
            codes = [(pick("4019", "I10"), 1)]
            if rng.random() < (0.7 if aki else 0.04):
                codes.append((pick("5849", "N179"), 2))
            if rng.random() < 0.05:
                codes.append((pick("586", "N19"), 3))
            r = rng.random()
            if r < 0.08:
                codes.append((pick("5856", "N186"), 4))
            elif r < 0.2:
                codes.append((pick("5859", "N183"), 4))
            if rng.random() < 0.02:
                procs_icd.append(dict(hadm_id=hadm, seq_num=1, icd_code="3995", icd_version=9))
            for c, s in codes:
                dxs.append(dict(subject_id=sid, hadm_id=hadm, seq_num=s, icd_code=c, icd_version=10 if v10 else 9))
    pd.DataFrame(pats).to_csv(root / "hosp/patients.csv.gz", index=False)
    pd.DataFrame(adms).to_csv(root / "hosp/admissions.csv.gz", index=False)
    pd.DataFrame(dxs).to_csv(root / "hosp/diagnoses_icd.csv.gz", index=False)
    pd.DataFrame(icus).to_csv(root / "icu/icustays.csv.gz", index=False)
    pd.DataFrame(outs).to_csv(root / "icu/outputevents.csv.gz", index=False)
    pd.DataFrame(wts).assign(subject_id=0, hadm_id=0).to_csv(root / "icu/inputevents.csv.gz", index=False)
    pd.DataFrame(procs_ev).to_csv(root / "icu/procedureevents.csv.gz", index=False)
    pd.DataFrame(procs_icd).to_csv(root / "hosp/procedures_icd.csv.gz", index=False)
    pd.DataFrame([(50912, "Creatinine", "Blood", "Chemistry"), (52546, "Creatinine", "Blood", "Chemistry"),
                  (51082, "Creatinine, Urine", "Urine", "Chemistry")],
                 columns=["itemid", "label", "fluid", "category"]).to_csv(root / "hosp/d_labitems.csv.gz", index=False)
    pd.DataFrame(labs, columns=["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
                 ).assign(specimen_id=1).to_csv(root / "hosp/labevents.csv.gz", index=False)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestKdigoCore(unittest.TestCase):
    def test_rolling_min_matches_pandas(self):
        rng = np.random.default_rng(1)
        t = np.sort(rng.choice(np.arange(0, 400 * 3600, 60), size=300, replace=False))
        v = rng.uniform(0.5, 3, size=t.size)
        mine = kdigo.rolling_min_48h(t, v)
        ref = pd.Series(v, index=pd.to_datetime(t, unit="s")).rolling("48h").min().to_numpy()
        np.testing.assert_allclose(mine, ref)

    def test_prior_min_bruteforce(self):
        rng = np.random.default_rng(2)
        t = np.sort(rng.choice(np.arange(0, 60 * 86400, 600), size=200, replace=False))
        v = rng.uniform(0.5, 3, size=t.size)
        mine = kdigo.prior_min(t, v)
        for i in range(t.size):
            m = (t < t[i]) & (t >= t[i] - 7 * 86400)
            if m.any():
                self.assertAlmostEqual(mine[i], v[m].min())
            else:
                self.assertTrue(np.isnan(mine[i]))

    def test_ties_are_order_independent(self):
        t = np.array([0, 0, 3600]); a = kdigo.rolling_min_48h(t, np.array([1.0, 1.4, 1.4]))
        b = kdigo.rolling_min_48h(t, np.array([1.4, 1.0, 1.4]))
        np.testing.assert_allclose(a, b)

    def test_backcalc_roundtrip_egfr_75(self):
        age = np.tile(np.arange(20, 95, 5), 2)
        female = np.repeat([False, True], age.size // 2)
        np.testing.assert_allclose(kdigo.ckd_epi_2021(kdigo.backcalc_scr(age, female), age, female), 75.0, rtol=1e-9)

    def test_stage_thresholds(self):
        v = np.array([1.5, 2.0, 3.0, 1.0, 4.0, 1.2])
        stage, _ = kdigo.stage_labs(v, min48=np.array([1.5, 2.0, 3.0, 1.0, 4.0, 1.2]), baseline=np.full(6, 1.0))
        self.assertEqual(stage.tolist(), [1, 2, 3, 0, 3, 0])
        # absolute rise alone gives stage 1; no baseline -> only the absolute rule can fire
        stage, _ = kdigo.stage_labs(np.array([1.4]), np.array([1.0]), np.array([np.nan]))
        self.assertEqual(stage.tolist(), [1])

    def test_b0_equals_existing_onset_function(self):
        mod = kdigo.load_existing_onset_module()
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(1, 15))
            t = np.sort(rng.choice(np.arange(0, 10 * 86400, 600), size=n, replace=False))
            v = np.round(rng.uniform(0.6, 3.0, size=n), 2)
            base = float(rng.choice([np.nan, round(float(rng.uniform(0.6, 1.6)), 2)]))
            win = kdigo.make_window(pd.DataFrame({"hadm_id": 1, "t": t, "valuenum": v, "prior7d_min": np.nan}))
            bb = np.nanmin(v) if np.isnan(base) else base          # B0 = median else in-admission min
            mine = kdigo.stage_window(win, win.per_lab([bb]))["max_stage"].iloc[0] >= 1
            df = pd.DataFrame({"charttime": pd.to_datetime(t, unit="s"), "valuenum": v})
            self.assertEqual(mine, mod.kdigo_onset_for_admission(df, base) is not None)


class TestUrineAndDialysis(unittest.TestCase):
    @staticmethod
    def stage(hours_vols, weight=80.0):
        h = np.array([x[0] for x in hours_vols]); v = np.array([x[1] for x in hours_vols], dtype=float)
        return criteria.stage_one_stay(h, v, weight)

    def test_stages_follow_kdigo_thresholds(self):
        normal = lambda n: [(i, 80.0) for i in range(n)]                   # 1 mL/kg/h
        self.assertEqual(self.stage(normal(30)), 0)
        self.assertEqual(self.stage([(i, 20.0) for i in range(6)]), 1)      # 0.25 mL/kg/h for 6 h
        self.assertEqual(self.stage([(i, 30.0) for i in range(12)]), 2)     # 0.375 mL/kg/h for 12 h
        self.assertEqual(self.stage([(i, 20.0) for i in range(24)]), 3)     # < 0.3 for 24 h
        self.assertEqual(self.stage([(i, 0.0) for i in range(12)]), 3)      # anuria 12 h
        self.assertEqual(self.stage([(i, 30.0) for i in range(5)]), 0)      # only 5 h: below the 6 h window
        # 6 h at 0.45 mL/kg/h is stage 1; 0.55 mL/kg/h is not
        self.assertEqual(self.stage([(i, 36.0) for i in range(6)]), 1)
        self.assertEqual(self.stage([(i, 44.0) for i in range(6)]), 0)

    def test_charting_gaps_are_not_oliguria(self):
        sparse = [(0, 5.0), (5, 5.0)]               # two charted hours inside a 6 h window
        self.assertEqual(self.stage(sparse), 0)

    def test_by_stay_weight_and_assessability(self):
        ev = pd.DataFrame({"stay_id": [1] * 6 + [2] * 6, "hour": list(range(6)) * 2, "vol": [10.0] * 12})
        res = criteria.uo_stage_by_stay(ev, pd.Series({1: 80.0, 2: 5.0}))   # stay 2: implausible weight
        self.assertEqual(int(res.loc[1, "uo_stage"]), 1)
        self.assertTrue(res.loc[1, "assessable"]); self.assertFalse(res.loc[2, "assessable"])
        self.assertEqual(int(res.loc[2, "uo_stage"]), 0)

    def test_dialysis_codes(self):
        pr = pd.DataFrame({"hadm_id": [1, 2, 3, 4, 5], "icd_version": [9, 9, 10, 10, 9],
                           "icd_code": ["3995", "3895", "5A1D90Z", "B50W0ZZ", "5498"]})
        self.assertEqual(criteria.rrt_icd_hadm(pr), {1, 3, 5})              # catheter / shunt imaging are not dialysis


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = stats.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertTrue(np.isnan(stats.wilson(0, 0)[0]))

    def test_kappa(self):
        self.assertAlmostEqual(stats.kappa(50, 0, 0, 50), 1.0)
        self.assertAlmostEqual(stats.kappa(20, 5, 10, 15), 0.4, places=6)   # po=.7, pe=.5

    def test_bootstrap_deterministic(self):
        a = stats.bootstrap_kappa_ci((40, 10, 5, 45), 200, 1, "k")
        b = stats.bootstrap_kappa_ci((40, 10, 5, 45), 200, 1, "k")
        self.assertEqual(a, b)
        self.assertLess(a[0], stats.kappa(40, 10, 5, 45))
        self.assertGreater(a[1], stats.kappa(40, 10, 5, 45))

    def test_suppression(self):
        s = stats.Suppressor(10)
        self.assertEqual(s.count(9), "<10")
        self.assertEqual(s.count(0), "<10")
        self.assertEqual(s.count(10), "10")
        self.assertEqual(s.pct(5, 100, 5, 95), "suppressed")
        self.assertEqual(s.pct(50, 100, 50, 50), "50.00")


class TestIcd(unittest.TestCase):
    def test_sets_and_versions(self):
        dx = pd.DataFrame({
            "hadm_id": [1, 2, 3, 4, 5, 6, 6],
            "icd_code": ["5849", "N179", "5845", "586", "N186", "58410", "N19"],
            "icd_version": [9, 10, 9, 9, 10, 9, 10]})
        f = icd.flag_admissions(dx)
        self.assertTrue(f.loc[1, "AKI_CORE"] and f.loc[2, "AKI_CORE"] and f.loc[3, "AKI_CORE"])
        self.assertFalse(f.loc[4, "AKI_CORE"]); self.assertTrue(f.loc[4, "AKI_BROAD"])
        self.assertTrue(f.loc[5, "ESRD"] and f.loc[5, "CKD"])
        self.assertFalse(f.loc[6, "AKI_CORE"])                   # 584.10 is not 5845-5849
        self.assertEqual(f.loc[6, "icd_version_group"], "mixed")
        self.assertEqual(f.loc[1, "icd_version_group"], "9")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        make_synthetic_mimic(root / "mimic")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cfg(self, out, min_cell):
        return Config(mimic_dir=str(self.root / "mimic"), outdir=str(self.root / out),
                      cache_dir=str(self.root / "cache"), min_cell=min_cell, n_boot=100, seed=5,
                      pipeline_target=None, modeling_target=None, quiet=True)

    def test_outputs_checks_and_determinism(self):
        run(self.cfg("out_a", 1))            # internal assertions: 2x2 sums, partitions, B0 == existing, output scan
        run(self.cfg("out_b", 1))
        a, b = self.root / "out_a", self.root / "out_b"
        names = sorted(p.name for p in a.iterdir())
        for want in ("d1_agreement_by_stage.csv", "d1_two_by_two.csv", "d2_criterion.csv",
                     "d3_baseline_incidence.csv", "d3_baseline_agreement.csv", "d3_variant_pairwise.csv",
                     "d4_discordance_icd_pos.csv", "d4_discordance_kdigo_pos.csv", "d5_stratified.csv", "summary.md"):
            self.assertIn(want, names)
            self.assertEqual((a / want).read_bytes(), (b / want).read_bytes(), f"{want} not byte-identical")
        checks.verify_outputs(str(a))

        d1 = pd.read_csv(a / "d1_two_by_two.csv", dtype=str)
        cnt = lambda col: col.map(lambda x: 0 if str(x).startswith("<") else int(x))  # noqa: E731
        self.assertTrue(((cnt(d1["tp"]) + cnt(d1["fp"]) + cnt(d1["fn"]) + cnt(d1["tn"])) == cnt(d1["n"])).all())
        st = pd.read_csv(a / "d1_agreement_by_stage.csv", dtype=str)
        for (w, s, c), g in st.groupby(["window", "icd_set", "criteria"]):
            parts = cnt(g[g["kdigo_stage"] != "all"]["n_adm"]).sum()
            self.assertEqual(parts, cnt(g[g["kdigo_stage"] == "all"]["n_adm"]).iloc[0])

        num = lambda col: col.map(lambda x: 0 if str(x).startswith("<") else int(x))  # noqa: E731  (0 prints as "<1")
        for tbl in ("d4_discordance_icd_pos", "d4_discordance_kdigo_pos"):
            d4 = pd.read_csv(a / f"{tbl}.csv", dtype=str)
            self.assertEqual(num(d4[d4["level"] == "reason"]["n"]).sum(), num(d4[d4["level"] == "total"]["n"]).iloc[0])
        icd_pos = pd.read_csv(a / "d4_discordance_icd_pos.csv", dtype=str)
        parent_n = num(icd_pos[(icd_pos["level"] == "reason") & (icd_pos["reason"] == "OUTSIDE_ICU_WINDOW")]["n"]).iloc[0]
        subs_n = num(icd_pos[(icd_pos["level"] == "subreason") & (icd_pos["parent"] == "OUTSIDE_ICU_WINDOW")]["n"]).sum()
        self.assertEqual(parent_n, subs_n)

        # the synthetic AKI events must show up: % ICD+ rises with stage
        r = d1[(d1["window"] == "W_HOSP") & (d1["icd_set"] == "AKI_CORE") & (d1["kdigo_threshold"] == "any")].iloc[0]
        fpr = int(r["fp"]) / (int(r["fp"]) + int(r["tn"]))
        self.assertGreater(float(r["sensitivity"]), fpr)

    def test_modeling_cohort_uses_the_model_label(self):
        mimic = self.root / "mimic"
        dx = pd.read_csv(mimic / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
        aki = dx[dx["icd_code"].isin(["5849", "N179"])]["hadm_id"].unique()
        adm = pd.read_csv(mimic / "hosp/admissions.csv.gz")
        keep = adm.iloc[::2]                                   # an arbitrary labeled subset
        tgt = pd.DataFrame({"subject_id": keep["subject_id"], "hadm_id": keep["hadm_id"], "version": "mimic4",
                            "incident_aki_label": keep["hadm_id"].isin(aki).astype(float)})
        tgt.loc[tgt.index[:5], "incident_aki_label"] = np.nan  # unlabeled rows must be dropped
        tpath = self.root / "target_syn.parquet"
        tgt.to_parquet(tpath)
        cfg = self.cfg("out_m", 1)
        cfg.modeling_target = str(tpath)
        run(cfg)
        d1 = pd.read_csv(self.root / "out_m" / "d1_two_by_two.csv", dtype=str)
        r = d1[(d1["cohort"] == "modeling") & (d1["window"] == "W_HOSP") & (d1["icd_set"] == "AKI_CORE")
               & (d1["kdigo_threshold"] == "any")].iloc[0]
        cnt = lambda x: 0 if str(x).startswith("<") else int(x)  # noqa: E731
        labeled = tgt.dropna()
        self.assertEqual(cnt(r["n"]), len(labeled))
        self.assertEqual(cnt(r["tp"]) + cnt(r["fp"]), int(labeled["incident_aki_label"].sum()))
        self.assertIn("Primary cohort for the results below: `modeling`", (self.root / "out_m" / "summary.md").read_text())

    def test_urine_output_and_dialysis_in_outputs(self):
        run(self.cfg("out_u", 1))
        out = self.root / "out_u"
        cnt = lambda col: col.map(lambda x: 0 if str(x).startswith("<") else int(x))  # noqa: E731
        d1 = pd.read_csv(out / "d1_two_by_two.csv", dtype=str)
        crits = set(d1[d1["window"] == "W_ICU"]["criteria"])
        self.assertEqual(crits, {"SCR_ONLY", "SCR+UO+RRT"})
        self.assertEqual(set(d1[d1["window"] == "W_HOSP"]["criteria"]), {"SCR_ONLY", "SCR+RRT"})
        d2 = pd.read_csv(out / "d2_criterion.csv", dtype=str)
        self.assertEqual(set(d2["status"]), {"ok"})
        self.assertEqual(d2[d2["group"] == "ALL"]["pct_of_kdigo_pos"].iloc[0], "100.00")
        groups = d2.set_index("group")
        parts = sum(cnt(groups.loc[[g], "n"]).iloc[0] for g in ("SCR_ONLY", "UO_ONLY", "BOTH", "RRT_ONLY"))
        self.assertEqual(parts, cnt(groups.loc[["ALL"], "n"]).iloc[0])
        r = d1[(d1["window"] == "W_ICU") & (d1["icd_set"] == "AKI_CORE") & (d1["kdigo_threshold"] == "any")
               & (d1["criteria"] == "SCR+UO+RRT")].iloc[0]
        self.assertEqual(cnt(groups.loc[["ALL"], "n"]).iloc[0], cnt(pd.Series([r["tp"]])).iloc[0] + cnt(pd.Series([r["fn"]])).iloc[0])
        self.assertGreater(cnt(groups.loc[["UO_ONLY"], "n"]).iloc[0], 0)    # synthetic oliguria without a creatinine rise
        full = d1[(d1["window"] == "W_ICU") & (d1["icd_set"] == "AKI_CORE") & (d1["kdigo_threshold"] == "any")]
        scr_pos = cnt(full[full["criteria"] == "SCR_ONLY"]["tp"]).iloc[0] + cnt(full[full["criteria"] == "SCR_ONLY"]["fn"]).iloc[0]
        all_pos = cnt(full[full["criteria"] == "SCR+UO+RRT"]["tp"]).iloc[0] + cnt(full[full["criteria"] == "SCR+UO+RRT"]["fn"]).iloc[0]
        self.assertGreater(all_pos, scr_pos)                                # adding UO / RRT only ever adds KDIGO+

    def test_switching_criteria_off_gives_creatinine_only(self):
        cfg = self.cfg("out_n", 1)
        cfg.use_uo = cfg.use_rrt = False
        run(cfg)
        out = self.root / "out_n"
        d1 = pd.read_csv(out / "d1_two_by_two.csv", dtype=str)
        self.assertEqual(set(d1["criteria"]), {"SCR_ONLY"})
        d2 = pd.read_csv(out / "d2_criterion.csv", dtype=str)
        self.assertEqual(set(d2["status"]), {"skipped"})

    def test_small_cells_are_suppressed(self):
        run(self.cfg("out_s", 10))
        out = self.root / "out_s"
        count_cols = {"n", "tp", "fp", "fn", "tn", "n_adm", "n_icd_pos", "n_same", "n_diff"}
        n_masked = 0
        for p in out.glob("*.csv"):
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            for c in count_cols & set(df.columns):
                for x in df[c]:
                    if x in ("NA", ""):
                        continue
                    if x.startswith("<"):
                        n_masked += 1
                        self.assertEqual(x, "<10")
                    else:
                        self.assertGreaterEqual(int(x), 10, f"{p.name}:{c}")
        self.assertGreater(n_masked, 0, "expected some small cells in the synthetic data")

    def test_verify_outputs_rejects_ids_and_timestamps(self):
        bad = self.root / "bad"
        bad.mkdir()
        pd.DataFrame({"hadm_id": ["1"], "n": ["5"]}).to_csv(bad / "x.csv", index=False)
        with self.assertRaises(AssertionError):
            checks.verify_outputs(str(bad))
        (bad / "x.csv").write_text("n\n2150-01-01 08:00:00\n")
        with self.assertRaises(AssertionError):
            checks.verify_outputs(str(bad))


class TestConfirmedUnits(unittest.TestCase):
    @staticmethod
    def window(rows):
        df = pd.DataFrame(rows, columns=["hadm_id", "h", "valuenum"])
        df["t"] = (df["h"] * 3600).astype("int64")
        df["prior7d_min"] = np.nan
        return kdigo.make_window(df[["hadm_id", "t", "valuenum", "prior7d_min"]])

    def test_arrival_fall(self):
        w = self.window([
            (1, 0, 3.0), (1, 12, 2.0), (1, 40, 1.0),      # peak on arrival, falls by >= 0.3 -> True
            (2, 0, 1.0), (2, 30, 1.2), (2, 60, 1.5),      # rises only -> False
            (3, 0, 1.0), (3, 48, 3.0), (3, 80, 1.5),      # peak on day 2, then falls -> not 'on arrival' -> False
            (4, 0, 1.4),                                  # single value -> False
            (5, 0, 1.0), (5, 6, 1.9), (5, 70, 1.0),       # peak within 24h, falls 1.9x -> True
        ])
        got = confirmed.arrival_fall(w)
        self.assertEqual(got.to_dict(), {1: True, 2: False, 3: False, 4: False, 5: True})

    def test_onset_guard(self):
        f = pd.DataFrame({"kdigo_any": [True, True, True, False],
                          "onset_hours": [3.0, 30.0, np.nan, np.nan]})     # NaN = untimed (coded dialysis)
        self.assertEqual(confirmed.confirmed(f, 24).tolist(), [False, True, False, False])
        self.assertEqual(confirmed.confirmed(f, 0).tolist(), [True, True, True, False])

    def test_dedup_matches_pipeline_without_filter_and_excludes_with_it(self):
        from src.dedup_patient_level import dedup_patient_level
        rng = np.random.default_rng(3)
        rows = []
        for sid in range(300):
            first_pos = int(rng.integers(0, 3)) if rng.random() < 0.4 else None
            for k in range(int(rng.integers(1, 4))):
                lab = 1 if first_pos == k else (0 if first_pos is None or k < first_pos else np.nan)
                rows.append(dict(subject_id=sid, hadm_id=sid * 10 + k, version="mimic4",
                                 admittime=pd.Timestamp("2150-01-01") + pd.Timedelta(days=100 * k), incident_aki_label=lab))
        inc = pd.DataFrame(rows)
        inc = inc[inc["incident_aki_label"].notna()].copy()
        inc["incident_aki_label"] = inc["incident_aki_label"].astype(int)
        f = pd.DataFrame({"confirmed": rng.random(len(inc)) < 0.5, "arrival_fall": rng.random(len(inc)) < 0.3,
                          "onset_hours": 30.0}, index=inc["hadm_id"].to_numpy())
        ref = dedup_patient_level(inc)
        none = confirmed.dedup_confirmed(confirmed.label_status(inc, f, "none", False))
        self.assertEqual(confirmed.check_reproduces(ref, none), 0)
        strict = confirmed.dedup_confirmed(confirmed.label_status(inc, f, "strict", False))
        light = confirmed.dedup_confirmed(confirmed.label_status(inc, f, "light", False))
        pos = strict[strict["incident_aki_label"] == 1]
        self.assertTrue(f.loc[pos["hadm_id"], "confirmed"].all())
        ref_pos = set(ref.loc[ref["incident_aki_label"] == 1, "subject_id"])
        unconfirmed = set(inc.loc[(inc["incident_aki_label"] == 1) & ~f["confirmed"].to_numpy(), "subject_id"])
        self.assertEqual(set(strict["subject_id"]), set(ref["subject_id"]) - unconfirmed)     # patient dropped outright
        self.assertTrue(set(strict["subject_id"]) <= set(light["subject_id"]) <= set(ref["subject_id"]))
        self.assertTrue(set(strict.loc[strict["incident_aki_label"] == 1, "subject_id"]) <= ref_pos)


class TestMimic3AndFlowchart(unittest.TestCase):
    @staticmethod
    def make_mimic3(root: Path) -> None:
        root.mkdir(parents=True)
        t = lambda s: pd.Timestamp(s)  # noqa: E731
        adm = pd.DataFrame({  # hadm, subject, admit, disch
            "SUBJECT_ID": [1, 2, 3, 4, 5], "HADM_ID": [11, 12, 13, 14, 15],
            "ADMITTIME": [t("2150-01-01"), t("2150-01-01"), t("2150-01-01"), t("2150-01-01"), t("2150-01-01")],
            "DISCHTIME": [t("2150-01-10")] * 5})
        pat = pd.DataFrame({"SUBJECT_ID": [1, 2, 3, 4, 5], "GENDER": ["M", "M", "F", "M", "M"],
                            "DOB": [t("2090-01-01"), t("2090-01-01"), t("2090-01-01"), t("2090-01-01"), t("1850-01-01")]})
        lab = lambda sid, hadm, day, v, item=50912: (sid, hadm, item, t("2150-01-01") + pd.Timedelta(days=day), v)  # noqa: E731
        labs = pd.DataFrame([
            lab(1, 11, 1, 2.0), lab(1, 11, 4, 2.0),        # 60 y man, flat 2.0: >= 1.5 x 1.12 -> B3 stage 1
            lab(2, 12, 1, 1.2), lab(2, 12, 4, 1.2),        # 60 y man, flat 1.2: below the B3 trigger
            lab(3, 13, 1, 1.4), lab(3, 13, 4, 1.4),        # 60 y woman, flat 1.4: >= 1.5 x 0.88 -> stage 1
            lab(4, 14, 1, 1.0), lab(4, 14, 1, 80.0, item=51082),   # urine creatinine must be ignored
            lab(5, 15, 1, 1.0), lab(5, 15, 3, 1.0),        # DOB shifted by > 150 y -> age 90
        ], columns=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
        proc = pd.DataFrame({"SUBJECT_ID": [4, 5], "HADM_ID": [14, 15], "SEQ_NUM": [1, 1], "ICD9_CODE": ["3995", "3995"]})
        dx = pd.DataFrame({"SUBJECT_ID": [5], "HADM_ID": [15], "SEQ_NUM": [1], "ICD9_CODE": ["5856"]})   # ESRD: dialysis ignored
        dl = pd.DataFrame({"ITEMID": [50912, 51082], "LABEL": ["Creatinine", "Creatinine, Urine"], "FLUID": ["Blood", "Urine"]})
        for name, df in (("ADMISSIONS", adm), ("PATIENTS", pat), ("LABEVENTS", labs), ("PROCEDURES_ICD", proc),
                         ("DIAGNOSES_ICD", dx), ("D_LABITEMS", dl)):
            df.to_csv(root / f"{name}.csv.gz", index=False)

    def test_mimic3_reader_flags(self):
        from src.concordance import mimic3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "m3"
            self.make_mimic3(root)
            cfg = Config(cache_dir=str(Path(tmp) / "cache"), quiet=True)
            f = mimic3.mimic3_flags(str(root), cfg)
        self.assertEqual(f.loc[11, "scr_B3"], 1)
        self.assertEqual(f.loc[12, "scr_B3"], 0)
        self.assertEqual(f.loc[13, "scr_B3"], 1)
        self.assertEqual(f.loc[14, "n_scr"], 1)                       # urine creatinine row dropped
        self.assertTrue(f.loc[14, "rrt"] and f.loc[14, "kdigo_b3"])   # dialysis code, no ESRD
        self.assertFalse(f.loc[15, "rrt"])                            # ESRD-coded: maintenance dialysis is not AKI
        self.assertFalse(f.loc[15, "kdigo_b3"])
        self.assertEqual(f.loc[11, "scr_B2"], 0)                      # B2 has no prior value in the window for a flat series

    def test_generic_flags_equal_the_admission_table(self):
        from src.concordance import data, mimic3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_synthetic_mimic(root / "mimic", n_patients=400)
            cfg = Config(mimic_dir=str(root / "mimic"), cache_dir=str(root / "cache"), use_uo=False, quiet=True)
            tab = data.load_tables(cfg)
            adm, _, _ = data.build_admission_table(tab, cfg)
            gen = mimic3.hospital_flags(adm[["hadm_id", "subject_id", "admittime", "dischtime", "age", "female"]],
                                        tab.labs_raw, set(adm.index[adm["rrt_HOSP"]]), set(adm.index[adm["ESRD"]]))
        pd.testing.assert_series_equal(gen["scr_B3"], adm.loc[gen.index, "scr_HOSP_B3"], check_names=False)
        pd.testing.assert_series_equal(gen["scr_B2"], adm.loc[gen.index, "scr_HOSP_B2"], check_names=False)

    def test_pre_admission_window(self):
        from src.concordance import mimic3
        d = lambda s: pd.Timestamp(s)  # noqa: E731
        adm = pd.DataFrame({"hadm_id": [1, 2], "subject_id": [10, 20], "admittime": [d("2150-01-05")] * 2,
                            "dischtime": [d("2150-01-12")] * 2, "age": [60.0, 60.0], "female": [False, False]})
        labs = pd.DataFrame({  # patient 10: ED value 12 h before admission is high, ward values normal; patient 20: ED value 3 days before
            "subject_id": [10, 10, 10, 20, 20], "hadm_id": [np.nan, 1, 1, np.nan, 2],
            "charttime": [d("2150-01-04 12:00"), d("2150-01-06"), d("2150-01-08"), d("2150-01-02"), d("2150-01-06")],
            "valuenum": [2.4, 1.0, 1.0, 2.4, 1.0]})
        f0 = mimic3.hospital_flags(adm, labs, set(), set())
        f24 = mimic3.hospital_flags(adm, labs, set(), set(), pre_hours=24)
        self.assertFalse(f0.loc[1, "kdigo_b3"]); self.assertTrue(f24.loc[1, "kdigo_b3"])    # ED value inside 24 h counts
        self.assertFalse(f24.loc[2, "kdigo_b3"])                                           # 3 days before does not
        self.assertEqual((f0["n_scr"].tolist(), f24["n_scr"].tolist()), ([2, 1], [3, 1]))

    def test_nadir_baseline(self):
        from src.concordance import mimic3
        d = lambda s: pd.Timestamp(s)  # noqa: E731
        adm = pd.DataFrame({"hadm_id": [1], "subject_id": [10], "admittime": [d("2150-01-05")], "dischtime": [d("2150-01-20")],
                            "age": [60.0], "female": [False]})
        # 60 y man: B3 baseline 1.12, trigger 1.68. Creatinine 1.5 (arrival) falling to 0.9: no B3 stage, but 1.5 >= 1.5 x 0.9
        labs = pd.DataFrame({"subject_id": [10, 10, 10], "hadm_id": [1, 1, 1],
                             "charttime": [d("2150-01-05 08:00"), d("2150-01-12"), d("2150-01-18")], "valuenum": [1.5, 1.0, 0.9]})
        f = mimic3.hospital_flags(adm, labs, set(), set())
        self.assertFalse(f.loc[1, "kdigo_b3"])
        self.assertTrue(f.loc[1, "kdigo_b3n"])
        self.assertTrue((f["kdigo_b3n"] | ~f["kdigo_b3"]).all())          # the nadir variant never removes a B3 case
        self.assertAlmostEqual(f.loc[1, "onset_h_B3N"], 8.0)              # first qualifying value: 08:00 on the admission day
        self.assertTrue(np.isnan(f.loc[1, "onset_h_B3"]))

    def test_final_label_frame(self):
        from src.concordance import final_label
        f24 = pd.DataFrame({"scr_B3": [0, 1, 2, 0, 1, 0], "rrt": [False, False, False, True, False, False],
                            "onset_h_B3": [np.nan, -5.0, 30.0, np.nan, 10.0, np.nan], "n_scr": [3, 4, 2, 1, 5, 0]},
                           index=[11, 12, 13, 14, 15, 16])
        cohort = pd.DataFrame({"hadm_id": [11, 12, 13, 14, 15, 16], "icd": [True, True, False, True, False, False],
                               "female": [True, False, True, False, True, False]}, index=[1, 2, 3, 4, 5, 6])
        out = final_label.label_frame("v", {24: f24}, cohort)
        self.assertEqual(out["kdigo_stage"].tolist(), [0, 1, 2, 3, 1, 0])              # dialysis counts as stage 3
        self.assertEqual(out["kdigo_final"].tolist(), [False, True, True, True, True, False])
        self.assertEqual(out["kdigo_incident_24h"].tolist(), [False, False, True, False, False, False])
        self.assertEqual(out["concordance"].tolist(), ["ICD only", "both", "KDIGO only", "both", "KDIGO only", "neither"])
        self.assertEqual(out["onset_group"].tolist(), [final_label.GROUPS[5], final_label.GROUPS[0], final_label.GROUPS[2],
                                                       final_label.GROUPS[4], final_label.GROUPS[1], final_label.GROUPS[5]])
        s = final_label.summary(out, Suppressor(1))
        self.assertEqual(s[(s["section"] == "stage")]["n"].astype(int).sum(), 6)

    def test_model_inputs_swap(self):
        from src.concordance import model_inputs as mi
        labels = pd.DataFrame({"version": ["mimic4"] * 5 + ["mimic3"], "hadm_id": [1, 2, 3, 4, 5, 6],
                               "icd_incident_label": [1, 1, 0, 0, 1, 0],
                               "kdigo_final": [True, True, True, False, False, False],
                               "kdigo_incident_24h": [True, False, False, False, False, False]})
        feat = pd.DataFrame({"version": ["mimic4"] * 6 + ["mimic3"] * 2, "hadm_id": [1, 2, 3, 4, 5, 99, 6, 7],
                             "x": np.arange(8.0), "incident_aki_label": [1, 1, 0, 0, 1, 0, 0, 0]})
        out = mi.swap_label(feat, labels, "KDIGO in-hospital onset")
        # kept: 1 (positive), 4, 5, 6 (KDIGO-negative); left out: 2, 3 (early-onset KDIGO), 99 and 7 (not in the cohort)
        self.assertEqual(out["hadm_id"].tolist(), [1, 4, 5, 6])
        self.assertEqual(out["incident_aki_label"].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(out.columns), list(feat.columns))
        same = mi.swap_label(feat, labels, "ICD incident")
        self.assertEqual(same["incident_aki_label"].tolist(), [1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(mi.slug("KDIGO in-hospital onset"), "kdigo_in_hospital_onset")

    def test_label_sensitivity_helpers(self):
        from src.concordance import label_sensitivity as ls
        d = pd.DataFrame({"icd_incident_label": [1, 1, 1, 0, 0, 0],
                          "kdigo_final": [True, True, False, True, True, False],
                          "kdigo_incident_24h": [True, False, False, True, False, False]})
        L = ls.make_labels(d)
        nan = np.nan
        self.assertEqual(L["ICD incident"].tolist(), [1, 1, 1, 0, 0, 0])
        self.assertEqual(L["KDIGO final"].tolist(), [1, 1, 0, 1, 1, 0])
        self.assertTrue(np.array_equal(L["KDIGO in-hospital onset"].to_numpy(), [1, nan, 0, 1, nan, 0], equal_nan=True))
        self.assertTrue(np.array_equal(L["ICD and KDIGO agree"].to_numpy(), [1, 1, nan, nan, nan, 0], equal_nan=True))
        self.assertTrue(np.array_equal(L["ICD, arrival-onset removed"].to_numpy(), [1, nan, 1, 0, 0, 0], equal_nan=True))
        # crude odds ratio from the logistic helper equals the 2x2 odds ratio
        rng = np.random.default_rng(0)
        male = rng.integers(0, 2, 4000).astype(float)
        y = (rng.random(4000) < np.where(male == 1, 0.30, 0.20)).astype(int)
        a, b = int(((male == 1) & (y == 1)).sum()), int(((male == 1) & (y == 0)).sum())
        c, e = int(((male == 0) & (y == 1)).sum()), int(((male == 0) & (y == 0)).sum())
        orr, lo, hi = ls.logit_or(y, male[:, None], 0)
        self.assertAlmostEqual(orr, (a * e) / (b * c), places=3)
        self.assertTrue(lo < orr < hi)
        s1, s2 = ls.split_subjects(np.arange(100), 5), ls.split_subjects(np.arange(100), 5)
        self.assertTrue((s1 == s2).all() and s1.sum() == 20)

    def test_nadir_overlap_helpers(self):
        from src.concordance import nadir
        f = pd.DataFrame({"n_scr": [0, 1, 4, 4], "scr_max": [np.nan, 1.0, 1.3, 0.9], "scr_min": [np.nan, 1.0, 1.0, 0.8],
                          "rise48": [np.nan, 0.0, 0.1, 0.0], "b3_base": [1.1] * 4}, index=[1, 2, 3, 4])
        r = nadir.icd_only_reason(f).tolist()
        self.assertEqual(r[0], nadir.ICD_ONLY_REASONS[0])
        self.assertEqual(r[1], nadir.ICD_ONLY_REASONS[1])
        self.assertEqual(r[2], nadir.ICD_ONLY_REASONS[2])          # baseline 1.0, trigger 1.5, peak 1.3 = 87% of it
        self.assertEqual(r[3], nadir.ICD_ONLY_REASONS[3])          # baseline 0.8, trigger 1.2, peak 0.9 = 75% of it
        rb = nadir.icd_only_reason(f, nadir=False).tolist()         # B3 baseline 1.1 (trigger 1.65): peaks of 1.3 and 0.9 are 79% / 55%
        self.assertEqual((rb[2], rb[3]), (nadir.ICD_ONLY_REASONS[3], nadir.ICD_ONLY_REASONS[3]))
        cfg = Config(n_boot=20, seed=1, quiet=True)
        icd = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
        kd = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], bool)
        row = nadir.overlap_row(Suppressor(1), cfg, "v", "x", "all", icd, kd)
        self.assertEqual((row["both"], row["icd_only"], row["kdigo_only"], row["neither"]), ("2", "1", "1", "6"))
        self.assertEqual(row["pct_icd_pos_also_kdigo"], "66.67")           # 2 of 3 ICD-positive
        self.assertEqual(row["pct_kdigo_pos_also_icd"], "66.67")
        self.assertEqual(row["accuracy"], "80.00")
        small = nadir.overlap_row(Suppressor(10), cfg, "v", "x", "all", icd, kd)
        self.assertEqual(small["kappa"], "suppressed")

    def test_naki_only_table_changes_only_row_7(self):
        from src.concordance import flowchart
        from src.concordance.stats import Suppressor
        res = {"mimic4": {"b3n": {"aki": 21907}}, "mimic3": {"b3n": {"aki": 6489}}}
        t = flowchart.naki_only_table(res, "b3n", Suppressor(10))
        ref = pd.DataFrame(flowchart.REFERENCE, columns=["n", "box", "s4", "c4", "s3", "c3", "act"])
        self.assertEqual(t.loc[t["#"] == "7", "MIMIC-IV correct"].iloc[0], "21907")
        self.assertEqual(t.loc[t["#"] == "7", "MIMIC-III correct"].iloc[0], "6489")
        others = t[t["#"] != "7"]
        self.assertEqual(others["MIMIC-IV correct"].tolist(), [str(v) for v in ref[ref["n"] != 7]["c4"]])
        self.assertEqual(others["MIMIC-III correct"].tolist(), [str(v) for v in ref[ref["n"] != 7]["c3"]])
        self.assertEqual(others["Action"].tolist(), ref[ref["n"] != 7]["act"].tolist())

    def test_survival_exclusions_and_block(self):
        from src.concordance import flowchart
        d = lambda s: pd.Timestamp(s)  # noqa: E731
        rows = []

        def add(sid, adm_day, disch_day, aki=0, ckd=0, post=0, death=0):
            rows.append(dict(version="mimic4", subject_id=sid, admittime=d("2150-01-01") + pd.Timedelta(days=adm_day),
                             dischtime=d("2150-01-01") + pd.Timedelta(days=disch_day), is_aki=aki, event_ckd_flag=ckd,
                             event_postckd_flag=post, event_death_flag=death))
        add(1, 0, 5, aki=1); add(1, 200, 205, ckd=1)                   # CKD at 200 d: kept
        add(2, 0, 5, aki=1); add(2, 30, 35, ckd=1)                     # CKD at 30 d: excluded (ckd)
        add(3, 0, 0, aki=1)                                            # censored at 0 d: excluded (no follow-up)
        add(4, 0, 5, aki=1); add(4, 50, 60, death=1)                   # death at 60 d: kept
        add(5, 0, 5, aki=1)                                            # censored at 5 d: kept
        pre = flowchart.survival_pre_exclusion(pd.DataFrame(rows))
        self.assertEqual(dict(zip(pre["subject_id"], pre["reason"])), {1: "", 2: "ckd", 3: "nofu", 4: "", 5: ""})
        surv = pd.DataFrame({"subject_id": [1, 4, 5], "event_type": ["CKD", "Death", "Censor"]})
        lab = pd.DataFrame({"subject_id": [5], "event_type": ["LabCKD"]})
        aki = pd.DataFrame({"subject_id": [1, 2, 3, 4, 5], "is_autocar": [0, 1, 0, 0, 1]})
        b = flowchart.block(aki, 100, surv, lab, pre)
        self.assertEqual((b["aki"], b["nonauto"], b["excl_total"], b["excl_ckd"], b["excl_nofu"], b["post"]), (5, 3, 2, 1, 1, 3))
        self.assertEqual((b["ckd"], b["death"], b["censor"], b["labckd"]), (1, 1, 1, 1))
        b3 = flowchart.block(aki[aki["subject_id"].isin([1, 2, 5])], 100, surv, lab, pre)      # a B3-confirmed subset
        self.assertEqual((b3["aki"], b3["excl_total"], b3["excl_ckd"], b3["post"], b3["labckd"]), (3, 1, 1, 2, 1))


class TestConfirmedEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        make_synthetic_mimic(root / "mimic", n_patients=900)
        from src.dedup_patient_level import dedup_patient_level
        mimic = root / "mimic"
        dx = pd.read_csv(mimic / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
        aki = set(dx[dx["icd_code"].isin(["5849", "N179"])]["hadm_id"])
        adm = pd.read_csv(mimic / "hosp/admissions.csv.gz", parse_dates=["admittime"]).sort_values(["subject_id", "admittime"])
        first = adm[adm["hadm_id"].isin(aki)].groupby("subject_id").head(1)["hadm_id"]
        inc = adm[["subject_id", "hadm_id", "admittime"]].copy()
        inc["version"] = "mimic4"
        inc["incident_aki_label"] = np.where(inc["hadm_id"].isin(first), 1.0, np.where(inc["hadm_id"].isin(aki), np.nan, 0.0))
        inc.to_parquet(root / "incident.parquet")
        dedup_patient_level(inc).to_parquet(root / "target_syn.parquet")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_confirmed_target_and_overlap(self):
        r = self.root
        cfg = Config(mimic_dir=str(r / "mimic"), outdir=str(r / "out"), cache_dir=str(r / "cache"), min_cell=1, n_boot=50,
                     seed=5, pipeline_target=None, modeling_target=str(r / "target_syn.parquet"), quiet=True,
                     write_confirmed=True, incident_target=str(r / "incident.parquet"),
                     confirmed_out=str(r / "conf.parquet"))
        run(cfg)                                         # asserts internally that the unfiltered rebuild equals target_syn
        base = pd.read_parquet(r / "target_syn.parquet")
        strict = pd.read_parquet(r / "conf.parquet")
        light = pd.read_parquet(r / "conf_light.parquet")
        self.assertTrue(set(strict["subject_id"]) <= set(light["subject_id"]) <= set(base["subject_id"]))
        self.assertTrue(strict.loc[strict["incident_aki_label"] == 1, "kdigo_confirmed"].all())
        self.assertGreater((base["incident_aki_label"] == 1).sum(), (strict["incident_aki_label"] == 1).sum())
        self.assertEqual(int((strict["incident_aki_label"] == 0).sum()) + int((strict["incident_aki_label"] == 1).sum()), len(strict))
        out = r / "out"
        for want in ("d6_target_overlap.csv", "d6_three_way.csv", "d6_confirmation_funnel.csv"):
            self.assertTrue((out / want).exists(), want)
        checks.verify_outputs(str(out))
        ov = pd.read_csv(out / "d6_target_overlap.csv", dtype=str)
        cnt = lambda x: 0 if str(x).startswith("<") else int(x)  # noqa: E731
        for _, row in ov.iterrows():
            self.assertEqual(cnt(row["both"]) + cnt(row["icd_only"]) + cnt(row["kdigo_only"]) + cnt(row["neither"]), cnt(row["n"]))
        al = ov[ov["stratum"] == "all"]
        self.assertEqual(cnt(al["n"].iloc[0]), len(base))
        self.assertEqual(cnt(al["n_icd_pos"].iloc[0]), int((base["incident_aki_label"] == 1).sum()))
        sexes = ov[ov["kdigo_target"] == al["kdigo_target"].iloc[0]]
        self.assertEqual(sum(cnt(x) for x in sexes[sexes["stratum"] != "all"]["n"]), cnt(al["n"].iloc[0]))
        tw = pd.read_csv(out / "d6_three_way.csv", dtype=str)
        self.assertEqual(sum(cnt(x) for x in tw[tw["stratum"] == "all"]["n"]), len(base))
        self.assertIn("## D6.", (out / "summary.md").read_text())

        # sex strata exist in D5 for both windows, and the confirmed target does not depend on the primary baseline
        d5 = pd.read_csv(out / "d5_stratified.csv", dtype=str)
        self.assertEqual(set(d5[d5["stratum"].str.startswith("sex=")]["window"]), {"W_ICU", "W_HOSP"})
        self.assertEqual(set(d5["baseline"]), {cfg.primary_baseline})
        cfg2 = Config(**{**cfg.__dict__, "outdir": str(r / "out_b2"), "primary_baseline": "B2" if cfg.primary_baseline != "B2" else "B3",
                         "confirmed_out": str(r / "conf2.parquet")})
        run(cfg2)
        for name in ("d6_target_overlap.csv", "d6_three_way.csv", "d6_confirmation_funnel.csv"):
            self.assertEqual((out / name).read_bytes(), (r / "out_b2" / name).read_bytes(), name)
        pd.testing.assert_frame_equal(strict, pd.read_parquet(r / "conf2.parquet"))


if __name__ == "__main__":
    unittest.main()
