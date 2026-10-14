import glob, os
import numpy as np, pandas as pd, pydicom
from multiprocessing import Pool

R = "/kaggle/input/competitions/rsna-knee-abnormality-detection"


def header(path):
    files = glob.glob(os.path.join(path, "*.dcm"))
    hs = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    h = hs[0]
    out = {"n": len(hs), "lat": str(getattr(h, "Laterality", "")), "pos": str(getattr(h, "PatientPosition", "")),
           "rows": int(h.Rows), "cols": int(h.Columns), "desc": str(getattr(h, "SeriesDescription", "")),
           "thick": str(getattr(h, "SliceThickness", "")), "man": str(getattr(h, "Manufacturer", "")),
           "part": str(getattr(h, "BodyPartExamined", "")), "pid": str(getattr(h, "PatientID", "")),
           "sex": str(getattr(h, "PatientSex", "")), "tags": len(h)}
    try:
        iop = np.array(h.ImageOrientationPatient, float)
        ps = np.array(h.PixelSpacing, float)
        ipp = np.array([np.array(x.ImagePositionPatient, float) for x in hs])
        r, c = iop[:3], iop[3:]
        ctr = ipp + r * ps[1] * (h.Columns - 1) / 2 + c * ps[0] * (h.Rows - 1) / 2
        m = ctr.mean(0)
        out.update({"iop": list(iop), "ps0": ps[0], "ps1": ps[1], "cx": m[0], "cy": m[1], "cz": m[2],
                    "sx": ipp[:, 0].min(), "ex": ipp[:, 0].max()})
    except Exception as e:
        out["err"] = str(e)[:60]
    return out


def work(a):
    try:
        return header(f"{R}/train_series/{a[0]}/{a[1]}")
    except Exception as e:
        return {"err": str(e)[:60]}


if __name__ == "__main__":
    ts = pd.read_csv(f"{R}/train_series.csv")
    with Pool(4) as p:
        res = p.map(work, list(zip(ts.StudyInstanceUID, ts.SeriesInstanceUID)), chunksize=16)
    g = pd.concat([ts, pd.DataFrame(res)], axis=1)
    g["iop"] = g["iop"].astype(str)
    g.to_csv("geom.csv", index=False)
    print(g.shape, g.err.notna().sum() if "err" in g else 0)
    print(g.lat.value_counts(dropna=False).head(8).to_dict())
    print(g.part.value_counts().head(5).to_dict(), g.pos.value_counts().head(5).to_dict())
    tg = g[g.lat.isin(["R", "L"])]
    print("series with tag", len(tg), "of", len(g), " studies with any tag", tg.StudyInstanceUID.nunique(), "of", g.StudyInstanceUID.nunique())
    from sklearn.metrics import roc_auc_score
    y = (tg.lat == "L").astype(int)
    for c in ["cx", "cy", "cz"]:
        print(c, "auc vs L", round(roc_auc_score(y, tg[c].fillna(0)), 3), tg.groupby("lat")[c].describe()[["mean", "std", "min", "max"]].round(1).to_dict("index"))
    for man, d in tg.groupby(tg.man.str.upper().str[:4]):
        if len(d) > 100 and d.lat.nunique() > 1:
            print(man, len(d), "auc cx", round(roc_auc_score((d.lat == "L").astype(int), d.cx.fillna(0)), 3))
    # tag consistency within study
    cons = tg.groupby("StudyInstanceUID").lat.nunique()
    print("studies with conflicting tags", (cons > 1).sum())
    print("patients with >1 study:", (g.groupby("pid").StudyInstanceUID.nunique() > 1).sum(), "of", g.pid.nunique())
