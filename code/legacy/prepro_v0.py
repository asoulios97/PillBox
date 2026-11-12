# ------------------------------ imports ------------------------------
import numpy as np               # αριθμητικοί πίνακες/διανύσματα
import pandas as pd              # ανάγνωση CSV + επεξεργασία πινάκων
from pathlib import Path         # ασφαλής χειρισμός διαδρομών (αντί για raw strings)
from collections import Counter  # γρήγορη σύνοψη κατανομής labels
import json                      # για αποθήκευση meta πληροφοριών σε κείμενο

# ------------------------------ σταθερές ------------------------------
RAW_FS = 64          # αρχική συχνότητα δειγματοληψίας (Hz) στο dataset (Right/Left). Βλέπε ReadMe.  :contentReference[oaicite:2]{index=2}
TARGET_FS = 16       # στόχος downsample (Hz), όπως προτείνεται στη βιβλιογραφία των Wang (συνήθης επιλογή)
DS_FACTOR = RAW_FS // TARGET_FS  # 64/16 = 4 → παίρνουμε κάθε 4ο δείγμα για απλό, σταθερό downsampling

# στήλες που περιμένουμε να υπάρχουν στο Right-hand CSV (όπως ορίζει το ReadMe του dataset)
EXPECTED_COLS = ["t","ax","ay","az","gx","gy","gz","anno"]  # 8 στήλες: χρόνος, 6 κανάλια IMU, label 0/1  :contentReference[oaicite:3]{index=3}

# ------------------------------ βοηθητικά ------------------------------
def load_right_csv(csv_path: Path) -> pd.DataFrame:
    """
    Διαβάζει ένα δεξί χέρι CSV και κάνει βασικούς ελέγχους ότι είναι του σωστού format.
    Επιστρέφει DataFrame με τις στήλες [t, ax, ay, az, gx, gy, gz, anno].
    """
    # ανάγνωση CSV σε DataFrame (pandas αντέχει μεγάλα αρχεία/δεκαδικά)
    df = pd.read_csv(csv_path)

    # βεβαιωνόμαστε ότι οι στήλες είναι αυτές που δηλώνει το ReadMe
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: λείπουν στήλες {missing} (αναμενόμενες: {EXPECTED_COLS})")

    # γρήγορο sort by time αν χρειάζεται (μερικές φορές χρήσιμο αν έχει γίνει merge upstream)
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)

    # τύποι δεδομένων: μετατρέπουμε τα IMU σε float32 (λιγότερη μνήμη, αρκετή ακρίβεια)
    imu_cols = ["ax","ay","az","gx","gy","gz"]
    df[imu_cols] = df[imu_cols].astype(np.float32)

    # label ως int8 (0/1)
    df["anno"] = df["anno"].astype(np.int8)

    return df

def summarize_labels(y: np.ndarray, title: str = ""):
    """
    Εκτυπώνει κατανομή labels (0/1) για γρήγορο sanity check.
    """
    cnt = Counter(y.tolist())
    total = int(y.size)
    pos = int(cnt.get(1, 0))
    neg = int(cnt.get(0, 0))
    ratio = f"{pos}/{total} = {pos/total:.4%}" if total > 0 else "N/A"
    print(f"[labels] {title} total={total} 0={neg} 1={pos} (positives={ratio})")

def downsample_64_to_16(df: pd.DataFrame, label_reduce: str = "any") -> pd.DataFrame:
    """
    Downsample από 64→16 Hz με απλό indexing ανά 4 δείγματα.
    Για το label:
        - 'any': αν σε οποιοδήποτε από τα 4 αρχικά δείγματα υπήρχε 1, το νέο δείγμα γίνεται 1 (ασφαλές για σπάνια events).
        - 'majority': 1 μόνο αν >=2/4 είναι 1 (πιο αυστηρό).
    """
    if label_reduce not in {"any","majority"}:
        raise ValueError("label_reduce must be 'any' or 'majority'")

    # θα δουλέψουμε σε αντίγραφο για καθαρότητα
    df = df.copy()

    # ευθεία downsample με slicing: κάθε 4ο δείγμα.
    # προϋπόθεση: σχεδόν ομοιόμορφο sampling 64Hz (όπως δηλώνεται από το dataset).  :contentReference[oaicite:4]{index=4}
    idx = np.arange(0, len(df), DS_FACTOR)

    # για τα labels θέλουμε μείωση σε blocks των 4
    # κόβουμε πλήρη blocks, υπολογίζουμε label per block, και προσέχουμε την ουρά
    y = df["anno"].to_numpy()
    n_full = (len(y) // DS_FACTOR) * DS_FACTOR
    y_full = y[:n_full].reshape(-1, DS_FACTOR)  # σχήμα: [#blocks, 4]

    if label_reduce == "any":
        y_ds_core = (y_full.max(axis=1)).astype(np.int8)
    else:
        y_ds_core = (y_full.mean(axis=1) >= 0.5).astype(np.int8)

    # tail (αν δεν διαιρείται ακριβώς)
    tail = y[n_full:]
    if tail.size > 0:
        y_tail = (tail.max() if label_reduce == "any" else (tail.mean() >= 0.5)).astype(np.int8)
        y_ds = np.concatenate([y_ds_core, [y_tail]]).astype(np.int8)
    else:
        y_ds = y_ds_core

    # κρατάμε τα downsampled rows με iloc (ίδια indices με idx)
    df_ds = df.iloc[idx].reset_index(drop=True)

    # αντικαθιστούμε το label με την “συρρικνωμένη” εκδοχή (ANY ή MAJ)
    # προσοχή: αν το τελευταίο βήμα indexing έδωσε 1 παραπάνω στοιχείο από το block-tail,
    # συγχρονίζουμε εύκολα ταιριάζοντας μήκη.
    if len(df_ds) != len(y_ds):
        # αν ο τελευταίος δείκτης idx έπεσε εκτός y_ds length, κόψε το df_ds στην ίδια διάρκεια
        min_len = min(len(df_ds), len(y_ds))
        df_ds = df_ds.iloc[:min_len].reset_index(drop=True)
        y_ds = y_ds[:min_len]

    df_ds["anno"] = y_ds
    return df_ds

def add_magnitudes(df: pd.DataFrame, add_acc=True, add_gyro=True) -> pd.DataFrame:
    """
    Προσθέτει προαιρετικά κανάλια magnitude για accel/gyro.
    """
    df = df.copy()
    if add_acc:
        df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    if add_gyro:
        df["gyro_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df

def zscore_normalize(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, dict]:
    """
    Z-score per session: (x - mean)/std πάνω στα δεδομένα του ΙΔΙΟΥ του session.
    Επιστρέφει (normalized_df, stats) όπου stats={'mean':..., 'std':..., 'cols':...}.
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0  # avoid divide-by-zero

    Xn = (X - mean) / std
    out = df.copy()
    out[feature_cols] = Xn

    stats = {"cols": feature_cols, "mean": mean.tolist(), "std": std.tolist()}
    return out, stats

def save_npz(output_path: Path, X: np.ndarray, y: np.ndarray, meta: dict):
    """
    Αποθηκεύει σε .npz (συμπιεσμένο): features, labels και μετα-πληροφορία ως JSON string.
    """
    # meta ως JSON για να είναι αναγνώσιμο με απλό τρόπο
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    np.savez_compressed(output_path, X=X, y=y, meta=np.array([meta_json], dtype=object))
    print(f"[save] wrote {output_path}  X{X.shape}  y{y.shape}")

# ------------------------------ κύρια ρουτίνα ------------------------------
def preprocess_right_csv(
    csv_path: str | Path,
    target_fs: int = TARGET_FS,
    add_mags: bool = True,
    label_reduce: str = "any",
    out_npz: str | Path | None = None,
):
    """
    End-to-end preprocessing ενός Right-hand CSV:
      1) load & validate
      2) quick label summary
      3) downsample 64→16 Hz με label reduction
      4) (optional) προσθήκη magnitude καναλιών
      5) z-score per session (χωρίς data leakage)
      6) αποθήκευση σε .npz (X, y, meta)
    """
    csv_path = Path(csv_path)

    # 1) φόρτωση & έλεγχος
    df_raw = load_right_csv(csv_path)

    # 2) σύνοψη αρχικών labels
    summarize_labels(df_raw["anno"].to_numpy(), title=f"{csv_path.name} @64Hz")

    # 3) downsample 64→16Hz (ή ό,τι έχεις ορίσει ως TARGET_FS)
    if target_fs != RAW_FS:
        # μείωση με σταθερό factor (εδώ 4)
        df = downsample_64_to_16(df_raw, label_reduce=label_reduce)
    else:
        df = df_raw.copy()

    summarize_labels(df["anno"].to_numpy(), title=f"{csv_path.name} @{target_fs}Hz")

    # 4) προαιρετικά magnitudes (καλό baseline για CNN/TCN)
    if add_mags:
        df = add_magnitudes(df, add_acc=True, add_gyro=True)

    # 5) z-score per session (fit & transform στο ίδιο session)
    #    για Right-only πείραμα, αυτό είναι ασφαλές· σε LOSO θα κάνεις fit στα train subjects
    feature_cols = ["ax","ay","az","gx","gy","gz"] + (["acc_mag","gyro_mag"] if add_mags else [])
    df_norm, stats = zscore_normalize(df, feature_cols)

    # 6) ετοιμάζουμε numpy arrays για το μοντέλο
    X = df_norm[feature_cols].to_numpy(dtype=np.float32)  # σχήμα [N, C]
    y = df_norm["anno"].to_numpy(dtype=np.int8)           # σχήμα [N]

    # μετα-πληροφορία για αναπαραγωγιμότητα
    meta = {
        "source_csv": csv_path.name,
        "right_only": True,
        "raw_fs_hz": RAW_FS,
        "target_fs_hz": target_fs,
        "label_reduce": label_reduce,
        "features": feature_cols,
        "zscore_stats": stats,           # mean/std ανά κανάλι αυτού του session
        "columns_expected": EXPECTED_COLS,
        "note": "Right-hand raw CSV preprocessed; timestamps removed as per dataset pkl convention.",
    }

    # 7) αποθήκευση (προαιρετική — αν δοθεί out_npz)
    if out_npz is not None:
        out_npz = Path(out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        save_npz(out_npz, X, y, meta)

    # επιστρέφουμε και στη μνήμη για άμεση χρήση/δοκιμές
    return X, y, meta

# ------------------------------ παράδειγμα χρήσης ------------------------------
if __name__ == "__main__":
    # βάλ’ το δικό σου path εδώ (π.χ. το '4-1.csv' που ανέβασες)
    in_csv = Path("4-1.csv")  # ή Path("/mnt/data/4-1.csv") αν τρέχεις στο ίδιο περιβάλλον

    # θα γράψουμε το αποτέλεσμα σε npz πλάι στο input
    out_npz = in_csv.with_suffix(".right16hz.npz")

    # τρέξε το pipeline
    X, y, meta = preprocess_right_csv(
        csv_path=in_csv,
        target_fs=16,          # 64→16 Hz
        add_mags=True,         # πρόσθεσε acc_mag & gyro_mag
        label_reduce="any",    # ασφαλές για διατήρηση positives
        out_npz=out_npz,
    )

    # γρήγορη επιθεώρηση σχήματος
    print("X shape:", X.shape, " y shape:", y.shape)
    print("feature order:", meta["features"])
