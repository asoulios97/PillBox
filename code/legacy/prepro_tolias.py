import numpy as np                     # αριθμητικοί πίνακες
import pandas as pd                    # ανάγνωση CSV & dataframes
from pathlib import Path               # ασφαλής χειρισμός διαδρομών
from collections import Counter        # μέτρηση συχνοτήτων labels
import json                            # αποθήκευση meta πληροφοριών
from typing import List, Dict, Tuple, Any

from scipy.signal import butter, filtfilt  # για anti-aliasing low-pass

# --------------------- Βασικές σταθερές ---------------------

RAW_FS = 64                 # συχνότητα δειγματοληψίας των raw/Hand-Mirrored αρχείων (Hz)
TARGET_FS = 16              # στόχος downsampling (Hz), όπως στο paper (64 -> 16)
DS_FACTOR = RAW_FS // TARGET_FS   # 64/16 = 4 → παίρνουμε κάθε 4ο δείγμα

# Οι στήλες που περιμένουμε σύμφωνα με το ReadMe
EXPECTED_COLS = ["t","ax","ay","az","gx","gy","gz","anno"]


# --------------------- Φόρτωση Hand-Mirrored CSV ---------------------

def load_handmirrored_csv(csv_path: Path) -> pd.DataFrame:
    """
    Διαβάζει ένα Hand-Mirrored αρχείο (.csv) και κάνει βασικούς ελέγχους:
    - Στήλες [t, ax, ay, az, gx, gy, gz, anno]
    - Ταξινόμηση ως προς t
    - Τύποι δεδομένων: float32 για IMU, int8 για labels
    """
    df = pd.read_csv(csv_path)  # διαβάζει όλο το CSV σε DataFrame

    # ελέγχει αν υπάρχουν όλες οι αναμενόμενες στήλες
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: λείπουν στήλες {missing}")

    # ταξινόμηση κατά χρόνο (t) για ασφάλεια
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)

    # μετατροπή IMU στηλών σε float32 (λιγότερη μνήμη, αρκετή ακρίβεια)
    imu_cols = ["ax","ay","az","gx","gy","gz"]
    df[imu_cols] = df[imu_cols].astype(np.float32)

    # labels 0/1 ως int8
    df["anno"] = df["anno"].astype(np.int8)

    return df


# --------------------- Σύνοψη labels ---------------------

def summarize_labels(y: np.ndarray, title: str = ""):
    """
    Εκτυπώνει κατανομή 0/1 για γρήγορο έλεγχο ανισορροπίας.
    """
    cnt = Counter(y.tolist())
    total = int(y.size)
    pos = int(cnt.get(1, 0))
    neg = int(cnt.get(0, 0))
    ratio = f"{pos}/{total} = {pos/total:.4%}" if total > 0 else "N/A"
    print(f"[labels] {title} total={total} 0={neg} 1={pos} (positives={ratio})")


# --------------------- Anti-aliasing low-pass ---------------------

def apply_antialias_lowpass(
    df: pd.DataFrame,
    fs: float = RAW_FS,
    cutoff: float = 7.5,
    order: int = 4
) -> pd.DataFrame:
    """
    Εφαρμόζει Butterworth low-pass φίλτρο σε όλες τις IMU στήλες πριν το downsampling.
    - fs: συχνότητα δειγματοληψίας εισόδου (64 Hz)
    - cutoff: συχνότητα αποκοπής (π.χ. 7.5 Hz < Nyquist του 16 Hz)
    - order: τάξη φίλτρου (4 → αρκετά ομαλό χωρίς τεράστιο phase distortion με filtfilt)
    """
    df = df.copy()
    imu_cols = ["ax","ay","az","gx","gy","gz"]

    nyq = 0.5 * fs                        # Nyquist συχνότητα
    norm_cutoff = cutoff / nyq            # κανονικοποιημένο cutoff [0,1]
    b, a = butter(order, norm_cutoff, btype='low', analog=False)

    # φιλτράρουμε κάθε κανάλι με filtfilt (zero-phase)
    for c in imu_cols:
        df[c] = filtfilt(b, a, df[c].to_numpy())

    return df


# --------------------- Downsampling 64 -> 16 Hz ---------------------

def downsample_df(df: pd.DataFrame, label_reduce: str = "any") -> pd.DataFrame:
    """
    Downsample από 64 Hz σε 16 Hz με factor 4.
    - Σήμα: παίρνουμε κάθε 4ο δείγμα (0,4,8,...)
    - Label:
        'any'      : αν σε οποιοδήποτε από τα 4 αρχικά δείγματα υπάρχει 1, η νέα ετικέτα είναι 1
        'majority' : 1 μόνο αν τουλάχιστον 2/4 είναι 1
    """
    if label_reduce not in {"any","majority"}:
        raise ValueError("label_reduce must be 'any' or 'majority'")

    df = df.copy()
    y = df["anno"].to_numpy()

    # κόβουμε σε πλήρη blocks των 4 δειγμάτων
    n_full = (len(y) // DS_FACTOR) * DS_FACTOR
    y_full = y[:n_full].reshape(-1, DS_FACTOR)  # shape: [num_blocks, 4]

    if label_reduce == "any":
        y_core = (y_full.max(axis=1)).astype(np.int8)
    else:
        y_core = (y_full.mean(axis=1) >= 0.5).astype(np.int8)

    # χειρισμός τυχόν ουράς (αν δεν διαιρείται ακριβώς)
    tail = y[n_full:]
    if tail.size:
        if label_reduce == "any":
            y_tail = tail.max()
        else:
            y_tail = (tail.mean() >= 0.5)
        y_tail = np.int8(y_tail)
        y_ds = np.concatenate([y_core, [y_tail]]).astype(np.int8)
    else:
        y_ds = y_core

    # παίρνουμε κάθε 4ο sample από το df
    idx = np.arange(0, len(df), DS_FACTOR)
    df_ds = df.iloc[idx].reset_index(drop=True)

    # συγχρονίζουμε μήκη df_ds και y_ds (προσοχή στην tail περίπτωση)
    if len(df_ds) != len(y_ds):
        min_len = min(len(df_ds), len(y_ds))
        df_ds = df_ds.iloc[:min_len].reset_index(drop=True)
        y_ds = y_ds[:min_len]

    df_ds["anno"] = y_ds
    return df_ds


# --------------------- Magnitude features ---------------------

def add_magnitudes(df: pd.DataFrame, add_acc=True, add_gyro=True) -> pd.DataFrame:
    """
    Προσθέτει κανάλια magnitude:
    - acc_mag  = ||a|| = sqrt(ax^2 + ay^2 + az^2)
    - gyro_mag = ||g|| = sqrt(gx^2 + gy^2 + gz^2)
    """
    df = df.copy()
    if add_acc:
        df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    if add_gyro:
        df["gyro_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    return df


# --------------------- Jerk features ---------------------

def add_jerk_features(
    df: pd.DataFrame,
    fs: float = TARGET_FS,
    add_axes: bool = False,
    add_mag: bool = True
) -> pd.DataFrame:
    """
    Προσθέτει jerk από τα κανάλια επιτάχυνσης (ax, ay, az):
    - jax, jay, jaz  ~ διακριτή παράγωγος ax/ay/az * fs
    - jerk_mag       = sqrt(jax^2 + jay^2 + jaz^2)
    Συνήθως:
      - add_axes = False, add_mag = True → ένα extra κανάλι με ένταση jerk.
      - Αν θες full detail, θέσε add_axes=True.
    """
    df = df.copy()

    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()

    # διακριτή παράγωγος 1ου βαθμού: diff * fs
    jax = np.empty_like(ax, dtype=np.float32)
    jay = np.empty_like(ay, dtype=np.float32)
    jaz = np.empty_like(az, dtype=np.float32)

    jax[1:] = np.diff(ax) * fs
    jay[1:] = np.diff(ay) * fs
    jaz[1:] = np.diff(az) * fs

    # πρώτο δείγμα: αντιγράφουμε τη δεύτερη τιμή (σταθεροποίηση)
    jax[0] = jax[1]
    jay[0] = jay[1]
    jaz[0] = jaz[1]

    if add_axes:
        df["jax"] = jax
        df["jay"] = jay
        df["jaz"] = jaz

    if add_mag:
        df["jerk_mag"] = np.sqrt(jax**2 + jay**2 + jaz**2)

    return df


# --------------------- Z-score normalization ---------------------

def zscore_normalize(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Z-score normalization per session:
      x_norm = (x - mean) / std
    Επιστρέφει:
      - normalized DataFrame
      - stats dict (mean/std ανά κανάλι)
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0   # αποφυγή διαίρεσης με 0

    Xn = (X - mean) / std

    out = df.copy()
    out[feature_cols] = Xn

    stats = {"cols": feature_cols, "mean": mean.tolist(), "std": std.tolist()}
    return out, stats


# --------------------- Preprocess ενός Hand-Mirrored CSV ---------------------

def preprocess_handmirrored_csv(
    csv_path: str | Path,
    use_antialias: bool = True,
    anti_alias_cutoff: float = 7.5,
    label_reduce: str = "any",
    add_mags: bool = True,
    add_jerk_axes: bool = False,
    add_jerk_mag: bool = True,
):
    """
    End-to-end προεπεξεργασία για ΕΝΑ Hand-Mirrored αρχείο:
      1) Φόρτωση & έλεγχος
      2) (optional) Anti-aliasing low-pass
      3) Downsampling 64 -> 16 Hz με ασφαλές label reduction
      4) (optional) Magnitude κανάλια
      5) (optional) Jerk κανάλια
      6) Z-score normalization per session
      7) Επιστρέφει X, y, meta (dict με πληροφορίες)
    """
    csv_path = Path(csv_path)

    # 1) φόρτωση raw Hand-Mirrored
    df_raw = load_handmirrored_csv(csv_path)

    # σύνοψη labels πριν το οτιδήποτε
    summarize_labels(df_raw["anno"].to_numpy(), title=f"{csv_path.name} @64Hz raw")

    df = df_raw

    # 2) anti-aliasing πριν το decimation
    if use_antialias:
        df = apply_antialias_lowpass(df, fs=RAW_FS, cutoff=anti_alias_cutoff, order=4)

    # 3) downsample 64 -> 16 Hz
    df_ds = downsample_df(df, label_reduce=label_reduce)
    summarize_labels(df_ds["anno"].to_numpy(), title=f"{csv_path.name} @{TARGET_FS}Hz downsampled")

    # 4) magnitudes
    if add_mags:
        df_ds = add_magnitudes(df_ds, add_acc=True, add_gyro=True)

    # 5) jerk features
    df_ds = add_jerk_features(
        df_ds,
        fs=TARGET_FS,
        add_axes=add_jerk_axes,
        add_mag=add_jerk_mag
    )

    # ορίζουμε τη σειρά των χαρακτηριστικών
    feature_cols = ["ax","ay","az","gx","gy","gz"]
    if add_mags:
        feature_cols += ["acc_mag","gyro_mag"]
    if add_jerk_axes:
        feature_cols += ["jax","jay","jaz"]
    if add_jerk_mag:
        feature_cols += ["jerk_mag"]

    # 6) z-score normalization per session (fit+transform)
    df_norm, stats = zscore_normalize(df_ds, feature_cols)

    # 7) μετατροπή σε numpy arrays έτοιμα για input σε μοντέλο
    X = df_norm[feature_cols].to_numpy(dtype=np.float32)  # [N, C]
    y = df_norm["anno"].to_numpy(dtype=np.int8)           # [N]

    meta = {
        "source_csv": csv_path.name,
        "fs_raw": RAW_FS,
        "fs_target": TARGET_FS,
        "label_reduce": label_reduce,
        "use_antialias": use_antialias,
        "anti_alias_cutoff": anti_alias_cutoff,
        "features": feature_cols,
        "zscore_stats": stats,
        "hand_mirrored": True,  # ήδη right + mirrored-left
        "note": "Hand-Mirrored: right-hand IMU concatenated with hand-mirrored left-hand IMU in time.",
    }

    return X, y, meta


# --------------------- Preprocess ΟΛΟΝ τον φάκελο Hand-Mirrored ---------------------

def preprocess_handmirrored_folder(root: str | Path):
    """
    Τρέχει το ίδιο pipeline για ΟΛΑ τα Hand-Mirrored αρχεία ενός φακέλου
    (π.χ. DX-I/Hand-Mirrored ή DX-II/Hand-Mirrored).
    Επιστρέφει λίστες [X_list, y_list, meta_list] αντίστοιχα με τη σειρά των .csv.
    """
    root = Path(root)
    csv_files = sorted(root.glob("*.csv"))  # sorted για determinism

    X_list, y_list, meta_list = [], [], []

    for csv_path in csv_files:
        print(f"\n=== Processing {csv_path.name} ===")
        X, y, meta = preprocess_handmirrored_csv(csv_path)
        X_list.append(X)
        y_list.append(y)
        meta_list.append(meta)

    return X_list, y_list, meta_list


# --------------------- Παράδειγμα χρήσης ---------------------

if __name__ == "__main__":
    # Παράδειγμα για ένα αρχείο:
    # in_csv = Path("DX-I/Hand-Mirrored/4-1.csv")
    # X, y, meta = preprocess_handmirrored_csv(in_csv)

    # Παράδειγμα για όλο τον φάκελο DX-I Hand-Mirrored:
    hm_root = Path(r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\dataverse_files\DX-I\DX-I\Hand-Mirrored"
)
    X_list, y_list, meta_list = preprocess_handmirrored_folder(hm_root)

    # π.χ. μπορείς να αποθηκεύσεις σε .pkl όπως στο original:
    import pickle as pkl
    out_dir = Path(r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\code")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "DX_I_X_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(X_list, f)
    with open(out_dir / "DX_I_Y_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(y_list, f)
    with open(out_dir / "DX_I_meta_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(meta_list, f)