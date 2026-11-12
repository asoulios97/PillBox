import numpy as np                     # αριθμητικοί πίνακες
import pandas as pd                    # ανάγνωση CSV & dataframes
from pathlib import Path               # ασφαλής χειρισμός διαδρομών
from collections import Counter        # μέτρηση συχνοτήτων labels
import json                            # αποθήκευση meta πληροφοριών
from typing import List, Dict, Tuple, Any
from tqdm import tqdm                  # progress bar

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


# --------------------- Catalog Creation ---------------------

def create_file_catalog(raw_annotation_root: str | Path) -> pd.DataFrame:
    """
    Δημιουργεί catalog DataFrame με userId, hand, και path για όλα τα αρχεία
    στους φακέλους Left hand και Right hand.
    
    Args:
        raw_annotation_root: Path to "Raw data and Annotation" folder
        
    Returns:
        DataFrame με στήλες: userId, hand, path
    """
    raw_annotation_root = Path(raw_annotation_root)
    
    catalog_data = []
    
    # Διαβάζουμε από Left hand folder
    left_hand_dir = raw_annotation_root / "Left hand"
    if left_hand_dir.exists():
        for csv_file in sorted(left_hand_dir.glob("*.csv")):
            # Το όνομα αρχείου (π.χ. "1-3.csv") χρησιμοποιείται ως userId
            user_id = csv_file.stem  # π.χ. "1-3"
            catalog_data.append({
                "userId": user_id,
                "hand": "Left",
                "path": str(csv_file)
            })
    
    # Διαβάζουμε από Right hand folder
    right_hand_dir = raw_annotation_root / "Right hand"
    if right_hand_dir.exists():
        for csv_file in sorted(right_hand_dir.glob("*.csv")):
            user_id = csv_file.stem  # π.χ. "1-3"
            catalog_data.append({
                "userId": user_id,
                "hand": "Right",
                "path": str(csv_file)
            })
    
    catalog_df = pd.DataFrame(catalog_data)
    # Ταξινόμηση κατά userId και hand για καλύτερη οργάνωση
    catalog_df = catalog_df.sort_values(["userId", "hand"]).reset_index(drop=True)
    
    return catalog_df


# --------------------- Hand Mirroring Transformation ---------------------

def apply_hand_mirroring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Εφαρμόζει το hand mirroring transformation σε Left Hand δεδομένα.
    Σύμφωνα με το paper, αντιστρέφουμε τα: ax, gy, gz (πολλαπλασιασμός με -1)
    
    Transformation matrix (equation 1):
    [-1  0  0  0  0  0 ]   [ax]
    [ 0  1  0  0  0  0 ]   [ay]
    [ 0  0  1  0  0  0 ] × [az]
    [ 0  0  0  1  0  0 ]   [gx]
    [ 0  0  0  0 -1  0 ]   [gy]
    [ 0  0  0  0  0 -1 ]   [gz]
    
    Args:
        df: DataFrame με Left hand IMU δεδομένα
        
    Returns:
        DataFrame με mirrored δεδομένα
    """
    df = df.copy()
    
    # Αντιστροφή ax, gy, gz
    df["ax"] = -df["ax"]
    df["gy"] = -df["gy"]
    df["gz"] = -df["gz"]
    
    return df


# --------------------- Load Raw Annotation CSV ---------------------

def load_raw_annotation_csv(csv_path: Path, is_left_hand: bool = False) -> pd.DataFrame:
    """
    Διαβάζει ένα Raw data and Annotation αρχείο (.csv) και κάνει βασικούς ελέγχους.
    Αν είναι Left hand, εφαρμόζει αυτόματα το hand mirroring transformation.
    
    Args:
        csv_path: Path to the CSV file
        is_left_hand: True αν το αρχείο είναι από Left hand folder
    
    Returns:
        DataFrame με (mirrored αν χρειάζεται) δεδομένα
    """
    df = pd.read_csv(csv_path)
    
    # Έλεγχος στηλών
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: λείπουν στήλες {missing}")
    
    # Ταξινόμηση κατά χρόνο
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    
    # Μετατροπή τύπων
    imu_cols = ["ax","ay","az","gx","gy","gz"]
    df[imu_cols] = df[imu_cols].astype(np.float32)
    df["anno"] = df["anno"].astype(np.int8)
    
    # Εφαρμογή hand mirroring αν είναι Left hand
    if is_left_hand:
        df = apply_hand_mirroring(df)
        try:
            # If tqdm is available and running, use tqdm.write
            tqdm.write(f"  ✓ Applied hand mirroring transformation (Left → Right coordinate frame)")
        except:
            # Fallback to regular print if not in tqdm context
            print(f"  ✓ Applied hand mirroring transformation (Left → Right coordinate frame)")
    
    return df


# --------------------- Preprocess από Catalog ---------------------

def preprocess_from_catalog(
    catalog_df: pd.DataFrame,
    use_antialias: bool = True,
    anti_alias_cutoff: float = 7.5,
    label_reduce: str = "any",
    add_mags: bool = True,
    add_jerk_axes: bool = False,
    add_jerk_mag: bool = True,
):
    """
    Επεξεργάζεται όλα τα αρχεία από το catalog DataFrame.
    Κάθε αρχείο (userId + hand) επεξεργάζεται ως ξεχωριστό, μοναδικό entry.
    Τα Left hand δεδομένα μετασχηματίζονται με mirroring στο Right coordinate frame.
    
    Args:
        catalog_df: DataFrame με userId, hand, path
        ... (υπόλοιπες παράμετροι preprocessing)
        
    Returns:
        X_list, y_list, meta_list (ένα entry ανά αρχείο: userId + hand)
    """
    X_list, y_list, meta_list = [], [], []
    
    # Process each file independently - each userId/hand combination is unique
    print(f"\nProcessing {len(catalog_df)} files...")
    
    for idx, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), 
                         desc="Preprocessing files", unit="file"):
        csv_path = Path(row["path"])
        user_id = row["userId"]
        hand = row["hand"]
        is_left = (hand == "Left")
        
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Processing: User {user_id} - {hand} hand ({idx+1}/{len(catalog_df)})")
        tqdm.write(f"{'='*60}")
        
        # Load with automatic mirroring for left hand
        df = load_raw_annotation_csv(csv_path, is_left_hand=is_left)
        
        # Capture label summary
        cnt = Counter(df["anno"].tolist())
        total = len(df)
        pos = cnt.get(1, 0)
        tqdm.write(f"  @64Hz raw: {total:,} samples, anno=1: {pos:,} ({pos/total*100:.2f}%)")
        
        # Apply preprocessing pipeline
        
        # Anti-aliasing
        if use_antialias:
            df = apply_antialias_lowpass(df, fs=RAW_FS, cutoff=anti_alias_cutoff, order=4)
        
        # Downsampling
        df_ds = downsample_df(df, label_reduce=label_reduce)
        
        # Capture downsampled label summary
        cnt_ds = Counter(df_ds["anno"].tolist())
        total_ds = len(df_ds)
        pos_ds = cnt_ds.get(1, 0)
        tqdm.write(f"  @{TARGET_FS}Hz downsampled: {total_ds:,} samples, anno=1: {pos_ds:,} ({pos_ds/total_ds*100:.2f}%)")
        
        # Magnitudes
        if add_mags:
            df_ds = add_magnitudes(df_ds, add_acc=True, add_gyro=True)
        
        # Jerk features
        df_ds = add_jerk_features(
            df_ds,
            fs=TARGET_FS,
            add_axes=add_jerk_axes,
            add_mag=add_jerk_mag
        )
        
        # Feature columns
        feature_cols = ["ax","ay","az","gx","gy","gz"]
        if add_mags:
            feature_cols += ["acc_mag","gyro_mag"]
        if add_jerk_axes:
            feature_cols += ["jax","jay","jaz"]
        if add_jerk_mag:
            feature_cols += ["jerk_mag"]
        
        # Z-score normalization
        df_norm, stats = zscore_normalize(df_ds, feature_cols)
        
        # Convert to numpy
        X = df_norm[feature_cols].to_numpy(dtype=np.float32)
        y = df_norm["anno"].to_numpy(dtype=np.int8)
        
        meta = {
            "user_id": user_id,
            "hand": hand,
            "source_file": csv_path.name,
            "fs_raw": RAW_FS,
            "fs_target": TARGET_FS,
            "label_reduce": label_reduce,
            "use_antialias": use_antialias,
            "anti_alias_cutoff": anti_alias_cutoff,
            "features": feature_cols,
            "zscore_stats": stats,
            "hand_mirrored": is_left,
            "note": f"{hand} hand IMU data. " + ("Left hand mirrored to Right coordinate frame." if is_left else "Right hand data used as-is."),
        }
        
        X_list.append(X)
        y_list.append(y)
        meta_list.append(meta)
    
    return X_list, y_list, meta_list


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
    import pickle as pkl
    import sys
    
    # Fix console encoding for Greek characters
    if sys.platform == "win32":
        try:
            # Try to set UTF-8 encoding for Windows console
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except:
            pass
    
    # ========== ΝΕΟΣ ΤΡΟΠΟΣ: Raw data and Annotation με Catalog ==========
    
    # 1) Ορίζουμε το path στον φάκελο "Raw data and Annotation"
    raw_annotation_root = Path(
        r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\dataverse_files\DX-I\DX-I\Raw data and Annotation"
    )
    
    # 2) Δημιουργούμε το catalog
    print("Creating file catalog...")
    catalog = create_file_catalog(raw_annotation_root)
    print(f"\nCatalog created with {len(catalog)} files:")
    
    # Print catalog safely (avoid full path display to prevent encoding issues)
    catalog_display = catalog.copy()
    catalog_display['path'] = catalog_display['path'].apply(lambda x: Path(x).name)
    print(catalog_display)
    
    print(f"\nUnique users: {catalog['userId'].nunique()}")
    print(f"Files per hand: Left={len(catalog[catalog['hand']=='Left'])}, Right={len(catalog[catalog['hand']=='Right'])}")
    
    # 3) Επεξεργασία από το catalog (με αυτόματο hand mirroring)
    print("\n" + "="*60)
    print("Starting preprocessing with hand mirroring...")
    print("="*60)
    
    X_list, y_list, meta_list = preprocess_from_catalog(
        catalog,
        use_antialias=True,
        anti_alias_cutoff=7.5,
        label_reduce="any",
        add_mags=True,
        add_jerk_axes=False,
        add_jerk_mag=True
    )
    
    # 4) Αποθήκευση σε .pkl αρχεία
    out_dir = Path(r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\code")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Saving preprocessed data...")
    print("="*60)
    
    with open(out_dir / "DX_I_X_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(X_list, f)
    print(f"✓ Saved: DX_I_X_16Hz_with_jerk.pkl ({len(X_list)} samples)")
    
    with open(out_dir / "DX_I_Y_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(y_list, f)
    print(f"✓ Saved: DX_I_Y_16Hz_with_jerk.pkl")
    
    with open(out_dir / "DX_I_meta_16Hz_with_jerk.pkl","wb") as f:
        pkl.dump(meta_list, f)
    print(f"✓ Saved: DX_I_meta_16Hz_with_jerk.pkl")
    
    # 5) Αποθήκευση του catalog για reference
    catalog.to_csv(out_dir / "DX_I_catalog.csv", index=False)
    print(f"✓ Saved: DX_I_catalog.csv")
    
    print(f"\nDataset info:")
    print(f"  - Total samples: {len(X_list)} (each userId + hand is unique)")
    print(f"  - Left hand samples (mirrored): {sum(1 for m in meta_list if m['hand'] == 'Left')}")
    print(f"  - Right hand samples: {sum(1 for m in meta_list if m['hand'] == 'Right')}")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    
    # Εμφάνιση summary
    print("\nSummary:")
    for i, (X, y, meta) in enumerate(zip(X_list, y_list, meta_list)):
        print(f"  [{i+1}] User {meta['user_id']} ({meta['hand']} hand): "
              f"X.shape={X.shape}, y.shape={y.shape}, "
              f"features={len(meta['features'])}, positive_samples={y.sum()}, "
              f"mirrored={meta['hand_mirrored']}")
    