import os, glob, warnings, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing   import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers    import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models    import Model
from tensorflow.keras.callbacks import EarlyStopping

print(f"TensorFlow : {tf.__version__}\n")

# =============================================================================
# STEP 1 : LOAD DATA
# =============================================================================
DATA_DIR  = r"C:\IDS_Project\data\MachineLearningCVE"
csv_paths = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
if not csv_paths:
    raise FileNotFoundError(f"No CSVs found in {DATA_DIR}")

print(f"Loading {len(csv_paths)} CSV file(s) ...")
df_list = []
for path in csv_paths:
    tmp = pd.read_csv(path, low_memory=False, encoding="latin-1")
    print(f"  v {os.path.basename(path):55s} {str(tmp.shape):>15}")
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()
print(f"\nCombined shape: {df.shape}")
LABEL_COL = [c for c in df.columns if c.lower() == "label"][0]

# =============================================================================
# STEP 2 : KEEP FINE-GRAINED LABELS FOR STRATIFIED SPLITTING
#   Split by fine-grained attack type FIRST, then map to broad classes.
#  ============================================================================
df[LABEL_COL] = df[LABEL_COL].str.strip()

KNOWN_LABELS = {
    "BENIGN","DoS Hulk","DoS GoldenEye","DoS slowloris","DoS Slowhttptest",
    "Heartbleed","DDoS","PortScan","FTP-Patator","SSH-Patator","Bot",
    "Infiltration",
    "Web Attack \x96 Brute Force","Web Attack \x96 XSS","Web Attack \x96 Sql Injection",
    "Web Attack \xe2\x80\x93 Brute Force",
    "Web Attack - Brute Force","Web Attack - XSS","Web Attack - Sql Injection",
    "Web Attack \u2013 Brute Force","Web Attack \u2013 XSS","Web Attack \u2013 Sql Injection",
}
before = len(df)
df = df[df[LABEL_COL].isin(KNOWN_LABELS)]
print(f"Dropped {before-len(df):,} unrecognised rows.\n")
print("Fine-grained label counts:")
print(df[LABEL_COL].value_counts().to_string(), "\n")

fine_le = LabelEncoder()
y_fine  = fine_le.fit_transform(df[LABEL_COL])

# =============================================================================
# STEP 3 : FEATURE CLEANING
# =============================================================================
X_raw = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw = X_raw.fillna(X_raw.median()).astype(np.float32)
X_raw = X_raw.loc[:, X_raw.std() > 0]
print(f"Features after cleaning : {X_raw.shape[1]}")

# =============================================================================
# STEP 4 : BROAD LABEL MAPPING
# =============================================================================
BROAD_MAP = {
    "BENIGN":"Normal",
    "DoS Hulk":"DoS","DoS GoldenEye":"DoS","DoS slowloris":"DoS",
    "DoS Slowhttptest":"DoS","Heartbleed":"DoS","DDoS":"DoS",
    "PortScan":"Intrusion","FTP-Patator":"Intrusion","SSH-Patator":"Intrusion",
    "Bot":"Intrusion","Infiltration":"Intrusion",
    "Web Attack \x96 Brute Force":"Intrusion",
    "Web Attack \x96 XSS":"Intrusion",
    "Web Attack \x96 Sql Injection":"Intrusion",
    "Web Attack \xe2\x80\x93 Brute Force":"Intrusion",
    "Web Attack - Brute Force":"Intrusion",
    "Web Attack - XSS":"Intrusion",
    "Web Attack - Sql Injection":"Intrusion",
    "Web Attack \u2013 Brute Force":"Intrusion",
    "Web Attack \u2013 XSS":"Intrusion",
    "Web Attack \u2013 Sql Injection":"Intrusion",
}

broad_labels = df[LABEL_COL].map(BROAD_MAP)
le           = LabelEncoder()
y_broad      = le.fit_transform(broad_labels)
CLASS_NAMES  = list(le.classes_)
NUM_CLASSES  = len(CLASS_NAMES)
print(f"Broad classes : {CLASS_NAMES}\n")

X_scaled = StandardScaler().fit_transform(X_raw).astype(np.float32)
del df, X_raw; gc.collect()

# =============================================================================
# STEP 5 : FINE-GRAINED STRATIFIED TEMPORAL SPLIT
#   For each fine-grained subtype: first 70% -> train, last 30% -> test
#   Cap per subtype at MAX_PER_FINE to keep memory manageable
# =============================================================================
MAX_PER_FINE = 5_000
TRAIN_RATIO  = 0.70

def fine_grained_split(X, y_fine, y_broad, max_per_fine=5000, train_ratio=0.70):
    rng = np.random.default_rng(42)
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for cls in np.unique(y_fine):
        idx   = np.where(y_fine == cls)[0]
        n     = min(len(idx), max_per_fine)
        idx   = np.sort(rng.choice(idx, n, replace=False))
        split = int(len(idx) * train_ratio)
        X_tr.append(X[idx[:split]]);  y_tr.append(y_broad[idx[:split]])
        X_te.append(X[idx[split:]]);  y_te.append(y_broad[idx[split:]])
    return (np.vstack(X_tr), np.concatenate(y_tr),
            np.vstack(X_te), np.concatenate(y_te))

X_tr, y_tr, X_te, y_te = fine_grained_split(X_scaled, y_fine, y_broad, MAX_PER_FINE, TRAIN_RATIO)

print(f"Train : {len(y_tr):,}")
print(pd.Series(y_tr).map(dict(enumerate(CLASS_NAMES))).value_counts().to_string())
print(f"\nTest  : {len(y_te):,}")
print(pd.Series(y_te).map(dict(enumerate(CLASS_NAMES))).value_counts().to_string(), "\n")

# =============================================================================
# STEP 6 : SLIDING WINDOW PER BROAD CLASS
# =============================================================================
WINDOW_SIZE = 10

def create_windows(X, y):
    Xw, yw = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        X_c, y_c = X[idx], y[idx]
        for i in range(len(X_c) - WINDOW_SIZE):
            Xw.append(X_c[i:i+WINDOW_SIZE])
            yw.append(y_c[i+WINDOW_SIZE])
    Xw   = np.array(Xw, dtype=np.float32)
    yw   = np.array(yw)
    perm = np.random.default_rng(42).permutation(len(yw))
    return Xw[perm], yw[perm]

X_seq_tr, y_seq_tr = create_windows(X_tr, y_tr)
X_seq_te, y_seq_te = create_windows(X_te, y_te)
print(f"Train sequences : {X_seq_tr.shape}")
print(f"Test  sequences : {X_seq_te.shape}\n")
gc.collect()

# =============================================================================
# STEP 7 : MODEL  (slightly deeper to handle Intrusion diversity)
# =============================================================================
def build_backbone(shape):
    inp = Input(shape=shape)
    x   = Conv1D(64, 3, activation="relu", padding="same")(inp)
    x   = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2))(x)
    x   = Dropout(0.3)(x)
    emb = Dense(64, activation="relu")(x)
    return Model(inp, emb)

def build_model(backbone):
    out   = Dense(NUM_CLASSES, activation="softmax")(backbone.output)
    model = Model(backbone.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

INPUT_SHAPE = (X_seq_tr.shape[1], X_seq_tr.shape[2])

# =============================================================================
# STEP 8 : FEDERATED TRAINING
# =============================================================================
NUM_CLIENTS   = 3
client_X      = np.array_split(X_seq_tr, NUM_CLIENTS)
client_y      = np.array_split(y_seq_tr, NUM_CLIENTS)
local_weights = []
CLS_W         = {0: 1.0, 1: 2.0, 2: 1.0}   # upweight Intrusion
early_stop    = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

for i in range(NUM_CLIENTS):
    print(f"\n{'='*50}\n  RSU-{i+1}  |  {len(client_X[i]):,} samples\n{'='*50}")
    backbone = build_backbone(INPUT_SHAPE)
    model    = build_model(backbone)
    model.fit(
        client_X[i], client_y[i],
        epochs=15, batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        class_weight=CLS_W,
        verbose=1,
    )
    local_weights.append(backbone.get_weights())

global_backbone = build_backbone(INPUT_SHAPE)
global_backbone.set_weights(
    [np.mean(np.stack(w), axis=0) for w in zip(*local_weights)]
)
print("\nv FedAvg complete.")

# =============================================================================
# STEP 9 : FEATURE EXTRACTION
# =============================================================================
print("Extracting features ...")
train_feat = global_backbone.predict(X_seq_tr, batch_size=256, verbose=0)
test_feat  = global_backbone.predict(X_seq_te, batch_size=256, verbose=0)

# =============================================================================
# STEP 10 : RANDOM FOREST
# =============================================================================
print("Training Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=2,
    class_weight={0:1.0, 1:2.0, 2:1.0},
    n_jobs=-1, random_state=42
)
rf.fit(train_feat, y_seq_tr)

# =============================================================================
# STEP 11 : THRESHOLD TUNING FOR INTRUSION
# =============================================================================
y_prob        = rf.predict_proba(test_feat)
intrusion_idx = CLASS_NAMES.index("Intrusion")
THRESHOLD     = 0.30

y_pred = np.where(
    y_prob[:, intrusion_idx] > THRESHOLD,
    intrusion_idx,
    np.argmax(y_prob, axis=1)
)

# =============================================================================
# STEP 12 : RESULTS
# =============================================================================
print("\n-- Classification Report ------------------------------------------")
print(classification_report(y_seq_te, y_pred, target_names=CLASS_NAMES))

# =============================================================================
# STEP 13 : PLOTS
# =============================================================================
OUT = r"C:\IDS_Project"

cm = confusion_matrix(y_seq_te, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid IDS (CIC-IDS-2017)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
print("v confusion_matrix.png saved")

y_test_bin = label_binarize(y_seq_te, classes=list(range(NUM_CLASSES)))
plt.figure(figsize=(7,5))
colors = ["#e74c3c","#2ecc71","#3498db"]
for i,(lbl,col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_prob[:,i])
    plt.plot(fpr, tpr, color=col, lw=2, label=f"{lbl}  (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],"k--",lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC-AUC | Hybrid IDS | CIC-IDS-2017")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "roc_auc.png"), dpi=150)
print("v roc_auc.png saved")

importances = rf.feature_importances_
indices     = np.argsort(importances)[::-1]
plt.figure(figsize=(10,4))
plt.bar(range(len(importances)), importances[indices], color="#3498db")
plt.xticks(range(len(importances)), [f"F{i}" for i in indices], rotation=90, fontsize=7)
plt.xlabel("Deep Feature (sorted)"); plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
print("v feature_importance.png saved")

print("\nv All done!")
