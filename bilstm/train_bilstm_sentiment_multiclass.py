import os, json, random
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ================== 0. Reproducibility ==================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================== 1. Config ==================
# Đổi path này nếu cần
CSV_PATH = "/home/dat/llm_ws/data/train/train.csv"

MAX_VOCAB     = 20000
SEQ_LEN       = 64
EMBED_DIM     = 128
BILSTM_UNITS  = 128
BATCH_SIZE    = 32
EPOCHS        = 7
PATIENCE      = 4
FOCAL_GAMMA   = 2.0   # gamma cho focal loss

FIVE_CLASS = ["very_negative", "negative", "neutral", "positive", "very_positive"]

EXPORT_DIR = "bilstm_vn_sentiment_5cls"


# ================== 2. Load & chuẩn hoá data ==================
df = pd.read_csv(CSV_PATH)
assert {"text", "label"}.issubset(df.columns), "CSV phải có cột text, label"

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Chỉ giữ 5 lớp chuẩn
df = df[df["label"].isin(FIVE_CLASS)].copy()

label_counts = df["label"].value_counts()
print("Label counts trước khi kiểm tra:")
print(label_counts)

# Kiểm tra đủ 5 class chưa
missing = [c for c in FIVE_CLASS if c not in label_counts.index]
if len(missing) > 0:
    raise ValueError(
        f"Thiếu các nhãn {missing} trong dữ liệu. "
        f"File {CSV_PATH} phải chứa đủ 5 class: {FIVE_CLASS}"
    )

# Map label -> id
class_list = FIVE_CLASS[:]  # giữ thứ tự cố định
label2id = {c: i for i, c in enumerate(class_list)}
id2label = {i: c for c, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

print("\nClasses (5-class):", class_list)
print("Counts:", df["label"].value_counts().to_dict())

# ================== 3. Split train / val ==================
y_all = df["label_id"].values
can_stratify = True
for lab, cnt in Counter(y_all).items():
    if cnt < 2:
        can_stratify = False
        print(f"⚠️  Cảnh báo: class {id2label[lab]} chỉ có {cnt} mẫu. "
              f"Sẽ không dùng stratify trong train_test_split.")
        break

if can_stratify:
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values,
        df["label_id"].values,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label_id"].values,
    )
else:
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].values,
        df["label_id"].values,
        test_size=0.2,
        random_state=SEED,
    )

print("\nTrain size:", len(X_train), " | Val size:", len(X_val))

# ================== 3.1. Class weights ==================
cnt_train = Counter(y_train)
max_count = max(cnt_train.values())
class_weight = {cls: max_count / cnt_train[cls] for cls in cnt_train}
print("Class weights (train):")
print({id2label[k]: round(v, 3) for k, v in class_weight.items()})

# Ta sẽ dùng class_weight cho cả:
# - Keras class_weight trong model.fit
# - alpha (per-class) trong focal loss
FOCAL_ALPHA = {int(cls): float(max_count / cnt_train[cls]) for cls in cnt_train}


# ================== 4. TextVectorization ==================
text_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_mode="int",
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
)

text_vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(X_train).batch(64)
)


# ================== 5. Model BiLSTM ==================
inputs = layers.Input(shape=(1,), dtype=tf.string, name="text")
x = text_vectorizer(inputs)                                   # [B, L]
x = layers.Embedding(MAX_VOCAB, EMBED_DIM, mask_zero=True)(x) # [B, L, D]
x = layers.SpatialDropout1D(0.2)(x)
x = layers.Bidirectional(
    layers.LSTM(BILSTM_UNITS, return_sequences=True)
)(x)

# Dual pooling
avg_pool = layers.GlobalAveragePooling1D()(x)
max_pool = layers.GlobalMaxPooling1D()(x)
x = layers.Concatenate()([avg_pool, max_pool])
x = layers.Dropout(0.35)(x)
x = layers.Dense(192, activation="relu")(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(len(class_list), activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()


# ================== 5.1. Focal loss (sparse) ==================
def make_sparse_categorical_focal_loss(gamma=2.0, alpha_dict=None):
    """
    Tạo focal loss cho multi-class với nhãn integer (sparse).
    alpha_dict: {class_id: alpha}
    """
    if alpha_dict is None:
        alpha_dict = {}

    alpha_vec = np.ones(len(class_list), dtype="float32")
    for cid, a in alpha_dict.items():
        if 0 <= cid < len(class_list):
            alpha_vec[cid] = float(a)
    alpha_vec = tf.constant(alpha_vec, dtype=tf.float32)

    @tf.function
    def loss_fn(y_true, y_pred):
        """
        y_true: int32 [B], y_pred: probs [B, C]
        """
        y_true_cast = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_clip = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # gather p_t
        idx = tf.stack([tf.range(tf.shape(y_pred_clip)[0]), y_true_cast], axis=1)
        p_t = tf.gather_nd(y_pred_clip, idx)  # [B]

        # alpha_t theo class
        alpha_t = tf.gather(alpha_vec, y_true_cast)  # [B]

        loss = - alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return loss_fn


focal_loss = make_sparse_categorical_focal_loss(
    gamma=FOCAL_GAMMA,
    alpha_dict=FOCAL_ALPHA,
)

opt = optimizers.Adam(learning_rate=2e-3)
model.compile(
    optimizer=opt,
    loss=focal_loss,
    metrics=["accuracy"],
)


# ================== 6. TF Datasets ==================
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(4096, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ================== 7. Callbacks ==================
class MacroF1Callback(callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.best_f1 = -1.0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.val_ds, verbose=0)
        y_pred = probs.argmax(axis=1)
        y_true = np.concatenate([y for _, y in self.val_ds], axis=0)
        f1 = f1_score(y_true, y_pred, average="macro")
        logs = logs or {}
        logs["val_macro_f1"] = f1
        print(f"\n[Epoch {epoch+1}] val_macro_f1 = {f1:.4f}")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"\nRestored best weights with val_macro_f1 = {self.best_f1:.4f}")


macro_cb = MacroF1Callback(val_ds)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
    verbose=1,
)

earlystop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1,
)


# ================== 8. Train ==================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[macro_cb, reduce_lr, earlystop],
    verbose=1,
    class_weight=class_weight,
)


# ================== 9. Evaluate ==================
val_probs = model.predict(val_ds, verbose=0)
val_pred = val_probs.argmax(axis=1)
y_true = np.concatenate([y for _, y in val_ds], axis=0)

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, val_pred))

print("\nClassification report (5 class):")
print(
    classification_report(
        y_true,
        val_pred,
        target_names=class_list,
        digits=4,
    )
)
macro_f1 = f1_score(y_true, val_pred, average="macro")
print(f"\nMacro-F1: {macro_f1:.4f}")


# ================== 10. Hàm predict nhanh ==================
def predict_texts(text_list):
    ds = tf.data.Dataset.from_tensor_slices(text_list).batch(64)
    probs = model.predict(ds, verbose=0)
    idx = probs.argmax(axis=1)
    res = []
    for i, t in enumerate(text_list):
        lab_id = int(idx[i])
        lab = id2label[lab_id]
        score = float(probs[i, lab_id])
        res.append((t, score, lab))
    return res


demo_texts = [
    "Thiết bị robot hút bụi thất vọng, ồn, shop phản hồi chậm.",
    "Màn hình tuyệt hảo, không chê vào đâu được.",
    "Sản phẩm tệ hại, hỏng ngay khi mở hộp, yêu cầu hoàn tiền.",
    "Mình thấy laptop đóng gói ổn, dùng tạm được.",
    "Điện thoại mượt mà, pin khoẻ, rất đáng tiền.",
]

print("\nDemo predictions:")
for t, p, lab in predict_texts(demo_texts):
    print(f"[{lab:14s}] {p:0.3f} | {t}")


# ================== 11. Export SavedModel + label_map ==================
if os.path.exists(EXPORT_DIR):
    import shutil

    shutil.rmtree(EXPORT_DIR)

# model đã nhận input string -> vectorizer nằm bên trong -> save thẳng
model.save(EXPORT_DIR)

with open(os.path.join(EXPORT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f,
              ensure_ascii=False, indent=2)

print(f"\nSavedModel exported to: {EXPORT_DIR}")
print("Label map:", label2id)
