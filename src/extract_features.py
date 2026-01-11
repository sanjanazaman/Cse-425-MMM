
import os
import numpy as np
import librosa
from tqdm import tqdm


SR = 22050
N_MFCC = 40
MAX_DURATION = 30  

BANGLA_DIR = "../data/bangladataset"  
ENGLISH_DIR = "../data/engdataset"
RESULTS_DIR = "../results"
OUTPUT_FILE = "features_combined.npz"

os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, OUTPUT_FILE)

def extract_audio_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SR, duration=MAX_DURATION)

    
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)

       
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        features = np.concatenate([
            mfcc_mean,
            [spec_centroid, spec_bandwidth, rolloff, zcr]
        ])

        return features

    except Exception as e:
        print(f" Error processing {filepath}: {e}")
        return None


def load_dataset(folder, label, language_id):
    X, y, lang = [], [], []

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return X, y, lang

    print(f"Extracting features from {folder} ...")
    for root, _, files in os.walk(folder):
        for file in tqdm(files, desc=os.path.basename(folder)):
            if file.lower().endswith(".wav"):
                path = os.path.join(root, file)
                features = extract_audio_features(path)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    lang.append(language_id)
    return X, y, lang


def main():
   
    X_bn, y_bn, lang_bn = load_dataset(BANGLA_DIR, label=0, language_id=0)

    
    X_en, y_en, lang_en = load_dataset(ENGLISH_DIR, label=1, language_id=1)

   
    X = np.array(X_bn + X_en)
    y = np.array(y_bn + y_en)
    lang = np.array(lang_bn + lang_en)

   
    if X.ndim == 1:
        print(" No valid features extracted!")
        return
    mask = ~np.isnan(X).any(axis=1)
    X, y, lang = X[mask], y[mask], lang[mask]

    print(f"Total samples: {X.shape[0]}")
    print(f"Feature dimension: {X.shape[1]}")

    
    np.savez(OUTPUT_PATH, X=X, y=y, language=lang)
    print(f"Features saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
