import librosa
import numpy as np
import os

# --- Speaker Verification Configuration ---

# File path to save the voice signature (MFCC mean)
VOICE_SIGNATURE_PATH = "voice_signature.npy"
# Threshold for Euclidean distance (lower is closer match)
VERIFICATION_THRESHOLD = 70.0

# --- Voice Signature Functions ---

def extract_mfccs(audio_path: str) -> np.ndarray | None:
    """Extracts mean MFCC features from an audio file."""
    try:
        # Load audio file (mono, 16kHz)
        y, sr = librosa.load(audio_path, sr=16000)
        # Extract MFCCs (13 coefficients by default)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Return the mean of all MFCCs across time (the 'signature')
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None


def register_voice(audio_path: str | None) -> str:
    """Registers the speaker by saving their voice signature."""
    if audio_path is None:
        return "Registration failed: Please record a sample of your voice [5-10sec]."

    signature = extract_mfccs(audio_path)

    if signature is not None:
        try:
            np.save(VOICE_SIGNATURE_PATH, signature)
            return f"Voice registered successfully! Signature saved to '{VOICE_SIGNATURE_PATH}'. You can now use the Friday AI Assistant."
        except Exception as e:
            return f"Registration failed: Could not save signature. Details: {e}"
    else:
        return "Registration failed: Could not process audio features."


def verify_voice(audio_path: str | None) -> tuple[bool, str]:
    """Verifies the speaker against the stored signature."""
    if not os.path.exists(VOICE_SIGNATURE_PATH):
        return False, "Verification required: No voice signature found. Please register your voice first."

    if audio_path is None:
        return False, "Verification failed: No audio provided for verification."

    stored_signature = np.load(VOICE_SIGNATURE_PATH)
    current_signature = extract_mfccs(audio_path)

    if current_signature is None:
        return False, "Verification failed: Could not process input audio features."

    # Calculate Euclidean distance between the stored and current signatures
    distance = np.linalg.norm(stored_signature - current_signature)

    # Simple verification logic: Check if distance is below the threshold
    if distance < VERIFICATION_THRESHOLD:
        return True, f"Voice Verification Successful!" # Distance: {distance:.2f}, Threshold: {VERIFICATION_THRESHOLD}
    else:
        return False, f"Verification failed: Speaker mismatch detected." # Distance: {distance:.2f} (Threshold: {VERIFICATION_THRESHOLD}
