import cv2
import os
import numpy as np

def load_real_notes(folder):
    references = {}
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            label = os.path.splitext(file)[0].strip()
            img = cv2.imread(os.path.join(folder, file), 0)
            if img is not None:
                references[label] = img
    return references

def get_orb_match_score(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.resize(img1, (400, 200)), None)
    kp2, des2 = orb.detectAndCompute(cv2.resize(img2, (400, 200)), None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def label_image(img, text):
    if img is None:
        img = np.zeros((250, 400, 3), dtype=np.uint8)
    labeled = cv2.resize(img, (400, 250))
    cv2.rectangle(labeled, (0, 0), (400, 30), (255, 255, 255), -1)
    cv2.putText(labeled, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return labeled

def detect_and_display(real_folder="real_notes", test_folder="test_notes", min_match_threshold=45):
    real_notes = load_real_notes(real_folder)
    if not real_notes:
        print("No real notes found in", real_folder)
        return

    for file in os.listdir(test_folder):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        test_path = os.path.join(test_folder, file)
        test_gray = cv2.imread(test_path, 0)
        test_color = cv2.imread(test_path)

        if test_gray is None or test_color is None:
            print(f"Could not read {file}")
            continue

        match_scores = {}
        for label, real_img in real_notes.items():
            score = get_orb_match_score(test_gray, real_img)
            match_scores[label] = score

        best_label = max(match_scores, key=match_scores.get)
        best_score = match_scores[best_label]

        # Optional: print all scores for debugging
        print(f"\n{file} — Matching scores:")
        for label, score in match_scores.items():
            print(f"  Rs.{label}: {score}")

        # Get second-best score
        sorted_scores = sorted(match_scores.values(), reverse=True)
        second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

        # ✅ Apply strict logic
        is_real = best_score >= min_match_threshold and best_score > second_best_score * 1.5
        result_text = "Real" if is_real else "Fake"

        ref_img_path = os.path.join(real_folder, f"{best_label}.jpg")
        ref_img_color = cv2.imread(ref_img_path)

        labeled_ref = label_image(ref_img_color, f"Real Note: Rs.{best_label}")
        labeled_test = label_image(test_color, f"Result: {result_text}")

        combined = cv2.hconcat([labeled_ref, labeled_test])
        cv2.imshow(f"Checking: {file}", combined)
        print(f"→ Final Decision: {result_text} (Best: Rs.{best_label} Score: {best_score})")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_display()
