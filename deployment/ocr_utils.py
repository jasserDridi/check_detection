from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from datetime import datetime
import re
import json
from word2number import w2n

# Load models once
yolo_model = YOLO("best.pt")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-stage1")

def clean_and_format_date(raw_date: str) -> str:
    cleaned = raw_date.replace(" ", "")
    try:
        parsed_date = datetime.strptime(cleaned, "%d%m%Y")
        month_map = {
            1: "Janv", 2: "Févr", 3: "Mars", 4: "Avr", 5: "Mai", 6: "Juin",
            7: "Juil", 8: "Août", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Déc"
        }
        return f"{parsed_date.day:02d} {month_map[parsed_date.month]} {parsed_date.year}"
    except Exception:
        return re.sub(r"(\d{2})(\d{2})(\d{4})", r"\1/\2/\3", cleaned)

def normalize_amount(amount: str) -> float:
    if not amount:
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", amount)
    try:
        return round(float(cleaned), 2)
    except:
        return 0.0

def words_to_number(amount_words: str) -> float:
    try:
        return round(w2n.word_to_num(amount_words.lower()), 2)
    except:
        return 0.0

def process_image(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    results = yolo_model(image_path)[0]
    label_map = results.names
    structured_data = {
        "image_path": image_path
    }

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = label_map[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_img = image.crop((x1, y1, x2, y2))

        pixel_values = processor(images=cropped_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if label == "signature":
            structured_data["signature"] = bool(predicted_text.strip())
        elif label == "date":
            structured_data["date"] = clean_and_format_date(predicted_text)
        elif label == "amount_in_numbers":
            cleaned = predicted_text.replace(" ", "").replace(",", "")
            cleaned = re.sub(r"^1-", "", cleaned)  # Remove "1-" from start
            structured_data["amount_in_numbers"] = cleaned
        elif label == "amount_in_words":
            structured_data["amount_in_words"] = predicted_text
        elif label == "payee":
            structured_data["payee"] = predicted_text

    # Post-processing
    raw_amount = structured_data.get("amount_in_numbers", "0")
    amount_number = normalize_amount(raw_amount)
    amount_words = structured_data.get("amount_in_words", "")
    converted_number = words_to_number(amount_words)
    signature = structured_data.get("signature", False)

    structured_data["converted_number"] = converted_number
    structured_data["status"] = signature and (amount_number == converted_number)

    return structured_data
