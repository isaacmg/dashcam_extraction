from transformers import OCRBertTokenizer, OCRBertForTokenClassification
import cv2


def load_model():
    model = OCRBertForTokenClassification.from_pretrained("microsoft/ocr-bert-base")
    tokenizer = OCRBertTokenizer.from_pretrained("microsoft/ocr-bert-base")
    return model, tokenizer

def get_frames(mp4_path):
    pass