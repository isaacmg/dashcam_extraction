import cv2


def load_model():
    model = OCRBertForTokenClassification.from_pretrained("microsoft/ocr-bert-base")
    tokenizer = OCRBertTokenizer.from_pretrained("microsoft/ocr-bert-base")
    return model, tokenizer
