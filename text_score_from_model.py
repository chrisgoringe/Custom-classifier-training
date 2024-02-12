from src.ap.feature_extractor import TextFeatureExtractor
from src.ap.aesthetic_predictor import AestheticPredictor
import torch

class Args:
    text_feature_extractor_model = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    model = r"training4\clip_g_model.safetensors"

def main():
    tfe = TextFeatureExtractor(pretrained=Args.text_feature_extractor_model)
    ap = AestheticPredictor.from_pretrained(pretrained=Args.model, explicit_nof=tfe.number_of_features)
    ap.eval()

    while text := input('text to evaluate: '):
        score = float(ap(tfe.get_text_features_tensor(text)))
        print (f"{text} {score}")

if __name__=="__main__":
    with torch.no_grad():
        main()