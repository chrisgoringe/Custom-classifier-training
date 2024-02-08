from src.ap.feature_extractor import TextFeatureExtractor
from src.ap.aesthetic_predictor import AestheticPredictor
from arguments import get_args, args
import torch

def main():
    get_args(aesthetic_model=True)
    tfe = TextFeatureExtractor(pretrained=args['clip_model'])
    ap = AestheticPredictor.from_pretrained(pretrained=args['load_model_path'], explicit_nof=tfe.number_of_features)
    ap.eval()

    for text in ("ugly old man", "energetic dog", "sunset"):
        score = float(ap(tfe.get_text_features_tensor(text)))
        print (f"{text} {score}")

if __name__=="__main__":
    with torch.no_grad():
        main()