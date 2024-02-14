from src.ap.feature_extractor import FeatureExtractor
import os
import torch
import matplotlib.pyplot as plt

class Args:
    directory = ""
    feature_extractor_model = ""

mse = torch.nn.MSELoss()

def compare_images(imgfilenames:list[str]):
    image_directory = Args.directory
    feature_extractor = FeatureExtractor.get_feature_extractor(pretrained=Args.feature_extractor_model, image_directory=image_directory, device="cuda")

    features = [feature_extractor.get_features_from_file(os.path.join(image_directory,f), caching=True) for f in imgfilenames]

    typical_delta =torch.abs(features[1]-features[0]) 
    edited_delta = torch.abs(features[2]-features[0]) 

    td = list(float(x) for x in typical_delta)
    ed = list(float(x) for x in edited_delta)

    td.sort(reverse=True)
    ed.sort(reverse=True)

    hwfm = lambda a, b : sum( x > a*b[0] for x in b )

    for f in (0.5, 0.6, 0.7, 0.8, 0.9):
        print("Values greater than {:>4.2}*maximum: unrelated images {:>4}, edited pair {:>4}".format(f, hwfm(f, td), hwfm(f, ed)))

    plt.plot(ed)
    plt.plot(td)
    plt.show()

    feature_extractor._save_cache()

def main():
    imgfilenames = [ "img1.png", "img2.png", "edited_img1.png",  ]
    compare_images(imgfilenames)

if __name__ == "__main__":
    main()