from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.feature_extractor import FeatureExtractor
from src.ap.image_scores import ImageScores
import torch, os, math
from safetensors.torch import save_file
from tqdm import tqdm

'''
For the given image directory (which is assumed to have been AB scored), create a nudge file.
If load_model_path is specified, use the scores given by the model, otherwise use the AB scores
'''

args = {
    'directory' : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output",
    'clip_model'                : ["openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"], 
    'load_model_path'           : None,
    "input_size"                : 2048,

    'weighting'        : 'tanh',
    'weighting_params' : None,
}
def make_weighter(scores_list:list):
    w = None
    if args['weighting']=='alpha':
        p = [ s for s in scores_list if s>0 ]
        n = [ s for s in scores_list if s<=0 ]
        alpha = (len(scores_list) - sum(n))/sum(p)
        w = lambda a : (a/len(scores_list))*(alpha if a>0 else 1)
        print(f"alpha = {alpha}")
    elif args['weighting']=='top_n' and args['weighting_params']:
        scores_list.sort(reverse=True)
        n = args['weighting_params']
        threshold = scores_list[n-1]
        while scores_list[n]==scores_list[n-1]: n += 1
        w = lambda a : (1/n) if a>=threshold else 0
        print(f"Top {n} will be used")
    elif args['weighting']=='tanh':
        sumtanhed = sum( list( 1+math.tanh(s) for s in scores_list ) )
        w = lambda a : (1+math.tanh(a))/sumtanhed

    assert w is not None, "No valid weighting"
    return w

    

def make_nudge_from_scores():
    dir = args['directory']
    scores:ImageScores = ImageScores.from_scorefile(dir)
    if args['load_model_path']:
        ap = AestheticPredictor(clipper=clipper, pretrained=args['load_model_path'], input_size=args['input_size'])
        scores = ImageScores.from_evaluator(ap.evaluate_file, scores.image_files(), dir)
    scores_dict = scores.scores_dictionary(normalised=True)
    weighter = make_weighter(list(scores_dict[f] for f in scores_dict))

    clipper = FeatureExtractor.get_feature_extractor(image_directory=dir, pretrained=args[f"clip_model"])
    clipper.precache(list(os.path.join(args['directory'],f) for f in scores_dict))
    
    nudge = torch.zeros(args[f"input_size"]).cuda()
    for f in tqdm(scores_dict):
        features = clipper.get_features_from_file(os.path.join(args['directory'],f))
        nudge = nudge + features * weighter(scores_dict[f])

    save_file({'nudge':nudge}, os.path.join(args['directory'], 'nudge.safetensors'))

def make_nudge_from_image(f:str):
    dir = args['directory']
    clipper = FeatureExtractor.get_feature_extractor(image_directory=dir, pretrained=args[f"clip_model"])
    nudge = clipper.get_features_from_file(os.path.join(args['directory'],f))
    save_file({'nudge':nudge}, os.path.join(args['directory'], os.path.splitext(f)[0]+".safetensors"))

if __name__=='__main__':
    #make_nudge_from_scores()
    f = r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\j6p_ads-corporate\ads-corporate_00003_.png"
    make_nudge_from_image(f)
