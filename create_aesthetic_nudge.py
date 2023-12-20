from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP
from src.ap.image_scores import ImageScores
import torch, os
from safetensors.torch import save_file
from tqdm import tqdm

'''
For the given image directory (which is assumed to have been AB scored), create a nudge file.
If load_model_path is specified, use the scores given by the model, otherwise use the AB scores

I think you probably have to use openai/clip-vit-large-patch14 because that's the one SDXL uses
'''

args = {
    'top_level_image_directory' : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output",
    'clip_model'                : ["openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"], 
    'load_model_path'           : None,
    "input_size"                : 2048,
}
def make_weighter(scores_dict):
    p = [ scores_dict[s] for s in scores_dict if scores_dict[s]>0 ]
    n = [ scores_dict[s] for s in scores_dict if scores_dict[s]<=0 ]
    alpha = (len(scores_dict) - sum(n))/sum(p)
    w = lambda a : (a/len(scores_dict))*(alpha if a>0 else 1)
    #print(f"TEST: {sum(w(scores_dict[s]) for s in scores_dict)}")
    return w
    

def make_nudge():
    dir = args['top_level_image_directory']
    scores:ImageScores = ImageScores.from_scorefile(dir)
    if args['load_model_path']:
        ap = AestheticPredictor(clipper=clipper, pretrained=args['load_model_path'])
        scores = ImageScores.from_evaluator(ap.evaluate_file, scores.image_files(), dir)
    scores_dict = scores.scores_dictionary(normalised=True)
    weighter = make_weighter(scores_dict)

    clipper = CLIP.get_clip(image_directory=dir, pretrained=args[f"clip_model"])
    nudge = torch.zeros(args[f"input_size"]).cuda()
    for f in tqdm(scores_dict):
        features = clipper.prepare_from_file(os.path.join(args['top_level_image_directory'],f))
        nudge = nudge + features * weighter(scores_dict[f])

    clipper.save_cache()
    save_file({'nudge':nudge}, os.path.join(args['top_level_image_directory'], 'nudge.safetensors'))

if __name__=='__main__':
    make_nudge()
