from src.ap.aesthetic_predictor import AestheticPredictor
from src.ap.clip import CLIP
from src.ap.image_scores import ImageScores
import torch, os
from safetensors.torch import save_file

'''
For the given image directory (which is assumed to have been AB scored), create a nudge file.
If load_model_path is specified, use the scores given by the model, otherwise use the AB scores

I think you probably have to use openai/clip-vit-large-patch14 because that's the one SDXL uses
'''

args = {
    'top_level_image_directory' : r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output",
    'clip_model'                : "openai/clip-vit-large-patch14",   # DONT CHANGE
    'load_model_path'           : None,
    "input_size"                : 768,
}

def make_nudge():
    dir = args['top_level_image_directory']
    scores:ImageScores = ImageScores.from_scorefile(dir)
    clipper=CLIP.get_clip(image_directory=dir, pretrained=args['clip_model'])
    if args['load_model_path']:
        ap = AestheticPredictor(clipper=clipper, pretrained=args['load_model_path'])
        scores = ImageScores.from_evaluator(ap.evaluate_file, scores.image_files(), dir)
    scores_dict = scores.scores_dictionary(normalised=True)
    nudge = torch.zeros((args['input_size'])).to('cuda')
    for f in scores_dict:
        features:torch.Tensor = clipper.prepare_from_file(os.path.join(args['top_level_image_directory'],f))
        nudge = nudge + features * scores_dict[f] / len(scores_dict)
    save_file({'nudge':nudge}, os.path.join(args['top_level_image_directory'], 'nudge.safetensors'))

if __name__=='__main__':
    make_nudge()
