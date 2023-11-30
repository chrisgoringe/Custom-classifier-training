import os
from arguments import args, get_args
from src.time_context import Timer
from src.ap.clip import CLIP
from src.ap.aesthetic_predictor import AestheticPredictor
from src.data_holder import DataHolder
from src.ap.dataset import QuickDataset
import torch
from renumics import spotlight

def main():
    get_args(aesthetic_model=True)
    top_level_images = args['top_level_image_directory']

    with Timer('load models'):
        clipper = CLIP(pretrained=args['clip_model'], image_directory=top_level_images)
        predictor = AestheticPredictor(pretrained=args['load_model_path'], clipper=clipper)

    with Timer('Prepare images'):
        data = DataHolder(top_level=top_level_images, save_model_folder=args['save_model'], use_score_file=args['use_score_file'])
        df = data.get_dataframe()
        ds = QuickDataset(df)
        with Timer('CLIP'):
            df['features'] = [clipper.prepare_from_file(f, device="cpu") for f in df['image']]
            clipper.save_cache()
        df['score'] = [float(l) for l in df['label_str']]

    with Timer('Evaluate images'):
        with torch.no_grad():
            ds.update_prediction(predictor)

        spotlight.show(ds._df)

if __name__=='__main__':
    main()