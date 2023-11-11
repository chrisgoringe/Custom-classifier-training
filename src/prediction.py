from transformers import ViTImageProcessor,  EfficientNetImageProcessor
from transformers import AutoModelForImageClassification
import torch

import pandas as pd
import numpy as np
import datasets

from arguments import args, evaluation_args

def _get_feature_extractor():
    if 'efficient' in args['base_model']:
        return EfficientNetImageProcessor.from_pretrained( args['load_model'])
    return ViTImageProcessor.from_pretrained( args['load_model'])

def _extract_probabilities_image(model, feature_extractor, col_name):
    """Utility to compute probabilites for images."""
    device = model.device

    def pp(batch):
        images = batch[ col_name ]  
        for i in range(len(images)):
            if images[i].mode != "RGB":
                images[i] = images[i].convert("RGB")
        inputs = feature_extractor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = ( torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy() )

        return {"probabilities": probabilities}

    return pp

def _generate_image_probabilites( image_paths ):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForImageClassification.from_pretrained( args['load_model'], output_hidden_states=True ).to(device)
    feature_extractor = _get_feature_extractor()

    df = pd.DataFrame(data={"image": image_paths})
    dataset = datasets.Dataset.from_pandas(df).cast_column("image", datasets.Image())

    extract_fn = _extract_probabilities_image(model, feature_extractor, "image")

    updated_dataset = dataset.map(
        extract_fn,
        batched=True,
        batch_size=evaluation_args['per_device_eval_batch_size'],
        remove_columns="image",
    ) 

    df_updated = updated_dataset.to_pandas()
    return np.array( [ emb.tolist() if emb is not None else None for emb in df_updated["probabilities"].values ] )

def predict( image_paths, labels ):
    probs = _generate_image_probabilites( image_paths ).tolist()
    preds = np.argmax(probs, axis=1).tolist()
    return preds, probs, list([probs[i][x] for i,x in enumerate(labels)])

