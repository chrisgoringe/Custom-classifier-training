
args = {
    # path to the top level image directory
    "top_level_image_directory" : r"training4", 
    # where to save the model
    "save_model"                : r"training4\vitH_model.safetensors",
    # the scores to train from (relative to tlid)
    "scorefile"                 : "scores.json",

    # three additional (optional) output files: the scores as predicted by the model, the errors (scores - model_scores), and the split (train/test) (relative to tlid)
    "model_scorefile"           : "model_scores.json",
    "error_scorefile"           : "error_scores.json",
    "splitfile"                 : "split.json",

# Feature extraction model. This is a list, or a string; if there are multiple entries the features are concatenated
# Default is laion/CLIP-ViT-H-14-laion2B-s32B-b79K, which has been resaved in torch.half format as ChrisGoringe/vitH16
#
# SDXL uses [openai/clip-vit-large-patch14, laion/CLIP-ViT-bigG-14-laion2B-39B-b160k] (mostly the second?)
# see https://github.com/huggingface/diffusers/blob/v0.26.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L149
#
# SD1.5 uses [openai/clip-vit-large-patch14]
# see https://github.com/huggingface/diffusers/blob/v0.26.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L312
#
# Others include:
# apple/aim-600M, apple/aim-1B, apple/aim-3B, apple/aim-7B, laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    "feature_extractor_model"   : "ChrisGoringe/vitH16", 

# loss model. 'nll', 'mse' or 'ab'. nll is negative log likelihood, which returns prediction and a sigma
    "loss_model"                : 'mse',
# should mse be calculated when loss is ab (or nll), and ab when loss is mse (or nll)
    "include_other_evaluations" : False,
# What defines "good" for a model when comparing between training runs? [full|train|eval]_[ab|mse|nll]
    "parameter_for_scoring"     : 'eval_mse',   
# what fraction of images to reserve as test images (when training), and a random seed for picking them
    "fraction_for_test"         : 0.25,
    "test_pick_seed"            : 42,        
}

aesthetic_model_extras = {
    "final_layer_bias" : False,
}

metaparameter_args = {
    "name"              : "vitH",     
    "meta_trials"       : 200,
    "sampler"           : "CmaEs",      # CmaEs, random, QMC.  CmaEs seems to work best

    # Each of these is a tuple (min, max) or a value.
    "num_train_epochs"   : (5, 50),
    "warmup_ratio"       : (0.0, 0.2),
    "log_learning_rate"  : (-3.5, -1.5),
    "half_batch_size"    : (1, 50),            

    # A list, each element is either a tuple (min, max) or a value
    "dropouts"           : [ (0.0, 0.8), (0.0, 0.8), ],
    "hidden_layers"      : [ (10, 1000), (10, 1000), ],
}

# passed to the constructor of the Trainer - https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer
trainer_extras = {}  

# training_args are passed directly into the TrainingArguments object.
# Below are the most common of the 101 arguments available
# see https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = {
    "lr_scheduler_type"             : "cosine",
    "per_device_eval_batch_size"    : 2000,     
    "save_strategy"                 : "no",
    "evaluation_strategy"           : "no",
    "output_dir"                    : "out",
}

### CALCULATED ARGUMENTS ###

args['direction']='maximize' if args['loss_model']=='ab' else 'minimize'
args['measures'] = {args['loss_model'],}
if args['include_other_evaluations']:
    args['measures'].add('ab')
    args['measures'].add('mse')
args['measures'] = list(args['measures'])

training_args["metric_for_best_model"] = "ab" if args['loss_model']=="ab" else "loss"

### ###

def show_args():
    for d in [args, aesthetic_model_extras, metaparameter_args, trainer_extras, training_args]:
        for a in d:
            print("{:>30} : {:<40}".format(a, str(d[a])))

class MetaRangeProcessor():
    def __init__(self):
        self.any_ranges = False

    def meta(self, mthd, label:str, rng:tuple|list):
        if isinstance(rng,(tuple,list)):
            self.any_ranges = True
            return mthd(label, *rng)
        else:
            return rng
        
    def meta_list(self, mthd, label:str, rnges:tuple|list):
        result = []
        for i, rng in enumerate(rnges):
            result.append(self.meta(mthd, f"{label}_{i}", rng))
        return result
