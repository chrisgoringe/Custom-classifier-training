from transformers import AutoImageProcessor, Trainer, TrainingArguments, DefaultDataCollator, AutoModelForImageClassification
import torch
from functools import partial
import datasets
from arguments import training_args, args


class CustomTrainer(Trainer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights, device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def _transform(example_batch, image_processor):
    inputs = image_processor( [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt" )
    inputs["label"] = example_batch["label"]
    return inputs

def prepare(df, transform_with_processor):
    ds = datasets.Dataset.from_pandas(df)
    prepared_ds = ds.cast_column("image", datasets.Image())
    prepared_ds = prepared_ds.with_transform(transform_with_processor)
    return prepared_ds

def finetune( df, eval_df, category_sizes ):      
        image_processor = AutoImageProcessor.from_pretrained( args['load_model'] )
        transform_with_processor = partial(_transform, image_processor=image_processor)

        prepared_ds = prepare(df, transform_with_processor)
        prepared_eval_ds = prepare(eval_df, transform_with_processor)

        
        # mismatched_sizes happens because the pretrained models have 1000 categories and we have fewer
        model = AutoModelForImageClassification.from_pretrained(args['load_model'], 
                                                                num_labels=df["label"].nunique(), 
                                                                ignore_mismatched_sizes=True)
        
        #pmw = PatchedModelWrapper(model, dim=32, dtype=model.dtype, device="cuda")
        if args['transfer_learning']:
            from .transfer_learning import TransferLearning
            assert model.base_model_prefix == 'vit'
            TransferLearning.prepare_model_for_transfer_learning(model, 
                    TransferLearning.createRule(replace=args['restart_layers'], thaw=args['thaw_layers']))
            
        if args['weight_category_loss']:
            weights = [ sum(category_sizes)/category_sizes[i] for i in range(len(category_sizes)) ]
        else:
            weights = [ 1.0 for _ in range(len(category_sizes)) ]

        CustomTrainer(
            weights = weights,
            model=model,
            args=TrainingArguments( remove_unused_columns=False, push_to_hub=False, **training_args ),
            data_collator=DefaultDataCollator(),
            train_dataset=prepared_ds,
            eval_dataset=prepared_eval_ds,
            tokenizer=image_processor,
        ).train()

        model.save_pretrained(args['save_model'])
        image_processor.save_pretrained(args['save_model'])