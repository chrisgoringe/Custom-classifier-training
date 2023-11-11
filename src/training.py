from transformers import AutoImageProcessor, Trainer, TrainingArguments, DefaultDataCollator, AutoModelForImageClassification
from functools import partial
import datasets
from arguments import training_args, args

def _transform(example_batch, image_processor):
    inputs = image_processor( [x.convert("RGB") for x in example_batch["image"]], return_tensors="pt" )
    inputs["label"] = example_batch["label"]
    return inputs

def finetune( df ):      
        ds = datasets.Dataset.from_pandas(df)
        image_processor = AutoImageProcessor.from_pretrained( args['load_model'] )
        transform_with_processor = partial(_transform, image_processor=image_processor)

        prepared_ds = ds.cast_column("image", datasets.Image())
        prepared_ds = prepared_ds.with_transform(transform_with_processor)
        
        # mismatched_sizes happens because the pretrained models have 1000 categories and we have fewer
        model = AutoModelForImageClassification.from_pretrained(args['load_model'], 
                                                                num_labels=df["label"].nunique(), 
                                                                ignore_mismatched_sizes=True)

        Trainer(
            model=model,
            args=TrainingArguments( remove_unused_columns=False, push_to_hub=False, **training_args ),
            data_collator=DefaultDataCollator(),
            train_dataset=prepared_ds,
            eval_dataset=prepared_ds,
            tokenizer=image_processor,
        ).train()

        model.save_pretrained(args['save_model'])
        image_processor.save_pretrained(args['save_model'])