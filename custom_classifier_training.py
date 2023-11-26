import os
from src.data_holder import DataHolder
from src.cat.training import finetune
from src.cat.prediction import predict
from src.time_context import Timer 
from arguments import args,training_args,get_args

VERSION = "0.3"

def load_data():
    with Timer("load"):
        dh = DataHolder(args['top_level_image_directory'], args['save_model'], args['fraction_for_test'], args['test_pick_seed'])
        df = dh.get_dataframe()
        category_sizes = dh.sizes
        return df, category_sizes
    
def check_arguments():
    if 'load_model' not in args or not args['load_model']:
        assert 'train' in args['mode'], "If not training, need to specify a model to reload!"
        args['load_model']=args['base_model']

    training_args['output_dir'] = args['save_model']

    if 'train' in args['mode']:
        assert 'save_model' in args and args['save_model'], "Training needs a save_model location!"
    
def main():
    get_args(category_training=True)
    check_arguments()
    df, category_sizes = load_data()
        
    if 'train' in args['mode']:
        with Timer("train"): 
            finetune( df[df["split"] == "train"], df[df["split"] == "test"], category_sizes )

        args['load_model'] = args['save_model']

    if 'evaluate' in args['mode'] or 'spotlight' in args['mode']:
        with Timer("predict"):
            df["prediction"], df["probs"], scores = predict( df["image"].values, df["label"] )
            dft = df[df["split"]=="test"]
            
            count = len(dft)
            correct = sum(dft["prediction"]==dft["label"])
            print("In test set: {:>3}/{:>3} ({:>6.2f}%) correct".format(correct, count, 100*correct/count))
            correct = sum(df["prediction"]==df["label"])
            print("Overall set: {:>3}/{:>3} ({:>6.2f}%) correct".format(correct, count, 100*correct/count))
            print("Average probability assigned to correct label {:>6.2f}%".format(100*sum(scores)/count))
            n_labels = dft["label"].nunique()
            print("{:>2} options, so random guesses would get {:>6.2f}%".format(n_labels, 100/n_labels))

        if 'spotlight' in args['mode']:
            try:
                from renumics import spotlight
            except:
                print("You need to 'pip install renumics-spotlight' to use it")
                return
            try:
                spotlight.show(df, folder=os.getcwd(), dtype={"embedding": spotlight.Image},)
            except ConnectionResetError:
                pass

if __name__=="__main__":
    main()   
