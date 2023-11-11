import os
from src.data_holder import DataHolder
from src.training import finetune
from src.prediction import predict
from src.time_context import Timer
from arguments import args, check_arguments

def main():
    check_arguments()

    with Timer("load"):
        dh = DataHolder(args['top_level_image_directory'], args['save_model'], args['fraction_for_test'], args['test_pick_seed'])
        df = dh.get_dataframe()
        
    if args['mode'] == 'train':
        with Timer("train"): 
            finetune( df[df["split"] == "train"] )

    if args['mode'] == 'evaluate' or args['mode'] == 'spotlight':
        with Timer("predict"):
            if args['evaluate_test_only']: df = df[df["split"]=="test"]
            df["prediction"], df["probs"], scores = predict( df["image"].values )
            count = len(df)
            correct = sum(df["prediction"]==df["label"])
            print("{:>3}/{:>3} ({:>6.2f}%) correct".format(correct, count, 100*correct/count))
            print("Average probability assigned to correct label {:>6.2f}%".format(100*sum(scores)/count))
            n_labels = df["label"].nunique()
            print("{:>2} options, so random guesses would get {:>6.2f}%".format(n_labels, 100/n_labels))

        if args['mode'] == 'spotlight':
            try:
                from renumics import spotlight
            except:
                print("You need to pip install spotlight to use it")
                return
            spotlight.show(df, folder=os.getcwd(), dtype={"embedding": spotlight.Image},)

if __name__=="__main__":
    main()   
