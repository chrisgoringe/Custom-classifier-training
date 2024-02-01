from _old_source_files.database import Database
import os, shutil, json, random

args = {
    # Where are the images?
    'top_level_image_directory':r"SOURCE/DIRECTORY",

    # export directory
    'export_directory':r"OUTPUT/DIRECTORY",
    
    # minimum score to export (use None for exporting all, or (eg) 1.0 for only images scoring > 1.0
    'minimum_score' : None,

    # filename format - available fields are ( filename, extension, rank, score, randomname )
    # randomname is an 8 character string of [a-z]
    # this is in the python string.format() form\
    # below are some examples, uncomment one of them or write your own
    'filename_format' : "{rank:0>5}_{score:0>6.3f}_{randomname}{extension}",
    #'filename_format' : "{rank:0>5}_{score:0>6.3f}_{filename}{extension}",

}

def random_name():
    return "".join( random.choices('abcdefghijklmnopqrstuvwxyz',k=8) )

def export():
    assert os.path.exists(args['top_level_image_directory']), f"{args['top_level_image_directory']} not found"
    d = Database(args['top_level_image_directory'])
    list = d.sorted_list(best_first=True)
    if not os.path.exists(args['export_directory']): os.makedirs(args['export_directory'])
    minimum = args['minimum_score'] if args['minimum_score'] is not None else float('-inf')
    new_scorefile = {}
    for rank, (filename, score, test_count) in enumerate(list):
        if score > minimum:
            details = {'rank':rank+1,
                       'filename':os.path.splitext(filename)[0],
                       'extension':os.path.splitext(filename)[1],
                       'score':score,
                       'randomname':random_name(),
            }
            old_filepath = os.path.join(args['top_level_image_directory'], filename)
            if os.path.exists(old_filepath):
                new_filename = args['filename_format'].format(**details)
                new_filepath = os.path.join(args['export_directory'], new_filename)
                if not os.path.exists(os.path.dirname(new_filepath)): os.makedirs(os.path.dirname(new_filepath))
                shutil.copyfile(old_filepath, new_filepath)
                new_scorefile[new_filename] = (score, test_count)
                print(f"Saved {new_filename}")
            else:
                print(f"{old_filepath} not found")
    new_scorefile['#meta#'] = d.meta
    with open(os.path.join(args['export_directory'],'score.json'), 'w') as f:
        print(json.dumps(new_scorefile, indent=2),file=f)

if __name__=='__main__':
    export()