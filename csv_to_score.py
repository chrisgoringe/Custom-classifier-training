import csv, json, os

# the json file will be created in the same place
csv_file = r"C:\Users\chris\Downloads\Peter_Mckinnon.csv"

# the columns count from 0
filename_column = 8
score_column = 7

# where will the images be placed relative to the resulting json file?
image_subfolder = "images"

def csv_to_json():
    json_file = os.path.join( os.path.dirname(csv_file), "score.json" )
    scores = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row)>max(filename_column,score_column) and row[filename_column] and row[score_column]:
                scores[os.path.join(image_subfolder,row[filename_column])] = [float(row[score_column]),100]
    scores['#meta#'] = { "source":"csv file" }
    with open(json_file,'w') as f:
        print(json.dumps(scores,indent=2), file=f)

if __name__=='__main__':
    csv_to_json()