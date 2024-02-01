import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os, json

def create_probability_calculator(model_directory, key, labels=[]):
    model = AutoModelForImageClassification.from_pretrained(model_directory, output_hidden_states=True)
    model.to('cuda')
    feature_extractor = AutoImageProcessor.from_pretrained(model_directory)
    with open(os.path.join(model_directory,'categories.json')) as f: labels = json.load(f)['categories']

    def calculate_probabilities(image, name):
        if image.mode != "RGB": image = image.convert("RGB")
        with torch.no_grad():
            inputs = feature_extractor(images=[image], return_tensors="pt").to('cuda')
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            return (probs[key], name, { labels[i]:"{:>6.2f}%".format(100*float(probs[i])) for i in range(len(labels)) })
    return calculate_probabilities

def main():
    model_directory = "C:\\Users\\chris\\Documents\\GitHub\\ComfyUI_windows_portable\\ComfyUI\\models\\customclassifier\\yg"
    image_directory = "C:\\Users\\chris\\Documents\\GitHub\\ComfyUI_windows_portable\\ComfyUI\\output\\thetwopercent"
    sort_category = 0

    cp = create_probability_calculator(model_directory, sort_category)
    
    results = []
    for filename in os.listdir(image_directory):
        try:
            im = Image.open(os.path.join(image_directory, filename))
            results.append(cp(im,filename))
        except:
            pass
    results.sort()
    for _, f, p in results:
        print("{:>40} : {:<60}".format(f, str(p)))



if __name__=='__main__':
    main()