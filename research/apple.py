from PIL import Image

# pip install git+https://git@github.com/apple/ml-aim.git

from aim.torch.models import AIMForImageClassification
from aim.torch.data import val_transforms

img = Image.open(r"C:\Users\chris\Documents\GitHub\ComfyUI_windows_portable\ComfyUI\output\compare\albedo2_00067_.png")
model = AIMForImageClassification.from_pretrained("apple/aim-7B", cache_dir="../models/apple")
transform = val_transforms()

inp = transform(img).unsqueeze(0)
logits, features = model(inp)
pass
model.extract_features(inp, max_block_id=-1)  # or -2 etc...