
import os, zipfile, argparse, sys, nltk, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionImageVariationPipeline
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
try:
    import psutil
except ImportError:
    psutil = None
try:
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
except ImportError:
    KNeighborsClassifier = None
    XGBClassifier = None
TOKENIZERS = ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4']
for pkg in TOKENIZERS:
    nltk.download(pkg, quiet=True)
def check_system_resources():
    if psutil is None:
        print("psutil not installed, skipping resource check.")
        return True
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    print(f"Available Memory: {mem.available/(1024*1024):.2f} MB")
    print(f"CPU Usage: {cpu}%")
    return mem.available >= 2*1024**3
def dynamic_adjustment(prompt):
    tokens = nltk.word_tokenize(prompt)
    return (5,7.0) if len(tokens)<5 else (50,10.0)
def classify_image(image_tensor):
    arr = image_tensor.view(-1).cpu().numpy()
    if KNeighborsClassifier and XGBClassifier:
        knn = KNeighborsClassifier(n_neighbors=3)
        xgb = XGBClassifier()
        Xd = np.random.rand(100, arr.shape[0])
        yd = np.random.randint(0,2,100)
        try:
            knn.fit(Xd,yd)
            xgb.fit(Xd,yd)
            pred_knn = knn.predict([arr])
            pred_xgb = xgb.predict([arr])
            print(f"KNN prediction: {pred_knn}, XGB prediction: {pred_xgb}")
            return pred_knn, pred_xgb
        except Exception as e:
            print(f"Classification error: {e}")
    else:
        print("Skipping classification: dependencies missing.")
    return None, None
def modify_image(image_path,prompt,strength,num_variations,resize_option,custom_size=None,batch_size=1,apply_custom=False):
    out_dir = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not check_system_resources():
        batch_size = max(1,batch_size//2)
        num_variations = max(1,num_variations//2)
        print(f"Adjusted batch_size={batch_size}, num_variations={num_variations}")
    try:
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "Savanthgc/Image-Transformation-model-Improved",revision="main",torch_dtype=torch.float32,use_safetensors=True
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return 0
    size = custom_size if resize_option=='custom' and custom_size else (224,224)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size,interpolation=InterpolationMode.BICUBIC),
        transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])
    ])
    inp = tform(img).to(device).unsqueeze(0)
    classify_image(inp)
    if apply_custom:
        inp = transforms.RandomHorizontalFlip()(inp)
    outputs = []
    for _ in range(batch_size):
        steps,default_guidance = dynamic_adjustment(prompt)
        try:
            guidance = float(strength)
        except Exception:
            guidance = default_guidance
        try:
            result = pipe(inp,num_inference_steps=steps,guidance_scale=guidance)
            images = result.images
        except Exception as e:
            print(f"Pipeline error: {e}")
            continue
        outputs.extend(images[:num_variations])
    if not outputs:
        print("No images generated.")
        return 0
    for idx,out_img in enumerate(outputs,1):
        path = os.path.join(out_dir,f"result_{idx}.jpg")
        out_img.save(path)
        print(f"Saved {path}")
    if len(outputs)>1:
        try:
            collages = [transforms.ToTensor()(i) for i in outputs]
            save_image(torch.stack(collages),os.path.join(out_dir,"collage.jpg"),nrow=5)
            print(f"Saved {os.path.join(out_dir,'collage.jpg')}")
        except Exception as e:
            print(f"Error saving collage: {e}")
    return len(outputs)
def zip_outputs(zip_name="all_variations.zip"):
    folder = os.getcwd()
    zip_path = os.path.join(folder,zip_name)
    try:
        with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(folder):
                if (f.startswith("result_") and f.endswith(".jpg")) or f=="collage.jpg":
                    zf.write(os.path.join(folder,f),f)
        print(f"Created zip archive at {zip_path}")
        return zip_path
    except Exception as e:
        print(f"Error creating zip: {e}")
        return ""
def main():
    parser = argparse.ArgumentParser(description="Console Image Variation Tool")
    parser.add_argument("-i","--image_path",help="Input image file path")
    parser.add_argument("-p","--prompt",default="",help="Variation prompt")
    parser.add_argument("-s","--strength",type=float,default=7.5,help="Guidance strength")
    parser.add_argument("-n","--num_variations",type=int,default=1,help="Number of variations")
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size")
    parser.add_argument("--resize_option",choices=["default","custom"],default="default")
    parser.add_argument("--custom_width",type=int)
    parser.add_argument("--custom_height",type=int)
    parser.add_argument("--apply_custom",action="store_true")
    parser.add_argument("--download_all",action="store_true")
    args = parser.parse_args()
    if not any([args.image_path,args.download_all]):
        parser.print_help()
        sys.exit(1)
    if args.download_all:
        zip_outputs()
    if args.image_path:
        custom_size = None
        if args.resize_option=="custom":
            if args.custom_width is None or args.custom_height is None:
                parser.error("When using custom resize, specify --custom_width and --custom_height")
            custom_size=(args.custom_width,args.custom_height)
        count = modify_image(args.image_path,args.prompt,args.strength,args.num_variations,args.resize_option,custom_size,args.batch_size,args.apply_custom)
        print(f"Generated {count} image variation(s)")
if __name__=="__main__":
    main()
