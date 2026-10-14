import glob, os, subprocess, sys, zipfile
import timm, torch

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
import gdown
out = gdown.download(id="1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR", output="official.bin", quiet=True)
zipfile.ZipFile(out).extractall("official")


def convert(src, arch, dst):
    obj = torch.load(src, map_location="cpu", weights_only=False)
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    ref = timm.create_model(arch, pretrained=False, num_classes=0).state_dict()
    vals = [(k, v) for k, v in sd.items() if not k.endswith("num_batches_tracked")]
    refk = [k for k in ref if not k.endswith("num_batches_tracked")]
    print(arch, len(vals), len(refk), vals[0][0], vals[-1][0], refk[0], refk[-1], flush=True)
    assert len(vals) == len(refk) and all(v.shape == ref[k].shape for (_, v), k in zip(vals, refk)), "shape/order mismatch"
    out = dict(ref)
    out.update({k: v for (_, v), k in zip(vals, refk)})
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    m.load_state_dict(out)
    torch.save(out, dst)
    print("saved", dst, flush=True)


convert("official/RadImageNet_pytorch/ResNet50.pt", "resnet50", "resnet50_rad.pt")
convert("official/RadImageNet_pytorch/DenseNet121.pt", "densenet121", "densenet121_rad.pt")
subprocess.run(["rm", "-rf", "official", "official.bin"])
