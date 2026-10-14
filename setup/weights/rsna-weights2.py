import timm, torch

NAMES = {"coatnet_0_rw_224": "coatnet_0_rw_224.sw_in1k", "coatnet_rmlp_1_rw_224": "coatnet_rmlp_1_rw_224.sw_in1k",
         "efficientnet_b3": "efficientnet_b3.ra2_in1k", "tf_efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
         "seresnext50_32x4d": "seresnext50_32x4d.racm_in1k", "convnext_nano": "convnext_nano.in12k_ft_in1k"}
print(timm.__version__, torch.__version__)
for arch, tag in NAMES.items():
    try:
        m = timm.create_model(tag, pretrained=True, num_classes=0)
        torch.save(m.state_dict(), f"{arch}.pt")
        print(arch, sum(p.numel() for p in m.parameters()) / 1e6, m.num_features, flush=True)
    except Exception as e:
        print("FAIL", arch, e, flush=True)
