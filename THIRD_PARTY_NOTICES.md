# Third-party code and models

Vendored code (`third_party/`, with small compatibility patches noted in the README):

| Component | Source | License |
|-----------|--------|---------|
| Segment-Any-Motion (SegAnyMo) | https://github.com/nnanhuang/SegAnyMo | MIT |
| SAM 2 (modified copy shipped by SegAnyMo) | https://github.com/facebookresearch/sam2 | Apache-2.0 |
| TAPNet / BootsTAPIR (PyTorch port, via SegAnyMo) | https://github.com/google-deepmind/tapnet | Apache-2.0 |
| DINOv2 (via SegAnyMo) | https://github.com/facebookresearch/dinov2 | Apache-2.0 |
| MiniMax-Remover (pipeline + transformer code) | https://github.com/zibojia/MiniMax-Remover | no license file in the upstream repository — check with the authors before redistribution |
| CogVideoX transformer / I2V pipeline (modified from diffusers) | https://github.com/huggingface/diffusers, https://github.com/THUDM/CogVideo | Apache-2.0 |

Pre-trained weights downloaded at setup time (not redistributed here):

| Model | Hub id | License |
|-------|--------|---------|
| InternVL2.5-1B (captioning) | `OpenGVLab/InternVL2_5-1B` | MIT |
| Depth-Anything-V2-Large | `depth-anything/Depth-Anything-V2-Large-hf` | CC-BY-NC-4.0 (Small/Base variants are Apache-2.0) |
| BootsTAPIR v2 | `dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt` | Apache-2.0 |
| DINOv2 ViT-B/14 | torch.hub `facebookresearch/dinov2` | Apache-2.0 |
| moseg (SegAnyMo classifier) | `Changearthmore/moseg` | MIT (SegAnyMo) |
| SAM 2 hiera-large | `segment_anything_2/072824/sam2_hiera_large.pt` | Apache-2.0 |
| MiniMax-Remover (Wan2.1-1.3B based) | `zibojia/minimax-remover` | see upstream |
| CogVideoX-5b-I2V (Composer base) | `THUDM/CogVideoX-5b-I2V` | CogVideoX License |
| ViCLIP-L/14 (metrics M1/M2/M5) | `OpenGVLab/ViCLIP-L-14-hf` | MIT (InternVid) |
| Video Swin-B Kinetics-400, RAFT-large (metrics M3/M4) | torchvision model zoo | BSD-3 (torchvision) |

Datasets: OpenVid-1M (`nkp37/OpenVid-1M`, CC-BY-4.0; source clips carry their own licenses), DAVIS 2017
(research use), Panda-70M (CSV metadata, YouTube videos).
