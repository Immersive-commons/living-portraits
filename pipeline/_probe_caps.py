"""throwaway capability probe for the img2vid prototype build (run on hil)."""
import sys, os, glob, importlib, pathlib

print("PY", sys.version.split()[0])

try:
    import torch
    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        cc = torch.cuda.get_device_capability(0)
        print("compute_capability", cc, "(7,5)=Turing,fp16-only" if cc == (7, 5) else "")
        free, total = torch.cuda.mem_get_info()
        print("vram_free_GB %.2f total_GB %.2f" % (free / 1e9, total / 1e9))
except Exception as e:
    print("torch ERR", repr(e))

for mod in ["diffusers", "transformers", "accelerate", "imageio", "imageio_ffmpeg", "PIL", "safetensors"]:
    try:
        m = importlib.import_module(mod)
        print("mod", mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print("mod", mod, "MISSING", repr(e))

try:
    import diffusers
    for name in ["AnimateDiffPipeline", "AnimateDiffSparseControlNetPipeline",
                 "AnimateDiffVideoToVideoPipeline", "StableVideoDiffusionPipeline",
                 "MotionAdapter", "I2VGenXLPipeline"]:
        print("pipe", name, hasattr(diffusers, name))
except Exception as e:
    print("diffusers introspect ERR", repr(e))

# ffmpeg for mp4 export
try:
    import imageio_ffmpeg
    print("ffmpeg_exe", imageio_ffmpeg.get_ffmpeg_exe())
except Exception as e:
    print("ffmpeg MISSING", repr(e))

# where did the portraits get generated?
for f in glob.glob(r"C:\living-portraits\**\generate*.py", recursive=True):
    print("found_generate", f)

# what SD/video models are already cached?
home = os.environ.get("HF_HOME") or os.path.expanduser(r"~\.cache\huggingface")
hub = pathlib.Path(home) / "hub"
print("HF_hub", hub, "exists", hub.exists())
if hub.exists():
    for d in sorted(hub.glob("models--*")):
        print("  cached", d.name)
