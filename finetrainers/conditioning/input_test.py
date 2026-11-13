import torch
from diffusers import LTXPipeline
from diffusers import LTXVideoTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.utils import testing_utils
from finetrainers.dataset import ImageOrVideoDataset
from finetrainers.dataset import BucketSampler, ImageOrVideoDatasetWithResizing, PrecomputedDataset
from finetrainers.ltx_video.full_finetune_condition import collate_fn_t2v_cond
from finetrainers.conditioning import condition_latents_prepare,post_conditioned_latent_patchify
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.ltx import LTXPipeine 

testing_utils.enable_full_determinism()
device = "cuda"

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video", torch_dtype=torch.bfloat16,
    # transformer=transformer
).to(device)
dtype = torch.bfloat16

vae_spatial_compression_ratio = 32
vae_temporal_compression_ratio = 8
transformer_temporal_patch_size = 1
transformer_spatial_patch_size = 1

video_processor = VideoProcessor(vae_scale_factor=vae_spatial_compression_ratio)

generator=torch.Generator(device=device).manual_seed(42),

vae = pipe.vae

dataset = ImageOrVideoDataset(
    data_root="${workspaceFolder}/video-dataset-pose-test",
    caption_column="prompts.txt",
    video_column="videos.txt", 
    resolution_buckets=[(96,512,768), (168,512,768)],
    id_token="",
    pose_column="poses.txt"
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    sampler=BucketSampler(dataset, batch_size=1, shuffle=True),
    collate_fn=collate_fn_t2v_cond,
    num_workers=1,
    pin_memory=True,
)

for step, batch in enumerate(dataloader):
    videos = batch["videos"]
    prompts = batch["prompts"]
    poses = batch["poses"]
    # generate video across all frames
    img_refs = batch["img_refs"]


# single img video ref ...
video_tensor = dataset._preprocess_video_image_reference_video()

video_latent = condition_latents_prepare.prepare_latents_for_conditioning(image_or_video=video_tensor)

tokens = condition_latents_prepare.post_conditioned_latent_patchify()


latents = LTXPipeine._unpack_latents(
                tokens,
                latent_num_frames,
                latent_height,
                latent_width,
                transformer_spatial_patch_size,
                transformer_temporal_patch_size,
            )

latents = LTXPipeine._denormalize_latents(
    latents, vae.latents_mean, vae.latents_std, vae.config.scaling_factor
)



video = vae.decode(latents, None, return_dict=False)[0]
video_processor.postprocess_video(video,output_type="pil")