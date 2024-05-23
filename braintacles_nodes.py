import torch
import random
import comfy.samplers
import comfy.sample
import latent_preview

class CLIPTextEncodeSDXL_Multi_IO:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),
        },
            "optional": {
            "clip2": ("CLIP", ),
            "clip3": ("CLIP", ),
            "clip4": ("CLIP", ),
            "latent": ("LATENT", ),
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",
                    "CONDITIONING", "CONDITIONING", "LATENT")
    FUNCTION = "encode"

    CATEGORY = "braintacles/conditioning"

    @staticmethod
    def encode_with_clip(clip, text_g, text_l):
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooled

    def encode(self, clip, text_g, text_l, clip2=None, clip3=None, clip4=None, latent=None, **kwargs):
        width = 1024
        height = 1024
        if latent is not None:
            width = latent["samples"].shape[-1] * 8
            height = latent["samples"].shape[-2] * 8
        cond, pooled = self.encode_with_clip(clip, text_g, text_l)
        return_list = [
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
        ]
        if clip2 is not None:
            cond2, pooled2 = self.encode_with_clip(clip2, text_g, text_l)
            return_list[1] = [[cond2, {"pooled_output": pooled2, "width": width, "height": height,
                                       "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]]
        if clip3 is not None:
            cond3, pooled3 = self.encode_with_clip(clip3, text_g, text_l)
            return_list[2] = [[cond3, {"pooled_output": pooled3, "width": width, "height": height,
                                       "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]]
        if clip4 is not None:
            cond4, pooled4 = self.encode_with_clip(clip4, text_g, text_l)
            return_list[3] = [[cond4, {"pooled_output": pooled4, "width": width, "height": height,
                                       "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]]
        return (return_list[0], return_list[1], return_list[2], return_list[3], latent, )


class CLIPTextEncodeSDXL_Pipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),
        },
            "optional": {
            "refiner_clip": ("CLIP", ),
            "latent": ("LATENT", ),
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CLIP", "CLIP", "LATENT")
    RETURN_NAMES = ("conditioning", "refiner conditioning",
                    "CLIP", "refiner CLIP", "latent")
    FUNCTION = "encode"

    CATEGORY = "braintacles/conditioning"

    @staticmethod
    def encode_with_clip(clip, text_g, text_l):
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return cond, pooled

    def encode(self, clip, text_g, text_l, refiner_clip=None, latent=None, **kwargs):
        width = 1024
        height = 1024
        if latent is not None:
            width = latent["samples"].shape[-1] * 8
            height = latent["samples"].shape[-2] * 8
        cond, pooled = self.encode_with_clip(clip, text_g, text_l)
        return_list = [
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
            [[cond, {"pooled_output": pooled, "width": width, "height": height,
                     "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]],
        ]
        if refiner_clip is not None:
            cond2, pooled2 = self.encode_with_clip(
                refiner_clip, text_g, text_l)
            return_list[1] = [[cond2, {"pooled_output": pooled2, "width": width, "height": height,
                                       "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]]
        return (return_list[0], return_list[1], clip, refiner_clip, latent, )


class EmptyLatentImageFromAspectRatio:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"short_side": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                             "orientation": (["square", "landscape", "portrait", "random"],),
                             "aspect_ratio": ("STRING", {"default": "1:1"},),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }
                }
    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("samples", "width", "height", "aspect ratio")
    FUNCTION = "generate"

    CATEGORY = "braintacles/latent"

    def generate(self, short_side, orientation, aspect_ratio, batch_size=1, seed=0):
        if orientation == "random":
            random.seed(seed)
            orientation = random.choice(["square", "landscape", "portrait"])
        if orientation == "square" or aspect_ratio == "1:1":
            width = height = short_side
        elif orientation == "landscape":
            # short side is height
            height = short_side
            width = int(height * float(aspect_ratio.split(":")
                        [0]) / float(aspect_ratio.split(":")[1]))
        elif orientation == "portrait":
            # short side is width
            width = short_side
            height = int(width * float(aspect_ratio.split(":")
                         [0]) / float(aspect_ratio.split(":")[1]))

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        aspect_ratio_float = float(aspect_ratio.split(":")[0]) / float(aspect_ratio.split(":")[
            1]) if orientation == "landscape" else float(aspect_ratio.split(":")[1]) / float(aspect_ratio.split(":")[0])
        return ({"samples": latent}, width, height, aspect_ratio_float,)


class RandomFindAndReplace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "find": ("STRING", {"default": "String to Find & Replace", "multiline": False}),
                "choices": ("STRING", {"default": "Choices", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("prompt", "random_choice", "seed",)
    FUNCTION = "replace_prompt_with_random_line"

    CATEGORY = "braintacles/Prompt"

    def replace_prompt_with_random_line(self, prompt, find, choices, seed):
        lines = choices.split("\n")
        random.seed(seed)
        choice = random.choice(lines)
        prompt = prompt.replace(find, choice)
        return (prompt, choice, seed,)


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class IntervalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"modelA": ("MODEL",),
                    "modelB": ("MODEL",),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "interval": ("INT", {"default": 1, "min": 1, "max": 1000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positiveA": ("CONDITIONING", ),
                    "negativeA": ("CONDITIONING", ),
                    "positiveB": ("CONDITIONING", ),
                    "negativeB": ("CONDITIONING", ),
                    "latent_image": ("LATENT", )
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "braintacles/sampling"

    def sample(self, modelA, modelB, noise_seed, steps, interval, cfg, sampler_name, scheduler, positiveA, negativeA, positiveB, negativeB, latent_image, denoise=1.0):
        force_full_denoise = False
        disable_noise = False
        latest_latent = latent_image
        latest_model = "B"
        for i in range(0, steps, interval):
            if i>0:
                disable_noise = True
            print(f"Sampling Steps {i} to {i+interval} out of {steps} with noise {'enabled' if not disable_noise else 'disabled'} on model {latest_model}")
            latest_model = "A" if latest_model == "B" else "B"
            model = modelA if latest_model == "A" else modelB
            latest_positive = positiveA if latest_model == "A" else positiveB
            latest_negative = negativeA if latest_model == "A" else negativeB
            latest_latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, latest_positive, latest_negative, latest_latent, denoise=denoise, disable_noise=disable_noise, start_step=i, last_step=i+interval, force_full_denoise=force_full_denoise)[0]
        return (latest_latent, )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXL-Multi-IO": CLIPTextEncodeSDXL_Multi_IO,
    "CLIPTextEncodeSDXL-Pipe": CLIPTextEncodeSDXL_Pipe,
    "Empty Latent Image from Aspect-Ratio": EmptyLatentImageFromAspectRatio,
    "Random Find and Replace": RandomFindAndReplace,
    "Interval Sampler": IntervalSampler
}
