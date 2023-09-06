import torch
import random


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


class VAEDecodePipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE","VAE",)
    FUNCTION = "decode"

    CATEGORY = "braintacles/latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]), vae, )


class VAEDecodeTiledPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ), "vae": ("VAE", ),
                             "tile_size": ("INT", {"default": 1024, "min": 320, "max": 4096, "step": 64})
                             }}
    RETURN_TYPES = ("IMAGE","VAE",)
    FUNCTION = "decode"

    CATEGORY = "braintacles/latent"

    def decode(self, vae, samples, tile_size):
        return (vae.decode_tiled(samples["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, ), vae, )


class VAEEncodePipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("LATENT","VAE",)
    FUNCTION = "encode"

    CATEGORY = "braintacles/latent"

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def encode(self, vae, pixels):
        pixels = self.vae_encode_crop_pixels(pixels)
        t = vae.encode(pixels[:, :, :, :3])
        return ({"samples": t}, vae, )


class VAEEncodeTiledPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pixels": ("IMAGE", ), "vae": ("VAE", ),
                             "tile_size": ("INT", {"default": 1024, "min": 320, "max": 4096, "step": 64})
                             }}
    RETURN_TYPES = ("LATENT","VAE",)
    FUNCTION = "encode"

    CATEGORY = "braintacles/latent"

    def encode(self, vae, pixels, tile_size):
        pixels = VAEEncodePipe.vae_encode_crop_pixels(pixels)
        t = vae.encode_tiled(pixels[:, :, :, :3],
                             tile_x=tile_size, tile_y=tile_size, )
        return ({"samples": t}, vae, )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXL-Multi-IO": CLIPTextEncodeSDXL_Multi_IO,
    "CLIPTextEncodeSDXL-Pipe": CLIPTextEncodeSDXL_Pipe,
    "Empty Latent Image from Aspect-Ratio": EmptyLatentImageFromAspectRatio,
    "Random Find and Replace": RandomFindAndReplace,
    "VAE Decode Pipe": VAEDecodePipe,
    "VAE Decode Tiled Pipe": VAEDecodeTiledPipe,
    "VAE Encode Pipe": VAEEncodePipe,
    "VAE Encode Tiled Pipe": VAEEncodeTiledPipe,
}
