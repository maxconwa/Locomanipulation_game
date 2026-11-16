"""The policy's frozen encoders: the vision-language model on the head camera's image and the instruction, and the
wrist cameras' image encoder. Neither is trained; their weights are not in the policy's checkpoints.

The VLM (SmolVLM2-500M) reads the head image's lower part as one 512 x 512 image, its 64 image tokens, and the instruction, in its chat format;
its last layer's hidden states of all those tokens are the policy's task embedding. The wrist encoder (DINOv2 ViT-S/14)
gives each wrist image's patch tokens. The instruction's words are also given as the VLM's input embeddings
(embed_words), which the policy's keypoint encoder queries the images by.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

VLM_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
WRIST_NAME = "facebook/dinov2-small"
VLM_IMAGE_SIZE = 512
HEAD_CROP_ROWS = 224  # the head image's lower rows given to the VLM: the table (the upper third is the floor beyond)
VLM_IMAGE_TOKENS = 64
VLM_TOKENS = 88  # the sequences are padded to this: 64 image tokens, the chat format's 7 and an instruction's 15
VLM_DIM = 960
WORD_TOKENS = 16  # an instruction's own tokens (14 for the template), padded to this
HEAD_SMALL = (160, 320)  # rows, columns of the head image's lower part as the keypoint encoder gets it
WRIST_DIM = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenEncoders:
    def __init__(self, device: str, vlm_name: str = VLM_NAME, wrist_name: str = WRIST_NAME):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_name)
        self.vlm = AutoModelForImageTextToText.from_pretrained(vlm_name, torch_dtype=torch.float16).to(device).eval()
        self.wrist = AutoModel.from_pretrained(wrist_name, torch_dtype=torch.float16).to(device).eval()
        for model in (self.vlm, self.wrist):
            model.requires_grad_(False)
        self._prompts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _prompt(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The instruction's token ids and mask (VLM_TOKENS,), in SmolVLM's chat format with one un-split image."""
        if instruction not in self._prompts:
            text = ("<|im_start|>User:<fake_token_around_image><global-img>" + "<image>" * VLM_IMAGE_TOKENS
                    + "<fake_token_around_image>" + instruction + "<end_of_utterance>\n")
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > VLM_TOKENS:
                raise ValueError(f"Instruction too long for VLM_TOKENS ({len(ids)} tokens): {instruction!r}")
            pad = VLM_TOKENS - len(ids)
            self._prompts[instruction] = (
                torch.tensor(ids + [self.tokenizer.pad_token_id] * pad, device=self.device),
                torch.tensor([1] * len(ids) + [0] * pad, device=self.device))
        return self._prompts[instruction]

    @torch.no_grad()
    def embed_task(self, head_rgb: torch.Tensor, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """head_rgb (B, H, W, 3) uint8 (its lower HEAD_CROP_ROWS rows are used) and B instructions -> the VLM's tokens (B, VLM_TOKENS, VLM_DIM) float16 and
        their mask (B, VLM_TOKENS) bool (False: padding)."""
        x = head_rgb[:, -HEAD_CROP_ROWS:].permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(VLM_IMAGE_SIZE, VLM_IMAGE_SIZE), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - 0.5) / 0.5).half().unsqueeze(1)
        prompts = [self._prompt(text) for text in instructions]
        ids = torch.stack([p[0] for p in prompts])
        mask = torch.stack([p[1] for p in prompts])
        pixel_mask = torch.ones(len(ids), 1, VLM_IMAGE_SIZE, VLM_IMAGE_SIZE, dtype=torch.long, device=self.device)
        out = self.vlm(input_ids=ids, attention_mask=mask, pixel_values=x, pixel_attention_mask=pixel_mask,
                       output_hidden_states=True)
        return out.hidden_states[-1], mask.bool()

    @torch.no_grad()
    def embed_words(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """B instructions -> the VLM's input embeddings of their words (B, WORD_TOKENS, VLM_DIM) float16 and their
        mask: the words themselves, before the VLM mixes them with the image (its last layer's states at the text
        positions no longer say which word was there plainly enough to query an image by)."""
        if not hasattr(self, "_words"):
            self._words: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for text in set(instructions) - set(self._words):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) > WORD_TOKENS:
                raise ValueError(f"Instruction too long for WORD_TOKENS ({len(ids)} tokens): {text!r}")
            emb = torch.zeros(WORD_TOKENS, VLM_DIM, dtype=torch.float16, device=self.device)
            emb[:len(ids)] = self.vlm.get_input_embeddings()(torch.tensor(ids, device=self.device))
            mask = torch.zeros(WORD_TOKENS, dtype=torch.bool, device=self.device)
            mask[:len(ids)] = True
            self._words[text] = (emb, mask)
        return (torch.stack([self._words[t][0] for t in instructions]),
                torch.stack([self._words[t][1] for t in instructions]))

    @staticmethod
    def head_small(head_rgb: torch.Tensor) -> torch.Tensor:
        """head_rgb (B, H, W, 3) uint8 -> its lower HEAD_CROP_ROWS rows at HEAD_SMALL, (B, rows, columns, 3) uint8."""
        x = head_rgb[:, -HEAD_CROP_ROWS:].permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=HEAD_SMALL, mode="bilinear", antialias=True, align_corners=False)
        return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1)

    @torch.no_grad()
    def embed_wrists(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        """wrist_rgb (B, cameras, 3, H, W) float in [0, 1] (H, W multiples of 14) -> patch tokens
        (B, cameras, H/14 x W/14, WRIST_DIM) float16."""
        b, c = wrist_rgb.shape[:2]
        x = ((wrist_rgb.flatten(0, 1) - self._mean) / self._std).half()
        tokens = self.wrist(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        return tokens.view(b, c, tokens.shape[1], tokens.shape[2])
