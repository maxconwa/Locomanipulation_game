"""Augmentation of the images of teacher episodes recorded over the chroma-green floor (scripts/vla/collect_dataset.py).

Nothing here renders: each function edits recorded pixels.

- the floor is an unlit green sheet, so its pixels fall in a small colour box (`floor_mask`) and take any background;
- the table is the only lit beige surface (`table_mask`): a new texture is multiplied by each pixel's brightness
  relative to the lit table, which keeps the shadows of the blocks and the arms on it (`retexture`);
- the blocks are the only saturated colours, one hue band per colour name (`block_classes`): `recolor_blocks` swaps
  the names' colours and `recolor_instruction` swaps the words to match.

Textures are drawn on the floor's and the table's planes as each camera sees them (`plane_texture`): the head camera
is fixed in the pelvis frame and a wrist camera's pose follows from the proprioception's grasp point
(`head_camera`, `wrist_cameras`), so a texture stays put on its surface while the hand moves. One draw per episode: a
real room's floor and table do not change between frames. All in the pelvis frame, images uint8 (..., H, W, 3).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FLOOR_BOX = (215, 50)  # a floor pixel has g >= 215 and r, b <= 50 (the sheet decodes as about (31, 232, 31))
# the lit table's colour as rendered (sRGB about (213, 200, 181)), linear; hue in OpenCV's half degrees
TABLE_LINEAR = (0.654, 0.562, 0.450)
TABLE_BAND = {"hue": (13, 24), "saturation": (20, 75), "value": 110}
# the blocks' names in mdp/vla.py BLOCK_COLORS' order: each one's hue bands (half degrees) and median rendered colour
# (linear). The renderer's tone curve moves orange towards yellow, so these are measured, not the materials' colours.
BLOCK_HUES = {"red": ((0, 6), (172, 180)), "green": ((50, 80),), "blue": ((95, 125),),
              "yellow": ((27, 40),), "orange": ((7, 26),), "purple": ((130, 160),)}
BLOCK_LINEAR = {"red": (0.708, 0.112, 0.112), "green": (0.147, 0.644, 0.220), "blue": (0.147, 0.296, 0.723),
                "yellow": (0.730, 0.708, 0.141), "orange": (0.624, 0.287, 0.021), "purple": (0.584, 0.223, 0.680)}
# of 255. The table's hue lies in orange's band and its saturation stays under 75; a lit purple or blue face seen
# from close falls to 60
BLOCK_MIN_SATURATION, ORANGE_MIN_SATURATION, BLOCK_MIN_VALUE = 50, 90, 40
PELVIS_HEIGHT = 0.98  # m over the floor
VFOV = math.radians(58.0)
HEAD_CAMERA_POS = (0.11109, 0.0175, 0.68789)
HEAD_CAMERA_QUAT = (0.23743, -0.66605, 0.66605, -0.23743)  # w, x, y, z of the camera's (right, down, forward) axes
WRIST_CAMERA_BACK = 0.10  # m behind the grasp point, looking along the approach axis: fitted to the blocks' pixels


def to_linear(images: torch.Tensor) -> torch.Tensor:
    x = images.float() / 255.0
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: torch.Tensor) -> torch.Tensor:
    x = linear.clamp(0.0, 1.0)
    x = torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-6) ** (1.0 / 2.4) - 0.055)
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def hsv(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hue in half degrees [0, 180), saturation and value of 255: OpenCV's convention."""
    x = images.float()
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    high, low = x.max(dim=-1).values, x.min(dim=-1).values
    span = (high - low).clamp(min=1e-6)
    hue = torch.where(high == r, (g - b) / span % 6.0, torch.where(high == g, (b - r) / span + 2.0, (r - g) / span + 4.0))
    return hue * 30.0, 255.0 * (high - low) / high.clamp(min=1e-6), high


def floor_mask(images: torch.Tensor, grow: bool = True) -> torch.Tensor:
    """The floor's pixels (..., H, W). With `grow`, also the green-tinted pixels next to them: where the video's
    compression blended the floor into its neighbour."""
    x = images.int()
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mask = (g >= FLOOR_BOX[0]) & (r <= FLOOR_BOX[1]) & (b <= FLOOR_BOX[1])
    if grow:
        shape = mask.shape
        near = F.max_pool2d(mask.reshape(-1, 1, *shape[-2:]).float(), 3, stride=1, padding=1).reshape(shape) > 0
        mask = mask | (near & (g - torch.maximum(r, b) > 30))
    return mask


def table_mask(images: torch.Tensor) -> torch.Tensor:
    """The table's pixels (..., H, W), lit or in shadow."""
    hue, saturation, value = hsv(images)
    band = TABLE_BAND
    return ((hue >= band["hue"][0]) & (hue < band["hue"][1]) & (saturation >= band["saturation"][0])
            & (saturation <= band["saturation"][1]) & (value >= band["value"]))


def block_classes(images: torch.Tensor) -> torch.Tensor:
    """Each pixel's block colour (..., H, W): an index into BLOCK_HUES, or -1 where there is no block."""
    hue, saturation, value = hsv(images)
    block = (value > BLOCK_MIN_VALUE) & ~floor_mask(images)
    classes = torch.full(hue.shape, -1, dtype=torch.long, device=images.device)
    for index, (name, bands) in enumerate(BLOCK_HUES.items()):
        floor = ORANGE_MIN_SATURATION if name == "orange" else BLOCK_MIN_SATURATION
        for low, high in bands:
            classes[block & (saturation > floor) & (hue >= low) & (hue < high)] = index
    return classes


def recolor_blocks(images: torch.Tensor, colors: torch.Tensor, classes: torch.Tensor | None = None) -> torch.Tensor:
    """Each block colour's pixels in another colour: `colors` (names, 3) linear, the colour that name's blocks take
    (BLOCK_LINEAR's rows change nothing; rows of another name's colour swap names, see recolor_instruction). A pixel
    is its colour times a brightness plus a remainder (the highlights): the brightness moves to the new colour and
    the remainder is kept as grey."""
    classes = block_classes(images) if classes is None else classes
    linear = to_linear(images)
    old = torch.tensor(list(BLOCK_LINEAR.values()), device=images.device)
    chosen = classes.clamp(min=0)
    base, new = old[chosen], colors.to(images.device).float()[chosen]
    brightness = (linear * base).sum(-1, keepdim=True) / (base * base).sum(-1, keepdim=True)
    remainder = (linear - brightness * base).mean(-1, keepdim=True)
    return torch.where((classes >= 0).unsqueeze(-1), to_srgb(brightness * new + remainder), images)


def recolor_instruction(instruction: str, names: list[str]) -> str:
    """The instruction after its blocks took other names' colours: names[i] is the new name of BLOCK_HUES' i-th."""
    swap = dict(zip(BLOCK_HUES, names))
    return " ".join(swap.get(word, word) for word in instruction.split(" "))


def quat_matrix(quat) -> torch.Tensor:
    w, x, y, z = quat
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def head_camera() -> tuple[torch.Tensor, torch.Tensor]:
    """The head camera's position (3,) and axes (3, 3: columns right, down, forward)."""
    return torch.tensor(HEAD_CAMERA_POS), quat_matrix(HEAD_CAMERA_QUAT)


def wrist_cameras(proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The wrist cameras' positions (T, 2, 3) and axes (T, 2, 3, 3), left then right, from the proprioception
    (T, 52: mdp/vla.py vla_proprio, each grasp point's position and first two axes at 16:34)."""
    tcp = proprio[:, 16:34].reshape(-1, 2, 9).float()
    forward, side = tcp[..., 3:6], tcp[..., 6:9]
    right = torch.cross(forward, side, dim=-1)
    return tcp[..., :3] - WRIST_CAMERA_BACK * forward, torch.stack((right, -side, forward), dim=-1)


def plane_texture(texture: torch.Tensor, size: tuple[int, int], pos: torch.Tensor, axes: torch.Tensor, plane_z: float,
                  tile: float = 1.0, yaw: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    """`texture` (h, w, 3) uint8, tiled every `tile` m on the plane z = plane_z, as cameras at `pos` (..., 3) with
    `axes` (..., 3, 3) see it: images (..., H, W, 3) for size (H, W). Far texels are averaged (a mip pyramid); rays
    that miss the plane get the texture's mean colour."""
    rows, cols = size
    focal = 0.5 * rows / math.tan(0.5 * VFOV)
    v, u = torch.meshgrid(torch.arange(rows, dtype=torch.float32) - 0.5 * (rows - 1),
                          torch.arange(cols, dtype=torch.float32) - 0.5 * (cols - 1), indexing="ij")
    ray = torch.stack((u, v, torch.full_like(u, focal)), dim=-1)  # (H, W, 3) in the camera's axes
    ray = torch.einsum("...ij,hwj->...hwi", axes.float(), ray)
    drop = pos.float()[..., None, None, 2] - plane_z
    hit = ray[..., 2] * drop < -1e-4 * focal * drop.abs()  # the ray heads for the plane
    reach = torch.where(hit, drop / torch.where(hit, -ray[..., 2], torch.ones_like(drop)), torch.zeros_like(drop))
    x = pos.float()[..., None, None, 0] + reach * ray[..., 0]
    y = pos.float()[..., None, None, 1] + reach * ray[..., 1]
    gx = (math.cos(yaw) * x - math.sin(yaw) * y + shift[0]) / tile % 1.0
    gy = (math.sin(yaw) * x + math.cos(yaw) * y + shift[1]) / tile % 1.0
    # texels under one pixel: the pixel's width on the plane, stretched where the ray grazes it
    length = ray.norm(dim=-1)
    graze = ((-ray[..., 2]).abs() / length).clamp(min=0.05)
    texels = reach * length / focal / graze / tile * texture.shape[0]
    pyramid = [texture.permute(2, 0, 1).float().unsqueeze(0)]
    while pyramid[-1].shape[-1] > 4:
        pyramid.append(F.avg_pool2d(pyramid[-1], 2))
    level = torch.log2(texels.clamp(min=1.0)).round().long().clamp(max=len(pyramid) - 1)
    out = torch.zeros(*gx.shape, 3)
    for index, image in enumerate(pyramid):
        chosen = level == index
        if chosen.any():
            n = image.shape[-1]
            out[chosen] = image[0][:, (gy[chosen] * n).long().clamp(max=n - 1), (gx[chosen] * n).long().clamp(max=n - 1)].T
    mean = texture.float().mean(dim=(0, 1))
    return torch.where(hit.unsqueeze(-1), out, mean.expand_as(out)).round().clamp(0, 255).to(torch.uint8)


def replace_floor(images: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Images with their floor pixels taken from `background` (the same shape, or (H, W, 3))."""
    return torch.where(floor_mask(images).unsqueeze(-1), background.to(images.device).expand_as(images), images)


def retexture(images: torch.Tensor, surface: torch.Tensor, mask: torch.Tensor | None = None, gain: float = 1.0) -> torch.Tensor:
    """Images with the table's pixels (or `mask`'s) taken from `surface`, each as bright as the pixel it replaces is
    against the lit table: shadows stay."""
    mask = table_mask(images) if mask is None else mask
    weights = torch.tensor((0.2126, 0.7152, 0.0722), device=images.device)
    shade = (to_linear(images) * weights).sum(-1, keepdim=True) / float((torch.tensor(TABLE_LINEAR) * weights.cpu()).sum())
    new = to_srgb(to_linear(surface.to(images.device).expand_as(images)) * shade.clamp(max=1.2) * gain)
    return torch.where(mask.unsqueeze(-1), new, images)


def jitter(images: torch.Tensor, gain: float = 1.0, bias: float = 0.0, saturation: float = 1.0,
           tint: tuple[float, float, float] = (1.0, 1.0, 1.0), noise: float = 0.0, blur: float = 0.0) -> torch.Tensor:
    """A camera's own look, after every edit above (they read the recorded colours): contrast `gain` about mid-grey,
    brightness `bias` (of 1), `saturation` about each pixel's grey, a white-balance `tint` per channel, a Gaussian
    `blur` (px) and sensor `noise` (of 1, drawn per pixel and frame). One draw per episode and camera, the noise
    apart; the hue is left alone: the instruction names the blocks by it."""
    x = images.float() / 255.0
    if blur > 0.0:
        radius = max(1, round(2.0 * blur))
        kernel = torch.exp(-0.5 * (torch.arange(-radius, radius + 1, dtype=torch.float32) / blur) ** 2)
        kernel = (kernel / kernel.sum()).to(x.device)
        shape = x.shape
        x = x.reshape(-1, *shape[-3:]).permute(0, 3, 1, 2).reshape(-1, 1, *shape[-3:-1])
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
        x = x.reshape(-1, 3, *shape[-3:-1]).permute(0, 2, 3, 1).reshape(shape)
    grey = x.mean(dim=-1, keepdim=True)
    x = (grey + saturation * (x - grey)) * torch.tensor(tint, device=x.device)
    x = (x - 0.5) * gain + 0.5 + bias
    if noise > 0.0:
        x = x + noise * torch.randn_like(x)
    return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
