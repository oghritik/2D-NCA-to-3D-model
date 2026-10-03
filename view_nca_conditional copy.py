"""

python3.11 view_nca_conditional\ copy.py --checkpoint "Conditional NCA Model.pt"

Multi-layout NCA viewer — shows ALL layouts growing simultaneously, side by side,
each from its own seed, from the beginning. Erase any panel independently and
watch that panel regrow while the others keep going untouched.

Controls:
    Left click + drag in a panel : erase (damage) that panel's pattern only
    R                             : reset ALL panels to seed
    P                             : pause / resume all panels
    T                             : damage every layout, record regeneration to test.gif
    [ / ]                        : decrease / increase brush size
    ESC / close window           : quit

--- Usage as a .py file (Terminal) ---
    pip3 install pygame torch pillow numpy
    python3 view_nca_conditional.py --checkpoint conditional_nca_model.pt

--- Usage inside a notebook cell ---
    Edit CONFIG below, then run this file's contents as a cell or `%run view_nca_conditional.py`.
"""

import sys
import math
import os as _os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
from PIL import Image

CONFIG = {
    "checkpoint": "conditional_nca_model.pt",
    "scale": None,   # None = auto-pick based on how many layouts (smaller grid -> bigger scale)
    "fps": 20,
    "brush": 4,
    "device": None,  # None = auto-detect (mps > cuda > cpu)
    "test_damage_radius": None,  # None = 20% of the grid width
    "test_warmup_steps": 80,
    "test_steps": 120,
    "test_gif": "test.gif",
}


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConditionalNCA(nn.Module):
    def __init__(self, channels=16, hidden=128, cond_dim=0, fire_rate=0.5):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.fire_rate = fire_rate

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        kernels = torch.stack([identity, sobel_x, sobel_y])
        kernels = kernels.repeat(channels, 1, 1).unsqueeze(1)
        self.register_buffer("perception_kernels", kernels)

        self.update = nn.Sequential(
            nn.Conv2d(channels * 3 + cond_dim, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def perceive(self, x):
        return F.conv2d(
            F.pad(x, (1, 1, 1, 1), mode="circular"),
            self.perception_kernels,
            groups=self.channels,
        )

    def alive_mask(self, x):
        alpha = x[:, 3:4, :, :]
        alive = F.max_pool2d(
            F.pad(alpha, (1, 1, 1, 1), mode="circular"), kernel_size=3, stride=1
        ) > 0.1
        return alive

    def forward(self, x, cond=None):
        B, _, H, W = x.shape
        pre_life = self.alive_mask(x)
        y = self.perceive(x)
        if self.cond_dim > 0:
            cond_map = cond.view(B, self.cond_dim, 1, 1).expand(-1, -1, H, W)
            y = torch.cat([y, cond_map], dim=1)
        dx = self.update(y)
        update_mask = (torch.rand(B, 1, H, W, device=x.device) <= self.fire_rate).float()
        x = x + dx * update_mask
        post_life = self.alive_mask(x)
        return x * (pre_life & post_life).float()


def make_seed(size, channels, device):
    seed = torch.zeros(channels, size, size, device=device)
    seed[3:, size // 2, size // 2] = 1.0
    return seed


def rgba_to_rgb(x):
    rgb, a = x[:, :3], x[:, 3:4].clamp(0, 1)
    return rgb + (1.0 - a)


def get_run_config():
    is_notebook = "ipykernel" in sys.modules
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CONFIG["checkpoint"])
    parser.add_argument("--scale", type=int, default=CONFIG["scale"])
    parser.add_argument("--fps", type=int, default=CONFIG["fps"])
    parser.add_argument("--brush", type=int, default=CONFIG["brush"])
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--test-damage-radius", type=int, default=CONFIG["test_damage_radius"])
    parser.add_argument("--test-warmup-steps", type=int, default=CONFIG["test_warmup_steps"])
    parser.add_argument("--test-steps", type=int, default=CONFIG["test_steps"])
    parser.add_argument("--test-gif", type=str, default=CONFIG["test_gif"])
    if is_notebook:
        args, _ = parser.parse_known_args([])
    else:
        args, _ = parser.parse_known_args()
    return args


def main():
    args = get_run_config()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    channels = ckpt["channels"]
    hidden = ckpt["hidden"]
    size = ckpt["size"]
    fire_rate = ckpt.get("fire_rate", 0.5)
    cond_dim = ckpt.get("cond_dim", 0)
    num_layouts = ckpt.get("num_layouts", 1)
    image_names = ckpt.get("image_names", ["target"])

    model = ConditionalNCA(channels=channels, hidden=hidden, cond_dim=cond_dim, fire_rate=fire_rate).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    is_conditional = cond_dim > 0
    embedding = None
    if is_conditional:
        embedding = nn.Embedding(num_layouts, cond_dim).to(device)
        embedding.load_state_dict(ckpt["embedding_state"])
        embedding.eval()
    else:
        num_layouts = 1  # single-target model -> just show the one panel

    print(f"{'Conditional' if is_conditional else 'Single-target'} model, {num_layouts} layout(s):")
    for i in range(num_layouts):
        name = image_names[i] if i < len(image_names) else str(i)
        print(f"  [{i}] {_os.path.basename(str(name))}")

    # --- Grid layout: pick rows x cols to fit num_layouts as squarely as possible ---
    cols = math.ceil(math.sqrt(num_layouts))
    rows = math.ceil(num_layouts / cols)

    # --- Auto scale: shrink cell size as grid grows, so window stays a sane size ---
    if args.scale is not None:
        scale = args.scale
    else:
        if num_layouts <= 1:
            scale = 8
        elif num_layouts <= 4:
            scale = 6
        elif num_layouts <= 9:
            scale = 4
        else:
            scale = 3

    panel_px = size * scale
    label_h = 20  # extra pixels above each panel for its name
    win_w = cols * panel_px
    win_h = rows * (panel_px + label_h)

    seed = make_seed(size, channels, device)
    # One independent state per layout, all starting from the seed, all growing from t=0
    states = [seed.unsqueeze(0).clone() for _ in range(num_layouts)]

    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Live NCA — drag to erase, T regeneration test, R reset, P pause")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14)

    brush_radius = args.brush
    paused = False
    dragging_panel = None  # which panel index is currently being dragged on
    test_active = False
    test_phase = ""
    test_step = 0
    test_warmup_steps = max(0, args.test_warmup_steps)
    test_frames = []
    test_damage_radius = args.test_damage_radius or max(1, size // 5)
    test_steps = max(1, args.test_steps)
    capture_every = max(1, test_steps // 40)

    yy, xx = torch.meshgrid(
        torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
    )

    def cond_for(layout_id):
        if not is_conditional:
            return None
        with torch.no_grad():
            return embedding(torch.tensor([layout_id], device=device))

    def erase_at(panel_idx, gx, gy, radius):
        mask = ((yy - gy) ** 2 + (xx - gx) ** 2) < (radius ** 2)
        with torch.no_grad():
            states[panel_idx][:, :, mask] = 0.0

    def capture_frame():
        pixels = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        return Image.fromarray(pixels)

    def save_test_gif():
        if not test_frames:
            return
        duration = max(20, round(1000 * capture_every / args.fps))
        test_frames[0].save(
            args.test_gif,
            save_all=True,
            append_images=test_frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        print(f"Regeneration test saved to {args.test_gif} ({len(test_frames)} frames)")

    def panel_origin(idx):
        r, c = divmod(idx, cols)
        return c * panel_px, r * (panel_px + label_h) + label_h

    def pixel_to_panel(px, py):
        """Map a window pixel coord to (panel_idx, grid_x, grid_y) or None if outside any panel."""
        col = px // panel_px
        row = py // (panel_px + label_h)
        if col >= cols or row >= rows:
            return None
        idx = row * cols + col
        if idx >= num_layouts:
            return None
        local_x = px - col * panel_px
        local_y = py - (row * (panel_px + label_h) + label_h)
        if local_y < 0:
            return None  # clicked on the label strip, not the panel
        gx, gy = local_x // scale, local_y // scale
        if not (0 <= gx < size and 0 <= gy < size):
            return None
        return idx, gx, gy

    def render():
        screen.fill((255, 255, 255))
        for i in range(num_layouts):
            with torch.no_grad():
                img = rgba_to_rgb(states[i])[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            img_up = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
            surf = pygame.surfarray.make_surface(img_up.swapaxes(0, 1))
            ox, oy = panel_origin(i)
            screen.blit(surf, (ox, oy))
            pygame.draw.rect(screen, (0, 0, 0), (ox, oy, panel_px, panel_px), 1)

            name = image_names[i] if i < len(image_names) else str(i)
            label = f"[{i}] {_os.path.basename(str(name))}"
            text = font.render(label, True, (0, 0, 0))
            screen.blit(text, (ox + 2, oy - label_h + 2))

        test_status = (
            f" | preparing={test_step}/{test_warmup_steps}"
            if test_active and test_phase == "warmup"
            else f" | regenerating={test_step}/{test_steps}"
            if test_active
            else ""
        )
        hud = f"{'PAUSED' if paused else 'running'} | brush={brush_radius}{test_status} | T=test  R=reset  P=pause"
        hud_surf = font.render(hud, True, (0, 0, 0), (255, 255, 255))
        screen.blit(hud_surf, (4, win_h - 18))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    states = [seed.unsqueeze(0).clone() for _ in range(num_layouts)]
                    test_active = False
                    test_phase = ""
                    test_frames.clear()
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_t and not test_active:
                    states = [seed.unsqueeze(0).clone() for _ in range(num_layouts)]
                    test_step = 0
                    test_frames.clear()
                    test_active = True
                    test_phase = "warmup" if test_warmup_steps else "regeneration"
                    print(
                        f"Regeneration test started: warmup={test_warmup_steps}, "
                        f"radius={test_damage_radius} cells, steps={test_steps}, "
                        f"output={args.test_gif}"
                    )
                elif event.key == pygame.K_LEFTBRACKET:
                    brush_radius = max(1, brush_radius - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    brush_radius = min(size // 2, brush_radius + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                hit = pixel_to_panel(*event.pos)
                if hit is not None:
                    idx, gx, gy = hit
                    dragging_panel = idx
                    erase_at(idx, gx, gy, brush_radius)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_panel = None
            elif event.type == pygame.MOUSEMOTION and dragging_panel is not None:
                hit = pixel_to_panel(*event.pos)
                if hit is not None and hit[0] == dragging_panel:
                    _, gx, gy = hit
                    erase_at(dragging_panel, gx, gy, brush_radius)

        if not paused:
            with torch.no_grad():
                for i in range(num_layouts):
                    cond = cond_for(i)
                    states[i] = model(states[i], cond)
            if test_active and test_phase == "warmup":
                test_step += 1
                if test_step >= test_warmup_steps:
                    center = size // 2
                    for i in range(num_layouts):
                        erase_at(i, center, center, test_damage_radius)
                    test_phase = "regeneration"
                    test_step = 0
            elif test_active:
                test_step += 1

        render()
        pygame.display.flip()
        if test_active and test_phase == "regeneration" and (
            test_step % capture_every == 0 or test_step >= test_steps
        ):
            test_frames.append(capture_frame())
        if test_active and test_phase == "regeneration" and test_step >= test_steps:
            save_test_gif()
            test_active = False
            test_phase = ""
        clock.tick(args.fps)

    pygame.quit()
    pygame.display.quit()


if __name__ == "__main__":
    main()
else:
    main()