"""

python3.11 view_nca_conditional.py --checkpoint "Conditional NCA Model.pt"
python3.11 view_nca_conditional.py --checkpoint "v2_Conditional NCA_Model.pt"

Interactive live NCA viewer — conditional (multi-layout) version, macOS-friendly.

Loads a trained checkpoint (from train_nca.py, the single-target Colab notebook, or the
conditional multi-layout Colab notebook) and runs it continuously in a pygame window.
Works with BOTH single-target and conditional checkpoints — it auto-detects which one
you loaded based on what's stored in the .pt file.

Controls:
    Left click + drag  : erase (damage) the pattern under the cursor
    R                  : reset to seed (start growth over)
    P                  : pause / resume stepping
    T                  : grow layout 0, then transition the same state through 1, 2, 3
    [ / ]              : decrease / increase brush size
    LEFT / RIGHT arrow : switch to previous / next layout (conditional checkpoints only)
    0-9                : jump directly to layout index 0-9 (conditional checkpoints only)
    ESC / close window : quit

--- Usage as a .py file (Terminal) ---
    pip3 install pygame torch pillow numpy
    python3 view_nca_conditional.py --checkpoint conditional_nca_model.pt

--- Usage inside a notebook cell ---
    Edit CONFIG below, then run this file's contents as a cell or `%run view_nca_conditional.py`.
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
from PIL import Image

CONFIG = {
    "checkpoint": "conditional_nca_model.pt",
    "scale": 8,
    "fps": 20,
    "brush": 4,
    "device": None,  # None = auto-detect (mps > cuda > cpu)
    "initial_growth_steps": 200,
    "transition_steps": 80,
    "transition_interval": 4,
    "transition_gif": "v2_transition.gif",
}


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------
# Model definition — matches both single-target and conditional checkpoints.
# cond_dim=0 makes this behave identically to the plain single-target NCA.
# ----------------------------
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
    parser.add_argument("--initial-growth-steps", type=int, default=CONFIG["initial_growth_steps"])
    parser.add_argument("--transition-steps", type=int, default=CONFIG["transition_steps"])
    parser.add_argument("--transition-interval", type=int, default=CONFIG["transition_interval"])
    parser.add_argument("--transition-gif", type=str, default=CONFIG["transition_gif"])
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
        print(f"Conditional model: {num_layouts} layouts, cond_dim={cond_dim}")
        for i, name in enumerate(image_names):
            print(f"  [{i}] {name}")
    else:
        print("Single-target model (no conditioning).")

    seed = make_seed(size, channels, device)
    x = seed.unsqueeze(0).clone()
    current_layout = 0

    win_size = size * args.scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption("Live NCA — drag to erase, arrows to switch layout, R reset, P pause")
    clock = pygame.time.Clock()

    brush_radius = args.brush
    paused = False
    dragging = False
    transition_active = False
    transition_step = 0
    initial_growth_steps = max(1, args.initial_growth_steps)
    transition_steps = max(1, args.transition_steps)
    transition_interval = max(1, args.transition_interval)
    transition_frames = []

    yy, xx = torch.meshgrid(
        torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
    )

    def current_cond():
        if not is_conditional:
            return None
        with torch.no_grad():
            return embedding(torch.tensor([current_layout], device=device))

    def erase_at(gx, gy, radius):
        mask = ((yy - gy) ** 2 + (xx - gx) ** 2) < (radius ** 2)
        with torch.no_grad():
            x[:, :, mask] = 0.0

    def render_surface():
        with torch.no_grad():
            img = rgba_to_rgb(x)[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (win_size, win_size))
        return surf

    def capture_transition_frame():
        pixels = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        return Image.fromarray(pixels)

    def save_transition_gif():
        duration = max(20, round(1000 * transition_interval / args.fps))
        transition_frames[0].save(
            args.transition_gif,
            save_all=True,
            append_images=transition_frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        print(
            f"Layout transition GIF saved to {args.transition_gif} "
            f"({len(transition_frames)} frames)"
        )

    running = True
    font = pygame.font.SysFont("consolas", 15)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif transition_active:
                    continue
                elif event.key == pygame.K_t and is_conditional and num_layouts >= 4 and not transition_active:
                    current_layout = 0
                    x = seed.unsqueeze(0).clone()
                    transition_step = 0
                    transition_frames.clear()
                    transition_active = True
                    paused = False
                    print(
                        f"Growing layout 0 for {initial_growth_steps} steps, then transitioning "
                        f"through layouts 1 -> 2 -> 3 for {transition_steps} steps each; "
                        f"output={args.transition_gif}"
                    )
                elif event.key == pygame.K_r:
                    x = seed.unsqueeze(0).clone()
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_LEFTBRACKET:
                    brush_radius = max(1, brush_radius - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    brush_radius = min(size // 2, brush_radius + 1)
                    
                elif is_conditional and event.key == pygame.K_LEFT:
                    current_layout = (current_layout - 1) % num_layouts
                    x = seed.unsqueeze(0).clone()
                    
                elif is_conditional and event.key == pygame.K_RIGHT:
                    current_layout = (current_layout + 1) % num_layouts
                    x = seed.unsqueeze(0).clone()
                    
                elif is_conditional and pygame.K_0 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_0
                    if idx < num_layouts:
                        current_layout = idx
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if transition_active:
                    continue
                dragging = True
                px, py = event.pos
                erase_at(px // args.scale, py // args.scale, brush_radius)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                if transition_active:
                    continue
                px, py = event.pos
                gx, gy = px // args.scale, py // args.scale
                if 0 <= gx < size and 0 <= gy < size:
                    erase_at(gx, gy, brush_radius)

        if transition_active:
            with torch.no_grad():
                cond = current_cond()
                x = model(x, cond)
            transition_step += 1
        elif not paused:
            with torch.no_grad():
                cond = current_cond()
                x = model(x, cond)

        surf = render_surface()
        screen.blit(surf, (0, 0))

        if is_conditional:
            layout_name = image_names[current_layout] if current_layout < len(image_names) else str(current_layout)
            import os as _os
            layout_label = f"layout {current_layout}: {_os.path.basename(str(layout_name))}"
        else:
            layout_label = "single-target model"
            
        hud1 = f"{'PAUSED' if paused else 'running'} | brush={brush_radius} | {layout_label}"
        segment_steps = initial_growth_steps if current_layout == 0 else transition_steps
        if transition_active:
            hud2 = f"Recording transition {current_layout}/3: {transition_step}/{transition_steps}"
            if current_layout == 0:
                hud2 = f"Growing layout 0: {transition_step}/{initial_growth_steps}"
        else:
            hud2 = "R=reset  P=pause  T=record  [ ]=brush" + ("  <-/->  0-9=layout" if is_conditional else "")
        text1 = font.render(hud1, True, (0, 0, 0), (255, 255, 255))
        text2 = font.render(hud2, True, (0, 0, 0), (255, 255, 255))
        screen.blit(text1, (4, 4))
        screen.blit(text2, (4, 22))

        pygame.display.flip()
        if transition_active and (
            transition_step % transition_interval == 0 or transition_step >= segment_steps
        ):
            transition_frames.append(capture_transition_frame())
        if transition_active and transition_step >= segment_steps:
            if current_layout < 3:
                current_layout += 1
                transition_step = 0
            else:
                save_transition_gif()
                transition_active = False
        clock.tick(args.fps)

    pygame.quit()
    pygame.display.quit()


if __name__ == "__main__":
    main()
else:
    main()
