"""

python3.11 view_nca_conditional\ copy.py --checkpoint "Conditional NCA Model.pt"

Interactive live NCA viewer — VS Code / local terminal version.

Loads a trained NCA checkpoint (produced by train_nca.py or the Colab notebook)
and runs it continuously in a pygame window. Click and drag with the mouse to
erase parts of the pattern; the NCA regrows it live.

Controls:
    Left click + drag : erase (damage) the pattern under the cursor
    R                 : reset to seed (start growth over)
    P                 : pause / resume stepping
    [ / ]             : decrease / increase brush size
    ESC / close window: quit

Usage:
    pip install pygame torch pillow numpy
    python view_nca.py --checkpoint nca_model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame


# ----------------------------
# Model definition (must match training script)
# ----------------------------
class NCA(nn.Module):
    def __init__(self, channels=16, hidden=128, fire_rate=0.5):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        kernels = torch.stack([identity, sobel_x, sobel_y])
        kernels = kernels.repeat(channels, 1, 1).unsqueeze(1)
        self.register_buffer("perception_kernels", kernels)

        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
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

    def forward(self, x):
        pre_life = self.alive_mask(x)
        y = self.perceive(x)
        dx = self.update(y)
        update_mask = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
                        <= self.fire_rate).float()
        x = x + dx * update_mask
        post_life = self.alive_mask(x)
        life_mask = (pre_life & post_life).float()
        return x * life_mask


def make_seed(size, channels, device):
    seed = torch.zeros(channels, size, size, device=device)
    seed[3:, size // 2, size // 2] = 1.0
    return seed


def rgba_to_rgb(x):
    rgb, a = x[:, :3], x[:, 3:4].clamp(0, 1)
    return rgb + (1.0 - a)  # composite over white


# ----------------------------
# Main interactive loop
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to nca_model.pt")
    parser.add_argument("--scale", type=int, default=8, help="Pixels per grid cell in window")
    parser.add_argument("--fps", type=int, default=20, help="NCA steps per second")
    parser.add_argument("--brush", type=int, default=4, help="Initial brush radius in grid cells")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    channels = ckpt["channels"]
    hidden = ckpt["hidden"]
    size = ckpt["size"]
    fire_rate = ckpt.get("fire_rate", 0.5)

    model = NCA(channels=channels, hidden=hidden, fire_rate=fire_rate).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seed = make_seed(size, channels, device)
    x = seed.unsqueeze(0).clone()

    win_size = size * args.scale
    pygame.init()
    screen = pygame.display.set_mode((win_size, win_size))
    pygame.display.set_caption("Live NCA — drag to erase, R to reset, P to pause, [ ] brush size")
    clock = pygame.time.Clock()

    brush_radius = args.brush
    paused = False
    dragging = False

    yy, xx = torch.meshgrid(
        torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
    )

    def erase_at(gx, gy, radius):
        mask = ((yy - gy) ** 2 + (xx - gx) ** 2) < (radius ** 2)
        with torch.no_grad():
            x[:, :, mask] = 0.0

    def render_surface():
        with torch.no_grad():
            img = rgba_to_rgb(x)[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))  # pygame wants (W, H, 3)
        surf = pygame.transform.scale(surf, (win_size, win_size))
        return surf

    running = True
    font = pygame.font.SysFont("consolas", 16)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    x = seed.unsqueeze(0).clone()
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_LEFTBRACKET:
                    brush_radius = max(1, brush_radius - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    brush_radius = min(size // 2, brush_radius + 1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                px, py = event.pos
                erase_at(px // args.scale, py // args.scale, brush_radius)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                px, py = event.pos
                gx, gy = px // args.scale, py // args.scale
                if 0 <= gx < size and 0 <= gy < size:
                    erase_at(gx, gy, brush_radius)

        if not paused:
            with torch.no_grad():
                x = model(x)

        surf = render_surface()
        screen.blit(surf, (0, 0))

        hud = f"{'PAUSED' if paused else 'running'} | brush={brush_radius} | R=reset P=pause [ ]=brush"
        text = font.render(hud, True, (0, 0, 0), (255, 255, 255))
        screen.blit(text, (4, 4))

        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
