from __future__ import annotations

import math
from typing import List

from PIL import Image
from PySide6.QtCore import QThread, Signal

from .adjust_worker import _apply_adjustments, _pil_to_qimage


def _scale_pil_image(im, cell_w: int, cell_h: int, scale_mode: str):
    """
    Scale a PIL Image to a target cell dimension based on the scaling mode:
      - 'fit': maintains aspect ratio, fits entirely within the cell, fills padding with transp/black.
      - 'fill': center-crops/scales to cover the cell completely.
      - 'stretch': stretches to fill cell exactly.
    """
    w, h = im.size
    if scale_mode == "stretch":
        return im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

    im_ratio = w / h
    cell_ratio = cell_w / cell_h

    if scale_mode == "fill":
        if im_ratio > cell_ratio:
            # image is wider: scale height to cell_h, then center-crop width
            new_w = int(cell_h * im_ratio)
            scaled = im.resize((new_w, cell_h), Image.Resampling.LANCZOS)
            left = (new_w - cell_w) // 2
            return scaled.crop((left, 0, left + cell_w, cell_h))
        else:
            # image is taller: scale width to cell_w, then center-crop height
            new_h = int(cell_w / im_ratio)
            scaled = im.resize((cell_w, new_h), Image.Resampling.LANCZOS)
            top = (new_h - cell_h) // 2
            return scaled.crop((0, top, cell_w, top + cell_h))

    else:  # 'fit'
        if im_ratio > cell_ratio:
            new_w = cell_w
            new_h = int(cell_w / im_ratio)
        else:
            new_h = cell_h
            new_w = int(cell_h * im_ratio)

        scaled = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        x = (cell_w - new_w) // 2
        y = (cell_h - new_h) // 2
        bg.paste(scaled, (x, y))
        return bg


class CanvasWorker(QThread):
    sig_finished = Signal(object)  # QImage
    sig_error = Signal(str)

    def __init__(
        self,
        images_params: List[tuple[str, dict]],  # list of (path, adj_params)
        layout_mode: str,  # 'horizontal', 'vertical', 'grid'
        canvas_w: int,
        canvas_h: int,
        bg_color: tuple[int, int, int],  # (R, G, B)
        scale_mode: str,  # 'fit', 'fill', 'stretch'
        gap: int = 10,
        preview: bool = True,
    ):
        super().__init__()
        self._images_params = images_params
        self._layout_mode = layout_mode
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._bg_color = bg_color
        self._scale_mode = scale_mode
        self._gap = gap
        self._preview = preview

    def run(self):
        if not self._images_params:
            bg = Image.new("RGB", (self._canvas_w, self._canvas_h), self._bg_color)
            self.sig_finished.emit(_pil_to_qimage(bg))
            return

        try:
            # 1. Load and adjust all images
            loaded_imgs = []
            for path, params in self._images_params:
                im = Image.open(path)
                adjusted = _apply_adjustments(im, params)
                loaded_imgs.append(adjusted)

            # 2. Build canvas
            canvas = Image.new("RGB", (self._canvas_w, self._canvas_h), self._bg_color)
            n = len(loaded_imgs)
            mode = self._layout_mode.lower()
            gap = self._gap
            scale_mode = self._scale_mode

            if mode == "horizontal":
                # divide width equally minus gaps
                total_gaps = gap * (n - 1)
                cell_w = max(1, (self._canvas_w - total_gaps) // n)
                cell_h = self._canvas_h
                for i, im in enumerate(loaded_imgs):
                    x = i * (cell_w + gap)
                    canvas.paste(
                        _scale_pil_image(im, cell_w, cell_h, scale_mode), (x, 0)
                    )

            elif mode == "vertical":
                # divide height equally minus gaps
                total_gaps = gap * (n - 1)
                cell_w = self._canvas_w
                cell_h = max(1, (self._canvas_h - total_gaps) // n)
                for i, im in enumerate(loaded_imgs):
                    y = i * (cell_h + gap)
                    canvas.paste(
                        _scale_pil_image(im, cell_w, cell_h, scale_mode), (0, y)
                    )

            else:  # 'grid'
                # Find optimal grid rows/cols matching canvas aspect ratio
                canvas_ratio = self._canvas_w / self._canvas_h
                best_rows, best_cols = 1, n
                best_err = 1e9
                for cols in range(1, n + 1):
                    rows = math.ceil(n / cols)
                    grid_ratio = cols / rows
                    err = abs(grid_ratio - canvas_ratio)
                    if err < best_err:
                        best_err = err
                        best_rows = rows
                        best_cols = cols

                cols, rows = best_cols, best_rows
                total_gaps_x = gap * (cols - 1)
                total_gaps_y = gap * (rows - 1)
                cell_w = max(1, (self._canvas_w - total_gaps_x) // cols)
                cell_h = max(1, (self._canvas_h - total_gaps_y) // rows)

                for i, im in enumerate(loaded_imgs):
                    row_i = i // cols
                    col_i = i % cols
                    x = col_i * (cell_w + gap)
                    y = row_i * (cell_h + gap)
                    canvas.paste(
                        _scale_pil_image(im, cell_w, cell_h, scale_mode), (x, y)
                    )

            if self._preview:
                max_dim = 900
                w, h = canvas.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    canvas = canvas.resize(
                        (int(w * scale), int(h * scale)),
                        Image.Resampling.LANCZOS,
                    )

            self.sig_finished.emit(_pil_to_qimage(canvas))
        except Exception as e:
            self.sig_error.emit(str(e))
