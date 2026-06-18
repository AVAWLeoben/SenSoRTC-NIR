# -*- coding: utf-8 -*-
"""
Modernized pygame-ce UI module.

Drop-in notes:
- Keeps the public display(...) entry point, now extended with vertical movement controls.
- Adds manual VERT_MOVEMENT control and a track-calibration toggle.
"""

import os
import time
import datetime
from queue import Empty

import cv2
import numpy as np
import pygame
import easygui

import threading
import traceback

COLOR_BG = (18, 20, 25)
COLOR_SURFACE = (28, 31, 38)
COLOR_SURFACE_2 = (36, 40, 49)
COLOR_PANEL_BORDER = (68, 74, 88)
COLOR_TEXT = (235, 238, 245)
COLOR_TEXT_MUTED = (174, 181, 196)
COLOR_PRIMARY = (74, 144, 226)
COLOR_PRIMARY_HOVER = (96, 160, 236)
COLOR_PRIMARY_PRESSED = (58, 122, 201)
COLOR_SUCCESS = (52, 168, 83)
COLOR_SUCCESS_HOVER = (73, 188, 104)
COLOR_SUCCESS_PRESSED = (42, 142, 70)
COLOR_DANGER = (220, 68, 55)
COLOR_DANGER_HOVER = (232, 87, 74)
COLOR_DANGER_PRESSED = (188, 54, 43)
COLOR_ACCENT = (105, 195, 255)
COLOR_CHECK = (82, 196, 26)

NIR_COLOR_CHOICES = [
    (0, 0, 0),
    (255, 64, 64),
    (64, 220, 64),
    (64, 128, 255),
    (255, 220, 64),
    (220, 64, 255),
    (64, 220, 220),
    (255, 140, 64),
    (255, 255, 255),
]


def normalise_rgb(value, fallback=(255, 255, 255)):
    try:
        if isinstance(value, str):
            text = value.strip().lstrip('#')
            if len(text) == 6:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
            return fallback
        rgb = tuple(int(v) for v in value[:3])
        return tuple(max(0, min(255, v)) for v in rgb)
    except Exception:
        return fallback


def rgb_to_hex(rgb):
    r, g, b = normalise_rgb(rgb)
    return f"#{r:02X}{g:02X}{b:02X}"

WINDOW_SIZE = (1400, 1025)
FRAME_DEST = (0, 0)
SIDEBAR_RECT = pygame.Rect(920, 10, 460, 950)

GUI_DISPLAY_FPS = 30

recording = False


def draw_rounded_panel(surface, rect, fill=COLOR_SURFACE, border=COLOR_PANEL_BORDER, radius=14, width=1):
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def draw_text(surface, text, font, color, pos, align="topleft"):
    text_surface = font.render(str(text), True, color)
    text_rect = text_surface.get_rect()
    setattr(text_rect, align, pos)
    surface.blit(text_surface, text_rect)
    return text_rect


class CheckBox:
    def __init__(self, rect, text="", checked=False, callback=None, font_size=18, color=COLOR_TEXT):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.checked = checked
        self.callback = callback
        self.font = pygame.font.SysFont("Segoe UI", font_size)
        self.text_color = color
        self.box_color = (120, 130, 150)
        self.box_fill = COLOR_SURFACE_2
        self.check_color = COLOR_CHECK
        self.hovered = False
        self.enabled = True

    def handle_event(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.checked = not self.checked
                if self.callback:
                    self.callback()

    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)
        for event in events:
            self.handle_event(event)

    def draw(self, surface):
        border_color = COLOR_ACCENT if self.hovered else self.box_color
        pygame.draw.rect(surface, self.box_fill, self.rect, border_radius=6)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=6)
        if self.checked:
            inner_rect = self.rect.inflate(-8, -8)
            pygame.draw.rect(surface, self.check_color, inner_rect, border_radius=4)
        if self.text:
            label_rect = self.rect.copy()
            label_rect.x = self.rect.right + 10
            label_rect.centery = self.rect.centery
            draw_text(surface, self.text, self.font, self.text_color, label_rect.midleft, align="midleft")


class ClassColorSwatch:
    def __init__(self, rect, class_index, colors_proxy=None):
        self.rect = pygame.Rect(rect)
        self.class_index = int(class_index)
        self.colors_proxy = colors_proxy
        self.hovered = False
        self.enabled = True
        self.small_font = pygame.font.SysFont("Segoe UI", 13)
        self.color_picker_active = False
        self.pending_color = None

    def _get_color(self):
        try:
            return normalise_rgb(self.colors_proxy[self.class_index])
        except Exception:
            return NIR_COLOR_CHOICES[self.class_index % len(NIR_COLOR_CHOICES)]

    def _set_color(self, rgb):
        if self.colors_proxy is None:
            return
        try:
            self.colors_proxy[self.class_index] = tuple(normalise_rgb(rgb))
        except Exception as exc:
            print(f"[UI] Could not set NIR class color: {exc}")

    def _cycle_color(self):
        current = self._get_color()
        choices = [normalise_rgb(c) for c in NIR_COLOR_CHOICES]
        try:
            idx = choices.index(current)
            self._set_color(choices[(idx + 1) % len(choices)])
        except ValueError:
            self._set_color(choices[self.class_index % len(choices)])

    def _prompt_color_hex(self):
        current = rgb_to_hex(self._get_color())
        text = easygui.enterbox(
            "Enter RGB hex color for this NIR class, e.g. #FF4040",
            "NIR class color",
            current,
        )
        if text:
            self._set_color(normalise_rgb(text, self._get_color()))

    def _prompt_color(self):
        if self.color_picker_active:
            return
    
        self.color_picker_active = True
        current = rgb_to_hex(self._get_color())
    
        def worker():
            try:
                from tkinter import Tk
                from tkinter.colorchooser import askcolor
    
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
    
                _, hex_color = askcolor(
                    color=current,
                    title=f"Choose color for class {self.class_index}",
                    parent=root,
                )
    
                root.destroy()
    
                if hex_color:
                    self.pending_color = hex_color
    
            except Exception as exc:
                print(f"[UI] Tk color picker failed: {exc}")
    
            finally:
                self.color_picker_active = False
    
        threading.Thread(target=worker, daemon=True).start()

    def update(self, events):
        if self.pending_color:
            self._set_color(normalise_rgb(self.pending_color, self._get_color()))
            self.pending_color = None
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)
        if not self.enabled:
            return
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
                if event.button == 1:
                    self._cycle_color()
                elif event.button == 3:
                    self._prompt_color()

    def draw(self, surface):
        color = self._get_color()
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        border = COLOR_ACCENT if self.hovered else COLOR_PANEL_BORDER
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=6)
        if self.hovered:
            draw_text(surface, rgb_to_hex(color), self.small_font, COLOR_TEXT_MUTED, (self.rect.right + 6, self.rect.centery), align="midleft")


class Button:
    def __init__(self, x, y, w, h, callback=None, text=None, *, style="primary", toggle=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.hovered = False
        self.pressed = False
        self.callback = callback
        self.text = text or ""
        self.style = style
        self.toggle = toggle
        self.clicked = False
        self.enabled = True
        self.font = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.border_radius = min(14, max(8, min(w, h) // 4))

    def _palette(self):
        if self.style == "success":
            return COLOR_SUCCESS, COLOR_SUCCESS_HOVER, COLOR_SUCCESS_PRESSED
        if self.style == "danger":
            return COLOR_DANGER, COLOR_DANGER_HOVER, COLOR_DANGER_PRESSED
        return COLOR_PRIMARY, COLOR_PRIMARY_HOVER, COLOR_PRIMARY_PRESSED

    def handle_event(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.clicked = not self.clicked
                if self.callback is not None:
                    self.callback()

    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos) if self.enabled else False
        for event in events:
            self.handle_event(event)

    def draw(self, surface):
        base, hover, pressed = self._palette()
        border_color = COLOR_PANEL_BORDER
        border_width = 1
        if not self.enabled:
            fill = (85, 90, 100)
            text_color = COLOR_TEXT_MUTED
        elif self.toggle and self.clicked:
            fill = pressed
            text_color = COLOR_TEXT
            border_color = COLOR_ACCENT
            border_width = 3
        elif self.pressed:
            fill = pressed
            text_color = COLOR_TEXT
        elif self.hovered:
            fill = hover
            text_color = COLOR_TEXT
        else:
            fill = base
            text_color = COLOR_TEXT
        pygame.draw.rect(surface, fill, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(
            surface,
            border_color,
            self.rect,
            border_width,
            border_radius=self.border_radius
        )
        if self.text:
            draw_text(surface, self.text, self.font, text_color, self.rect.center, align="center")


class ToggleButton(Button):
    def __init__(self, x, y, w, h, callback=None, text=None, *, style="success"):
        super().__init__(x, y, w, h, callback=callback, text=text, style=style, toggle=True)
        self.debounce_time = 0.20
        self.last_clicked_time = 0.0

    def handle_event(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            now = time.time()
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos) and now > self.last_clicked_time + self.debounce_time:
                self.clicked = not self.clicked
                self.last_clicked_time = now
                if self.callback is not None:
                    self.callback()


class ValueStepper:
    def __init__(self, label, x, y, width, value_getter, dec_callback, inc_callback, hint="", fmt="{:.2f}", height=52):
        self.label = label
        self.rect = pygame.Rect(x, y, width, height)
        self.value_getter = value_getter
        self.hint = hint
        self.fmt = fmt
        button_size = 40
        padding = 4
        self.minus = Button(x + width - (button_size * 2 + padding + 8), y + 4, button_size, 40, dec_callback, "-", style="primary")
        self.plus = Button(x + width - (button_size + 4), y + 4, button_size, 40, inc_callback, "+", style="primary")
        self.font = pygame.font.SysFont("Segoe UI", 20)
        self.small_font = pygame.font.SysFont("Segoe UI", 15)

    def update(self, events):
        self.minus.update(events)
        self.plus.update(events)

    def draw(self, surface):
        draw_rounded_panel(surface, self.rect, fill=COLOR_SURFACE, border=COLOR_PANEL_BORDER, radius=12)
        draw_text(surface, self.label, self.font, COLOR_TEXT, (self.rect.x + 12, self.rect.y + 2))
        draw_text(surface, self.fmt.format(self.value_getter()), self.font, COLOR_ACCENT, (self.rect.x + 12, self.rect.y + 20))
        if self.hint:
            draw_text(surface, self.hint, self.small_font, COLOR_TEXT_MUTED, (self.rect.x + 60, self.rect.y + 23))
        self.minus.draw(surface)
        self.plus.draw(surface)

def getSurface(class_img, mask):
    mask_rgb = np.stack([mask, mask, mask], axis=-1)

    # Match camera/class image height to mask height
    if class_img.shape[0] != 640:
        class_img = cv2.resize(
            class_img,
            (640, 640),
            interpolation=cv2.INTER_AREA
        )

    combined = np.concatenate((class_img, mask_rgb), axis=1).astype(np.uint8)
    combined = np.transpose(combined, (1, 0, 2))
    return pygame.surfarray.make_surface(combined)


def toggleRawRecording(RECORD_RAW):
    RECORD_RAW.value = not RECORD_RAW.value


def toggleBBoxes(DRAW_BBOXES):
    DRAW_BBOXES.value = not DRAW_BBOXES.value
    print(DRAW_BBOXES)


def toggleRecording():
    global recording
    recording = not recording


def toggleVertCalibration(RUN_VERT_CALIBRATION):
    RUN_VERT_CALIBRATION.value = not RUN_VERT_CALIBRATION.value

# Rotate and Flip image Buttons!
#ROTATE, FLIP_H, FLIP_V
def toggleROTATE(ROTATE):
    old_rotation = ROTATE.value
    new_rotation = old_rotation+1
    if new_rotation > 4:
        new_rotation = 0
    ROTATE.value = new_rotation            
def toggleFLIP_H(FLIP_H):
    FLIP_H.value = not FLIP_H.value
def toggleFLIP_V(FLIP_V):
    FLIP_V.value = not FLIP_V.value


def increaseDelay(DELAY): DELAY.value = min(DELAY.value + 0.01, 5)
def decreaseDelay(DELAY): DELAY.value = max(DELAY.value - 0.01, 0.0)

def increaseCONF(CONF):
    if CONF.value < 0.10:
        CONF.value = min(round(CONF.value + 0.01, 3), 1.0)
    else:
        CONF.value = min(round(CONF.value + 0.05, 3), 1.0)

def decreaseCONF(CONF):
    if CONF.value <= 0.10:
        CONF.value = max(round(CONF.value - 0.01, 3), 0.0)
    else:
        CONF.value = max(round(CONF.value - 0.05, 3), 0.0)


def increaseIOU(IOU): IOU.value = min(IOU.value + 0.05, 1)
def decreaseIOU(IOU): IOU.value = max(IOU.value - 0.05, 0)
def increaseThreshold(THRESHOLD):
    # NIR SAM background threshold is in raw ALU units, not 0..1.
    THRESHOLD.value = min(round(float(THRESHOLD.value) + 50.0, 3), 2000.0)

def decreaseThreshold(THRESHOLD):
    # NIR SAM background threshold is in raw ALU units, not 0..1.
    THRESHOLD.value = max(round(float(THRESHOLD.value) - 50.0, 3), 0.0)
def increaseVertMovement(VERT_MOVEMENT): VERT_MOVEMENT.value = min(VERT_MOVEMENT.value + 1, 200)
def decreaseVertMovement(VERT_MOVEMENT): VERT_MOVEMENT.value = max(VERT_MOVEMENT.value - 1, 0)

def increaseRawFPS(RAW_RECORDING_FPS):
    RAW_RECORDING_FPS.value = min(round(RAW_RECORDING_FPS.value + 1, 2), 60.0)

def decreaseRawFPS(RAW_RECORDING_FPS):
    RAW_RECORDING_FPS.value = max(round(RAW_RECORDING_FPS.value - 1, 2), 1)

def increaseGuiFPS(GUI_RECORDING_FPS):
    GUI_RECORDING_FPS.value = min(round(GUI_RECORDING_FPS.value + 1, 2), 60.0)

def decreaseGuiFPS(GUI_RECORDING_FPS):
    GUI_RECORDING_FPS.value = max(round(GUI_RECORDING_FPS.value - 1, 2), 1)

def increaseCameraFPS(FPS):
    print("Not Implemented Yet")
    #FPS.value = min(int(FPS.value) + 1, 1000)

def decreaseCameraFPS(FPS):
    print("Not Implemented Yet")
    #FPS.value = max(int(FPS.value) - 1, 1)

def increaseGUI_DISPLAY_FPS():
    global GUI_DISPLAY_FPS
    GUI_DISPLAY_FPS = min(int(GUI_DISPLAY_FPS) + 1, 60)

def decreaseGUI_DISPLAY_FPS():
    global GUI_DISPLAY_FPS
    GUI_DISPLAY_FPS = max(int(GUI_DISPLAY_FPS) - 1, 20)

def chooseRawRecordingFolder(RECORDING_PATHS):
    path = easygui.diropenbox("Select RAW recording folder")
    if path:
        RECORDING_PATHS.raw_dir = path

def chooseGuiRecordingFolder(RECORDING_PATHS):
    path = easygui.diropenbox("Select GUI recording folder")
    if path:
        RECORDING_PATHS.gui_dir = path

def display(
    DISPLAY_QUEUE, STOP_FLAG, DELAY, TARGET_CLASSES, CONF, IOU,
    VORSCHUSS, NACHSCHUSS, BEISCHUSS, THRESHOLD,
    RECORD_RAW, DRAW_BBOXES, ALL_CLASSES, MODEL_NAMES, DETECTION_POS,
    VERT_MOVEMENT, CALIBRATED_VERT_MOVEMENT, RUN_VERT_CALIBRATION,
    ROTATE, FLIP_H, FLIP_V,
    RAW_RECORDING_FPS,GUI_RECORDING_FPS,RECORDING_PATHS,
    FPS=None, SCALEABLE=False, NIR_MODE=False, NIR_CLASSIFIER_KIND="", NIR_CLASS_COLORS=None,
    MODEL_INFO=None, MODEL_SWAP_QUEUE=None, N_CLASSES=None
):
    global recording
    try:
        pygame.init()
        pygame.font.init()
        
        if SCALEABLE:
            SCREEN = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # real window
            screen = pygame.Surface(WINDOW_SIZE).convert()                   # virtual canvas
        else:
            SCREEN = pygame.display.set_mode(WINDOW_SIZE)                    # real window
            screen = SCREEN                                                  # draw directly
        
        pygame.display.set_caption("Vision Control UI")
        clock = pygame.time.Clock()
        font_title = pygame.font.SysFont("Segoe UI", 28, bold=True)
        font_small = pygame.font.SysFont("Segoe UI", 16)
        font_small_bold = pygame.font.SysFont("Segoe UI", 15, bold=True)
    except Exception as e:
        print(f"Error in Display Process @ init Pygame: {e}")
        STOP_FLAG.set()
        easygui.exceptionbox(f"Error in Display Process @ init Pygame: {e}")
        return

    try:
        recording = False
        last_save_time = 0
        save_name = "gui_aufnahme"
    except Exception as e:
        print(f"Error in Display Process @ Define Recording Settings: {e}")
        STOP_FLAG.set()
        return

    try:
        def current_class_count():
            return len(class_toggle_checkboxes)

        def set_all_classes(value):
            for i in range(current_class_count()):
                TARGET_CLASSES[i] = int(value)

        def toggle_all_classes():
            n = current_class_count()
            all_on = all(TARGET_CLASSES[i] == 1 for i in range(n))
            set_all_classes(0 if all_on else 1)

        def toggle_target_classes(i):
            TARGET_CLASSES[i] = 1 - TARGET_CLASSES[i]

        # Sidebar layout: in RGB mode with hot-swap support, the sidebar is
        # split into a model panel (top) and the class list (below).
        HAS_MODEL_SWAP = MODEL_SWAP_QUEUE is not None
        if HAS_MODEL_SWAP:
            model_panel_rect = pygame.Rect(SIDEBAR_RECT.x, SIDEBAR_RECT.y, SIDEBAR_RECT.w, 118)
            class_selection_rect = pygame.Rect(
                SIDEBAR_RECT.x, SIDEBAR_RECT.y + 130,
                SIDEBAR_RECT.w, SIDEBAR_RECT.h - 130
            )
        else:
            model_panel_rect = None
            class_selection_rect = SIDEBAR_RECT.copy()

        row_height = 38
        list_start_y = class_selection_rect.y + 76

        def build_class_widgets(names_list):
            checkboxes, swatches = [], []
            for i, name in enumerate(names_list):
                checkbox_x = class_selection_rect.x + 18
                if NIR_MODE:
                    checkbox_x = class_selection_rect.x + 54
                    swatches.append(ClassColorSwatch(
                        rect=(class_selection_rect.x + 18, list_start_y + i * row_height, 24, 24),
                        class_index=i,
                        colors_proxy=NIR_CLASS_COLORS,
                    ))
                checkboxes.append(CheckBox(
                    rect=(checkbox_x, list_start_y + i * row_height, 24, 24),
                    text=str(name),
                    checked=(TARGET_CLASSES[i] == 1) if i < len(TARGET_CLASSES) else False,
                    callback=lambda i=i: toggle_target_classes(i),
                    font_size=18,
                    color=COLOR_TEXT,
                ))
            return checkboxes, swatches

        class_names = list(MODEL_NAMES.values()) if isinstance(MODEL_NAMES, dict) else list(MODEL_NAMES)
        class_toggle_checkboxes, class_color_swatches = build_class_widgets(class_names)

        # UI-side view of the active detection model (RGB hot-swap modes).
        ui_model = {
            "generation": -1,
            "kind": "",
            "path": "",
            "status": "",
        }
        last_model_poll = 0.0

        list_offset_y = 0
        scroll_velocity = 0.0
        is_dragging = False
        last_mouse_y = 0
        momentum = 0.90
    except Exception as e:
        print(f"Error in Display Process @ Define Class Toggle Buttons: {e}")
        easygui.exceptionbox(f"Error in Display Process @ Define Class Toggle Buttons: {e}")
        STOP_FLAG.set()
        return

    try:
        btn_calibrate = ToggleButton(470, 771-25, 170, 44, lambda: toggleVertCalibration(RUN_VERT_CALIBRATION), "Calibrate Speed", style="primary")
        btn_boxes = ToggleButton(650, 771-25, 170, 44, lambda: toggleBBoxes(DRAW_BBOXES), "Draw Boxes", style="primary")
        btn_raw = ToggleButton(650, 825-25, 170, 44, lambda: toggleRawRecording(RECORD_RAW), "Raw Recording", style="success")
        btn_record = ToggleButton(470, 825-25, 170, 44, toggleRecording, "Save GUI Frames", style="danger")
        buttons = [btn_calibrate, btn_boxes, btn_raw, btn_record]

        if NIR_MODE:
            # Tracking-based speed calibration and YOLO box drawing have no
            # effect in NIR line-scan mode. Worse, enabling calibration would
            # silently suppress ejection (createMask gates on it), so both are
            # disabled instead of left as dead/dangerous controls.
            btn_calibrate.enabled = False
            btn_boxes.enabled = False
        
        
        
        btn_rotate = Button(830, 771-25, 54, 44, lambda: toggleROTATE(ROTATE), "Rot", style="primary")
        buttons.append(btn_rotate)
        btn_flip_h = ToggleButton(830, 825-25, 54, 44, lambda: toggleFLIP_H(FLIP_H), "FlipH", style="success")
        buttons.append(btn_flip_h)
        btn_flip_v = ToggleButton(830, 825+54-25, 54, 44, lambda: toggleFLIP_V(FLIP_V), "FlipV", style="success")
        buttons.append(btn_flip_v)

        # Buttons that live inside the sidebar panels; drawn after the panels
        # so the panel background does not paint over them.
        panel_buttons = []

        btn_classes_all = Button(
            class_selection_rect.right - 156, class_selection_rect.y + 12, 64, 30,
            lambda: set_all_classes(1), "All", style="primary"
        )
        btn_classes_none = Button(
            class_selection_rect.right - 84, class_selection_rect.y + 12, 64, 30,
            lambda: set_all_classes(0), "None", style="primary"
        )
        panel_buttons.extend([btn_classes_all, btn_classes_none])

        btn_load_model = None
        
        if NIR_MODE:
            filetypes = ["*.joblib", "*.pkl"]
            msg = "Select a NIR sklearn/joblib classifier"
        else:
            filetypes = ["*.pt", "*.onnx", "*.engine"]
            msg = "Select an Ultralytics YOLO or RT-DETR model (auto-detected)"
        
        if HAS_MODEL_SWAP:
            def request_model_swap():
                path = easygui.fileopenbox(
                    msg=msg,
                    title="Load detection model",
                    default=filetypes[0],                                        
                    filetypes=filetypes,
                )
                if path:
                    try:
                        MODEL_SWAP_QUEUE.put_nowait(path)
                        ui_model["status"] = f"requested {os.path.basename(path)} ..."
                    except Exception as exc:
                        print(f"[UI] Could not request model swap: {exc}")

            btn_load_model = Button(
                model_panel_rect.right - 162, model_panel_rect.y + 14, 146, 40,
                request_model_swap, "Load Model", style="primary"
            )
            panel_buttons.append(btn_load_model)
        
        if NIR_MODE:
            controls = [
                ValueStepper("Delay", 20, 690, 430, lambda: DELAY.value, lambda: decreaseDelay(DELAY), lambda: increaseDelay(DELAY), "[↑/↓ 0.01, ←/→ 0.001]", "{:.3f}s"),
                ValueStepper("SAM Background Threshold", 20, 746, 430, lambda: THRESHOLD.value, lambda: decreaseThreshold(THRESHOLD), lambda: increaseThreshold(THRESHOLD), "ALU mean [O/P ±50]", "{:.0f}"),
                ValueStepper("Beischuss / Width Expansion", 20, 802, 430, lambda: BEISCHUSS.value, lambda: setattr(BEISCHUSS, 'value', max(BEISCHUSS.value - 1, 0)), lambda: setattr(BEISCHUSS, 'value', min(BEISCHUSS.value + 1, 100)), "px/nozzle dilation", "{:.0f}"),
                #ValueStepper("Vertical Movement", 470, 690, 410, lambda: VERT_MOVEMENT.value, lambda: decreaseVertMovement(VERT_MOVEMENT), lambda: increaseVertMovement(VERT_MOVEMENT), "px/frame", "{:.0f}"),
                ValueStepper("NIR Confidence Threshold", 470, 690, 410, lambda: CONF.value, lambda: decreaseCONF(CONF), lambda: increaseCONF(CONF), "below -> Not classified [Num +/-]", "{:.2f}"),
            ]
        else:
            controls = [
                ValueStepper("Delay", 20, 690, 430, lambda: DELAY.value, lambda: decreaseDelay(DELAY), lambda: increaseDelay(DELAY), "[↑/↓ 0.01, ←/→ 0.001]", "{:.3f}s"),
                ValueStepper("Prediction Confidence", 20, 746, 430, lambda: CONF.value, lambda: decreaseCONF(CONF), lambda: increaseCONF(CONF), "[Num +/-]", "{:.2f}"),
                ValueStepper("Prediction IoU", 20, 802, 430, lambda: IOU.value, lambda: decreaseIOU(IOU), lambda: increaseIOU(IOU), "[PgUp/PgDn]", "{:.2f}"),
                ValueStepper("Vertical Movement", 470, 690, 410, lambda: VERT_MOVEMENT.value, lambda: decreaseVertMovement(VERT_MOVEMENT), lambda: increaseVertMovement(VERT_MOVEMENT), "px/frame", "{:.0f}"),
            ]
        
        # Right
        folder_w = 130
        folder_spacing = 20
        gui_folder_x = WINDOW_SIZE[0] - folder_w - folder_spacing
        
        # Mid-right        
        raw_folder_x = gui_folder_x - folder_w - folder_spacing
        
        # === Lower recording toolbar ===
        recording_y = 970 
        recording_controls_y = recording_y 
        recording_buttons_y = recording_y
        
        margin = 20
        spacing = 20
        n_items = 4
        
        available_width = raw_folder_x - 2 * margin - (n_items - 1) * spacing
        item_w = available_width // n_items
        
        raw_fps_x = margin
        gui_fps_x = raw_fps_x + item_w + spacing
        camera_fps_x = gui_fps_x + item_w + spacing
        gui_display_fps_x = camera_fps_x + item_w + spacing
        
        raw_fps_w = gui_fps_w = camera_fps_w = gui_display_fps_w = item_w
        
        
        
        
        
        
        btn_raw_folder = Button(
            raw_folder_x,
            recording_buttons_y,
            folder_w,
            44,
            lambda: chooseRawRecordingFolder(RECORDING_PATHS),
            "Raw Folder",
            style="primary"
        )
        
        btn_gui_folder = Button(
            gui_folder_x,
            recording_buttons_y,
            folder_w,
            44,
            lambda: chooseGuiRecordingFolder(RECORDING_PATHS),
            "GUI Folder",
            style="primary"
        )
        
        buttons.extend([btn_raw_folder, btn_gui_folder])
        
        controls.extend([
            ValueStepper(
                "Raw Recording FPS",
                raw_fps_x,
                recording_controls_y,
                raw_fps_w,
                lambda: RAW_RECORDING_FPS.value,
                lambda: decreaseRawFPS(RAW_RECORDING_FPS),
                lambda: increaseRawFPS(RAW_RECORDING_FPS),
                "camera fps",
                "{:.1f}"
            ),
        
            ValueStepper(
                "GUI Recording FPS",
                gui_fps_x,
                recording_controls_y,
                gui_fps_w,
                lambda: GUI_RECORDING_FPS.value,
                lambda: decreaseGuiFPS(GUI_RECORDING_FPS),
                lambda: increaseGuiFPS(GUI_RECORDING_FPS),
                "ui fps",
                "{:.1f}"
            ),            
            ValueStepper(
                "GUI Display FPS",
                gui_display_fps_x,
                recording_controls_y,
                gui_display_fps_w,
                lambda: GUI_DISPLAY_FPS,
                decreaseGUI_DISPLAY_FPS,
                increaseGUI_DISPLAY_FPS,                
                "display refresh",
                "{:.0f}"
            ),        
        ])

        

        if FPS is not None:
            controls.append(
                ValueStepper(
                    "Camera FPS",
                    camera_fps_x,
                    recording_controls_y,
                    camera_fps_w,
                    lambda: FPS.value,
                    lambda: decreaseCameraFPS(FPS),
                    lambda: increaseCameraFPS(FPS),
                    "Basler",
                    "{:.0f}"
                )
            )
        
    except Exception as e:
        print(f"Error in Display Process @ Define Buttons/Controls: {e}")
        STOP_FLAG.set()
        return

    frame = None
    nir_stats = {}
    esc_armed_until = 0.0
    show_help = False
    last_frame_received = time.time()
    while not STOP_FLAG.is_set():
        try:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    STOP_FLAG.set(); break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: increaseDelay(DELAY)
                    elif event.key == pygame.K_DOWN: decreaseDelay(DELAY)
                    elif event.key == pygame.K_RIGHT: DELAY.value = min(DELAY.value + 0.001, 5.0)
                    elif event.key == pygame.K_LEFT: DELAY.value = max(DELAY.value - 0.001, 0.0)
                    elif event.key == pygame.K_0: toggle_all_classes()
                    elif event.key == pygame.K_1: TARGET_CLASSES[0] = 1 - TARGET_CLASSES[0]
                    elif event.key == pygame.K_2 and len(TARGET_CLASSES) > 1: TARGET_CLASSES[1] = 1 - TARGET_CLASSES[1]
                    elif event.key == pygame.K_KP_PLUS: increaseCONF(CONF)
                    elif event.key == pygame.K_KP_MINUS: decreaseCONF(CONF)
                    elif event.key == pygame.K_PAGEUP: increaseIOU(IOU)
                    elif event.key == pygame.K_PAGEDOWN: decreaseIOU(IOU)
                    elif event.key == pygame.K_q: VORSCHUSS.value = min(VORSCHUSS.value + 1, 100)
                    elif event.key == pygame.K_a: VORSCHUSS.value = max(VORSCHUSS.value - 1, 0)
                    elif event.key == pygame.K_w: NACHSCHUSS.value = min(NACHSCHUSS.value + 1, 100)
                    elif event.key == pygame.K_s: NACHSCHUSS.value = max(NACHSCHUSS.value - 1, 0)
                    elif event.key == pygame.K_x: BEISCHUSS.value = min(BEISCHUSS.value + 1, 100)
                    elif event.key == pygame.K_y: BEISCHUSS.value = max(BEISCHUSS.value - 1, 0)
                    elif event.key == pygame.K_p: increaseThreshold(THRESHOLD)
                    elif event.key == pygame.K_o: decreaseThreshold(THRESHOLD)
                    elif event.key == pygame.K_v: increaseVertMovement(VERT_MOVEMENT)
                    elif event.key == pygame.K_b: decreaseVertMovement(VERT_MOVEMENT)
                    elif event.key == pygame.K_SPACE: recording = not recording
                    elif event.key in (pygame.K_F1, pygame.K_h): show_help = not show_help
                    elif event.key == pygame.K_ESCAPE:
                        # Double-press guard: a single accidental ESC must not
                        # shut down the live sorting pipeline.
                        now = time.time()
                        if now < esc_armed_until:
                            STOP_FLAG.set(); break
                        esc_armed_until = now + 1.5

            try:
                item = DISPLAY_QUEUE.get(timeout=0.01)
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    nozzle_mask, class_image, nir_stats = item[0], item[1], item[2] or {}
                else:
                    nozzle_mask, class_image = item
                    nir_stats = {}

                # Keep only the newest available frame
                while True:
                    try:
                        item = DISPLAY_QUEUE.get_nowait()
                        if isinstance(item, (tuple, list)) and len(item) >= 3:
                            nozzle_mask, class_image, nir_stats = item[0], item[1], item[2] or {}
                        else:
                            nozzle_mask, class_image = item
                            nir_stats = {}
                    except Empty:
                        break      

                display_mask = cv2.resize(nozzle_mask, (256, 640), interpolation=cv2.INTER_NEAREST)
                class_image = cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB)
                frame = getSurface(class_image, display_mask)
                last_frame_received = time.time()
            except Empty:
                pass

            # Poll the producer's model info (hot-swap status / class names).
            # Throttled: a manager-dict read is one IPC round trip.
            if MODEL_INFO is not None and time.time() - last_model_poll > 0.5:
                last_model_poll = time.time()
                try:
                    ui_model["kind"] = str(MODEL_INFO.get("kind", ""))
                    ui_model["path"] = str(MODEL_INFO.get("path", ""))
                    ui_model["status"] = str(MODEL_INFO.get("status", ""))
                    gen = int(MODEL_INFO.get("generation", 0))
                    if gen != ui_model["generation"]:
                        ui_model["generation"] = gen
                        names_map = dict(MODEL_INFO.get("names", {}))
                        names_list = [str(names_map[k]) for k in sorted(names_map)]
                        if names_list:
                            class_toggle_checkboxes, class_color_swatches = build_class_widgets(names_list)
                            list_offset_y = 0
                            scroll_velocity = 0.0
                except Exception as exc:
                    print(f"[UI] Model info poll failed: {exc}")

            if frame is None:
                clock.tick(GUI_DISPLAY_FPS); continue

            btn_calibrate.clicked = bool(RUN_VERT_CALIBRATION.value)
            btn_boxes.clicked = bool(DRAW_BBOXES.value)
            btn_raw.clicked = bool(RECORD_RAW.value)
            btn_record.clicked = bool(recording)
            btn_flip_h.clicked = bool(FLIP_H.value)
            btn_flip_v.clicked = bool(FLIP_V.value)
            for button in buttons: button.update(events)
            for button in panel_buttons: button.update(events)
            for control in controls: control.update(events)

            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            list_top = class_selection_rect.y + 72
            list_bottom = class_selection_rect.bottom - 12
            content_height = len(class_toggle_checkboxes) * row_height
            visible_height = list_bottom - list_top
            max_negative_offset = min(0, visible_height - content_height - 4)
            if class_selection_rect.collidepoint(mouse_pos):
                if mouse_pressed and not is_dragging:
                    is_dragging = True; last_mouse_y = mouse_pos[1]; scroll_velocity = 0
                elif mouse_pressed and is_dragging:
                    dy = mouse_pos[1] - last_mouse_y; last_mouse_y = mouse_pos[1]; list_offset_y += dy; scroll_velocity = dy
                elif not mouse_pressed:
                    is_dragging = False
            elif not mouse_pressed:
                is_dragging = False
            for event in events:
                if event.type == pygame.MOUSEWHEEL and class_selection_rect.collidepoint(mouse_pos):
                    list_offset_y += event.y * 28; scroll_velocity = event.y * 10
            if not is_dragging and abs(scroll_velocity) > 0.3:
                list_offset_y += scroll_velocity; scroll_velocity *= momentum
            elif not is_dragging:
                scroll_velocity = 0
            list_offset_y = clamp(list_offset_y, max_negative_offset, 0)
            for i, checkbox in enumerate(class_toggle_checkboxes):
                checkbox.checked = bool(TARGET_CLASSES[i])
                x = class_selection_rect.x + (54 if NIR_MODE else 18)
                checkbox.rect.topleft = (x, list_start_y + i * row_height + int(list_offset_y))
                checkbox.update(events)
                if NIR_MODE and i < len(class_color_swatches):
                    swatch = class_color_swatches[i]
                    swatch.rect.topleft = (class_selection_rect.x + 18, list_start_y + i * row_height + int(list_offset_y))
                    swatch.update(events)

            screen.fill(COLOR_BG)            
            screen.blit(frame, FRAME_DEST)            
            if not NIR_MODE:
                # Detection line: the y position where the mask is sampled for
                # ejection. NIR line-scan mode has no meaningful y position, so
                # no line is drawn there.
                pygame.draw.line(screen, COLOR_CHECK, (0, DETECTION_POS), (640, DETECTION_POS), 2)
                draw_text(screen, "detection line", font_small, COLOR_CHECK, (8, DETECTION_POS - 22))

            info_panel = pygame.Rect(10, 645, 900, SIDEBAR_RECT.bottom - 645)
            draw_rounded_panel(screen, info_panel, fill=COLOR_SURFACE, border=COLOR_PANEL_BORDER, radius=16)
            draw_text(screen, "Control Panel", font_title, COLOR_TEXT, (28, 654))
            
            recording_panel = pygame.Rect(10, 960, 900, 90)
            draw_rounded_panel(screen, recording_panel, fill=COLOR_SURFACE_2)
            
            for control in controls: control.draw(screen)

            if NIR_MODE:
                metrics = [
                    ("Mode", "NIR", NIR_CLASSIFIER_KIND or "line classifier"),
                    ("BG Threshold", f"{THRESHOLD.value:.0f}", "ALU mean"),
                    ("Beischuss", f"{BEISCHUSS.value:.0f}", "[Y/X]"),
                    ("Classes", f"{sum(1 for v in TARGET_CLASSES if v)}", "eject enabled"),
                    ("Delay", f"{DELAY.value:.3f}s", "nozzle timing"),
                ]
            else:
                metrics = [
                    ("Suggested Move", f"{CALIBRATED_VERT_MOVEMENT.value:.1f}", "track estimate"),
                    ("Calibration", "ON" if RUN_VERT_CALIBRATION.value else "OFF", "manual apply"),
                    ("Vorschuss", f"{VORSCHUSS.value:.0f}", "[Q/A]"),
                    ("Nachschuss", f"{NACHSCHUSS.value:.0f}", "[W/S]"),
                    ("Beischuss", f"{BEISCHUSS.value:.0f}", "[Y/X]"),
                ]
            card_x = 20
            for title, value, hint in metrics:
                card_width = 150
                card_spacing = 10
                rect = pygame.Rect(card_x, 858, card_width, 92)
                draw_rounded_panel(screen, rect, fill=COLOR_SURFACE_2, border=COLOR_PANEL_BORDER, radius=12)
                draw_text(screen, title, font_small, COLOR_TEXT_MUTED, (rect.x + 12, rect.y + 10))
                draw_text(screen, value, font_title, COLOR_ACCENT, (rect.x + 12, rect.y + 30))
                draw_text(screen, hint, font_small, COLOR_TEXT_MUTED, (rect.x + 12, rect.y + 62))
                card_x += card_width+card_spacing
            for button in buttons: button.draw(screen)

            # --- Model panel (RGB hot-swap modes) ---
            if HAS_MODEL_SWAP:
                draw_rounded_panel(screen, model_panel_rect, fill=COLOR_SURFACE, border=COLOR_PANEL_BORDER, radius=16)
                draw_text(screen, "Detection Model", font_title, COLOR_TEXT, (model_panel_rect.x + 16, model_panel_rect.y + 8))
                model_kind = ui_model["kind"] or "?"
                kind_color = (150, 96, 230) if model_kind == "RTDETR" else COLOR_PRIMARY
                kind_rect = pygame.Rect(model_panel_rect.x + 16, model_panel_rect.y + 48, 76, 24)
                pygame.draw.rect(screen, kind_color, kind_rect, border_radius=12)
                draw_text(screen, model_kind, font_small_bold, COLOR_TEXT, kind_rect.center, align="center")
                model_base = os.path.basename(ui_model["path"]) or "-"
                if len(model_base) > 36:
                    model_base = model_base[:33] + "..."
                draw_text(screen, model_base, font_small, COLOR_ACCENT, (kind_rect.right + 10, model_panel_rect.y + 51))
                status_text = ui_model["status"] or ""
                status_color = COLOR_DANGER if "failed" in status_text.lower() else COLOR_TEXT_MUTED
                if len(status_text) > 52:
                    status_text = status_text[:49] + "..."
                draw_text(screen, status_text, font_small, status_color, (model_panel_rect.x + 16, model_panel_rect.y + 84))

            # --- Class selection panel ---
            draw_rounded_panel(screen, class_selection_rect, fill=COLOR_SURFACE, border=COLOR_PANEL_BORDER, radius=16)
            draw_text(screen, "NIR Eject Class Selection" if NIR_MODE else "Target Class Selection", font_title, COLOR_TEXT, (class_selection_rect.x + 16, class_selection_rect.y + 12))
            n_cls = len(class_toggle_checkboxes)
            n_on = sum(1 for i in range(min(n_cls, len(TARGET_CLASSES))) if TARGET_CLASSES[i] == 1)
            if NIR_MODE:
                draw_text(screen, f"{n_on}/{n_cls} enabled  •  color square: left-click cycle, right-click hex", font_small, COLOR_TEXT_MUTED, (class_selection_rect.x + 16, class_selection_rect.y + 48))
            else:
                draw_text(screen, f"{n_on}/{n_cls} enabled  •  [0] toggles all", font_small, COLOR_TEXT_MUTED, (class_selection_rect.x + 16, class_selection_rect.y + 48))
            clip_rect = pygame.Rect(class_selection_rect.x + 8, class_selection_rect.y + 68, class_selection_rect.w - 16, class_selection_rect.h - 78)
            old_clip = screen.get_clip(); screen.set_clip(clip_rect)
            if NIR_MODE:
                for swatch in class_color_swatches:
                    if swatch.rect.bottom >= clip_rect.top - 4 and swatch.rect.top <= clip_rect.bottom + 4:
                        swatch.draw(screen)
            for checkbox in class_toggle_checkboxes:
                if checkbox.rect.bottom >= clip_rect.top - 4 and checkbox.rect.top <= clip_rect.bottom + 4:
                    checkbox.draw(screen)
            screen.set_clip(old_clip)
            if content_height > visible_height:
                track_rect = pygame.Rect(class_selection_rect.right - 10, clip_rect.y + 8, 4, clip_rect.h - 16)
                pygame.draw.rect(screen, COLOR_PANEL_BORDER, track_rect, border_radius=2)
                thumb_h = max(60, int(track_rect.h * (visible_height / max(content_height, 1))))
                scroll_ratio = 0 if max_negative_offset == 0 else (list_offset_y / max_negative_offset)
                thumb_y = track_rect.y + int((track_rect.h - thumb_h) * scroll_ratio)
                pygame.draw.rect(screen, COLOR_ACCENT, pygame.Rect(track_rect.x, thumb_y, track_rect.w, thumb_h), border_radius=2)

            for button in panel_buttons: button.draw(screen)

            # --- Status bar over the live view ---
            def draw_status_pill(x, text, color, blink=False):
                label = font_small_bold.render(text, True, COLOR_TEXT)
                pad_left = 26 if blink else 10
                rect = pygame.Rect(x, 10, label.get_width() + pad_left + 10, 26)
                pill = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(pill, (*color, 225), pill.get_rect(), border_radius=13)
                screen.blit(pill, rect.topleft)
                if blink and int(time.time() * 2) % 2 == 0:
                    pygame.draw.circle(screen, COLOR_TEXT, (rect.x + 14, rect.centery), 4)
                screen.blit(label, (rect.x + pad_left, rect.y + 4))
                return rect.right + 8

            pill_x = 10
            if time.time() < esc_armed_until:
                pill_x = draw_status_pill(pill_x, "PRESS ESC AGAIN TO QUIT", COLOR_DANGER)
            if recording:
                pill_x = draw_status_pill(pill_x, "REC GUI", COLOR_DANGER, blink=True)
            if RECORD_RAW.value:
                pill_x = draw_status_pill(pill_x, "REC RAW", COLOR_DANGER, blink=True)
            if NIR_MODE:
                pill_x = draw_status_pill(pill_x, "NIR LINE-SCAN", COLOR_PRIMARY)

                nir_lps = float(nir_stats.get("nir_lps", 0.0) or 0.0)
                nir_cls_ms = float(nir_stats.get("nir_cls_ms", 0.0) or 0.0)
                nir_line_ms = float(nir_stats.get("nir_line_ms", 0.0) or 0.0)
                nir_record_ms = float(nir_stats.get("nir_read_ms",0.0)or 0.0)

                
                if nir_lps > 0.0:
                    pill_x = draw_status_pill(pill_x, f"LPS {nir_lps:.0f}", COLOR_SUCCESS)
                if nir_cls_ms > 0.0:
                    if nir_line_ms > 0.0:
                        pill_x = draw_status_pill(
                            pill_x,
                            f"CLS {nir_cls_ms:.2f}/{nir_line_ms:.2f} ms",
                            COLOR_SUCCESS if nir_cls_ms < nir_line_ms else COLOR_DANGER,
                        )
                    else:
                        pill_x = draw_status_pill(pill_x, f"CLS {nir_cls_ms:.2f} ms", COLOR_SUCCESS)
                if nir_record_ms > 0.0:
                    pill_x = draw_status_pill(pill_x, f"READ {nir_record_ms:.2f} ms", COLOR_SUCCESS)

            frame_age = time.time() - last_frame_received
            if frame_age > 1.0:
                pill_x = draw_status_pill(pill_x, f"NO NEW FRAMES {frame_age:.0f}s", COLOR_DANGER, blink=True)
            draw_text(screen, f"UI {clock.get_fps():.0f} fps  •  F1/H help  •  ESC x2 quit", font_small, COLOR_TEXT_MUTED, (12, 42))

            # --- Help overlay ---
            if show_help:
                overlay = pygame.Surface((640, 640), pygame.SRCALPHA)
                overlay.fill((10, 12, 16, 220))
                screen.blit(overlay, (0, 0))
                draw_text(screen, "Keyboard Shortcuts", font_title, COLOR_TEXT, (24, 18))
                help_lines = [
                    ("ESC  ESC", "quit (press twice within 1.5 s)"),
                    ("F1 / H", "toggle this help"),
                    ("SPACE", "toggle GUI frame recording"),
                    ("0", "toggle all classes"),
                    ("1 / 2", "toggle class 1 / class 2"),
                    ("UP/DOWN", "delay +/- 0.01 s"),
                    ("LEFT/RIGHT", "delay +/- 0.001 s"),
                    ("Num + / -", "prediction confidence"),
                    ("PgUp / PgDn", "prediction IoU"),
                    ("Q / A", "Vorschuss + / -"),
                    ("W / S", "Nachschuss + / -"),
                    ("X / Y", "Beischuss + / -"),
                    ("P / O", "NIR background threshold + / -" if NIR_MODE else "threshold + / -"),
                    ("V / B", "vertical movement + / -"),
                ]
                hy = 66
                for keys_txt, desc in help_lines:
                    draw_text(screen, keys_txt, font_small_bold, COLOR_ACCENT, (24, hy))
                    draw_text(screen, desc, font_small, COLOR_TEXT, (190, hy))
                    hy += 30

            if SCALEABLE:
                scaled = pygame.transform.smoothscale(screen, SCREEN.get_size())
                SCREEN.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(GUI_DISPLAY_FPS)
        except Exception as e:
            print(f"Error in Screen Draw Block: {e}")
            traceback.print_exc()
            STOP_FLAG.set()
            pygame.quit()
            return

        # Recording GUI Frames
        try:
            gui_fps = max(0.1, GUI_RECORDING_FPS.value)
            current_time = time.time()
        
            if frame is not None and recording and current_time > last_save_time + (1.0 / gui_fps):
                os.makedirs(RECORDING_PATHS.gui_dir, exist_ok=True)
        
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
                pygame.image.save(
                    frame,
                    os.path.join(
                        RECORDING_PATHS.gui_dir,
                        f"{save_name}_{timestamp}.jpg"
                    )
                )
        
                print(f"Saved GUI Frame: {timestamp}")
                last_save_time = current_time
        
        except Exception as e:
            print(f"[Display] Error in Saving of GUI Image: {e}")
    pygame.quit()
