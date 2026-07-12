# """
# QQ 窗口检测与截图模块
# ====================
#
# 使用 ctypes 直接调用 Windows API（user32.dll）来：
#   1. 枚举所有顶层窗口，按标题查找 QQ 聊天窗口
#   2. 获取窗口位置和尺寸
#   3. 根据配置的比例计算聊天消息区域
#   4. 截取聊天区域的屏幕截图
#   5. 将 QQ 窗口切换到前台（用于后续键盘输入）
#
# 设计目标：不依赖 pywin32 / pyautogui 等额外包，ctypes 是 Python 内置模块。
# 键盘输入部分使用已安装的 `keyboard` 包（在 qq_bot.py 中调用）。
#
# 支持的 Qt 版本：QQ NT v9.x（Windows 11，100% DPI）
# """
#
# import ctypes
# import time
# from ctypes import wintypes
# from PIL import ImageGrab, Image
#
# # ============================================================================
# # 一、Windows API 类型与函数声明
# # ============================================================================
#
# user32 = ctypes.windll.user32
# kernel32 = ctypes.windll.kernel32
#
# # ---- 类型别名 ----
# HWND   = wintypes.HWND
# LPARAM = wintypes.LPARAM
# RECT   = wintypes.RECT
# BOOL   = wintypes.BOOL
# DWORD  = wintypes.DWORD
#
# # ---- EnumWindows 回调函数原型 ----
# # BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam)
# WNDENUMPROC = ctypes.WINFUNCTYPE(BOOL, HWND, LPARAM)
#
# # ---- 函数签名 ----
# user32.EnumWindows.argtypes  = [WNDENUMPROC, LPARAM]
# user32.EnumWindows.restype   = BOOL
#
# user32.IsWindowVisible.argtypes = [HWND]
# user32.IsWindowVisible.restype  = BOOL
#
# user32.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
# user32.GetWindowTextW.restype  = ctypes.c_int
#
# user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(RECT)]
# user32.GetWindowRect.restype  = BOOL
#
# user32.SetForegroundWindow.argtypes = [HWND]
# user32.SetForegroundWindow.restype  = BOOL
#
# user32.GetForegroundWindow.argtypes = []
# user32.GetForegroundWindow.restype  = HWND
#
# user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(DWORD)]
# user32.GetWindowThreadProcessId.restype  = DWORD
#
# user32.AttachThreadInput.argtypes = [DWORD, DWORD, BOOL]
# user32.AttachThreadInput.restype  = BOOL
#
# kernel32.GetCurrentThreadId.argtypes = []
# kernel32.GetCurrentThreadId.restype  = DWORD
#
# user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
# user32.SetCursorPos.restype  = BOOL
#
# user32.mouse_event.argtypes = [DWORD, DWORD, DWORD, DWORD, LPARAM]
# user32.mouse_event.restype  = None
#
# # SetProcessDPIAware（Windows Vista+）
# try:
#     user32.SetProcessDPIAware.argtypes = []
#     user32.SetProcessDPIAware.restype  = BOOL
#     _DPI_AWARE_SET = False
# except AttributeError:
#     _DPI_AWARE_SET = True  # 系统不支持，跳过
#
# # 鼠标事件常量
# MOUSEEVENTF_LEFTDOWN  = 0x0002
# MOUSEEVENTF_LEFTUP    = 0x0004
#
# # 键盘事件常量（用于 Alt 键绕过前台锁定）
# VK_MENU               = 0x12
# KEYEVENTF_KEYUP       = 0x0002
#
# user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, DWORD, LPARAM]
# user32.keybd_event.restype  = None
#
#
# # ============================================================================
# # 二、窗口查找辅助
# # ============================================================================
#
# def _enum_visible_windows():
#     """
#     枚举所有可见的顶层窗口，返回 (hwnd, title) 列表。
#     使用 EnumWindows API + 回调函数。
#     """
#     results = []
#
#     def callback(hwnd, lparam):
#         if user32.IsWindowVisible(hwnd):
#             # 获取窗口标题（最大 512 字符）
#             buf = ctypes.create_unicode_buffer(512)
#             user32.GetWindowTextW(hwnd, buf, 511)
#             title = buf.value
#             if title and title.strip():
#                 results.append((hwnd, title.strip()))
#         return True  # 继续枚举
#
#     user32.EnumWindows(WNDENUMPROC(callback), 0)
#     return results
#
#
# def find_window_by_keywords(keywords, exact_title=None):
#     """
#     按标题查找窗口。
#
#     参数：
#         keywords    : list[str] —— 标题关键词列表（模糊匹配，包含任一即命中）
#         exact_title : str|None  —— 精确标题（优先匹配）
#
#     返回：
#         (hwnd, title) —— 匹配到的窗口句柄和标题
#         (None, None)  —— 未找到
#     """
#     windows = _enum_visible_windows()
#
#     # 优先级 1：精确匹配
#     if exact_title:
#         for hwnd, title in windows:
#             if title == exact_title:
#                 return hwnd, title
#
#     # 优先级 2：关键词匹配
#     if keywords:
#         for hwnd, title in windows:
#             for kw in keywords:
#                 if kw.lower() in title.lower():
#                     return hwnd, title
#
#     return None, None
#
#
# # ============================================================================
# # 三、QQWindowCapture 类
# # ============================================================================
#
# class QQWindowCapture:
#     """
#     QQ 窗口检测与聊天区域截图。
#
#     用法示例：
#         cap = QQWindowCapture(keywords=["QQ"], region_ratios={...})
#         hwnd, title = cap.find_window()
#         if hwnd:
#             img = cap.capture_chat(hwnd)
#             img.save("test.png")
#     """
#
#     def __init__(self, title_keywords=None, exact_title=None, region_ratios=None):
#         """
#         参数：
#             title_keywords : list[str] —— 窗口标题关键词
#             exact_title    : str|None  —— 精确窗口标题
#             region_ratios  : dict      —— 聊天区域比例 {'left','top','right','bottom'}
#         """
#         self.title_keywords = title_keywords or ["QQ", "腾讯QQ"]
#         self.exact_title = exact_title
#         # 默认区域：左下角对方消息，约两行高度
#         self.region_ratios = region_ratios or {
#             "left": 0.26,  # 左边界：跳过左侧会话列表，保持不变
#             "top": 0.71,  # 上边界：控制两行消息高度，保持不变
#             "right": 0.57,  # 右边界：左移，完全避开右侧自己的消息区域
#             "bottom": 0.78  # 下边界：紧贴输入框上方，保持不变
#         }
#
#         self._dpi_aware_set = False
#         self._ensure_dpi_aware()
#
#     # ---- DPI 感知 ----
#     def _ensure_dpi_aware(self):
#         """设置进程 DPI 感知，避免高 DPI 下坐标缩放问题。"""
#         global _DPI_AWARE_SET
#         if not _DPI_AWARE_SET:
#             try:
#                 user32.SetProcessDPIAware()
#             except Exception:
#                 pass
#             _DPI_AWARE_SET = True
#
#     # ---- 窗口查找 ----
#     def find_window(self):
#         """查找 QQ 窗口，返回 (hwnd, title) 或 (None, None)。"""
#         return find_window_by_keywords(self.title_keywords, self.exact_title)
#
#     def get_window_title(self, hwnd):
#         """获取指定窗口句柄的标题文本。"""
#         buf = ctypes.create_unicode_buffer(512)
#         user32.GetWindowTextW(hwnd, buf, 511)
#         return buf.value
#
#     # ---- 坐标计算 ----
#     def get_window_rect(self, hwnd):
#         """
#         获取窗口在屏幕上的矩形坐标。
#         返回 (left, top, right, bottom)，失败返回 None。
#         """
#         rect = RECT()
#         if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
#             return (rect.left, rect.top, rect.right, rect.bottom)
#         return None
#
#     def get_chat_bbox(self, hwnd):
#         """
#         根据窗口矩形和配置的比例，计算聊天消息区域的屏幕坐标。
#         返回 (left, top, right, bottom)。
#         """
#         rect = self.get_window_rect(hwnd)
#         if rect is None:
#             return None
#
#         left, top, right, bottom = rect
#         width  = right - left
#         height = bottom - top
#
#         chat_left   = int(left   + width  * self.region_ratios["left"])
#         chat_top    = int(top    + height * self.region_ratios["top"])
#         chat_right  = int(left   + width  * self.region_ratios["right"])
#         chat_bottom = int(top    + height * self.region_ratios["bottom"])
#
#         return (chat_left, chat_top, chat_right, chat_bottom)
#
#     def get_input_box_position(self, hwnd):
#         """
#         计算输入框的推荐点击位置（窗口底部中央偏下）。
#         返回 (screen_x, screen_y)。
#         """
#         rect = self.get_window_rect(hwnd)
#         if rect is None:
#             return None
#         left, top, right, bottom = rect
#         x = left + int((right - left) * 0.50)
#         y = top  + int((bottom - top) * 0.92)
#         return (x, y)
#
#     # ---- 截图 ----
#     def capture_chat(self, hwnd):
#         """
#         截取聊天区域的屏幕截图。
#
#         参数：
#             hwnd : 窗口句柄
#
#         返回：
#             PIL.Image —— 聊天区域的截图
#             None      —— 截取失败（窗口无效或区域异常）
#
#         注意：ImageGrab.grab 直接截取屏幕指定区域，QQ 窗口无需在最前台，
#         但该区域不能被其他窗口遮挡，否则会截到遮挡内容。
#         """
#         bbox = self.get_chat_bbox(hwnd)
#         if bbox is None:
#             return None
#
#         left, top, right, bottom = bbox
#         if left >= right or top >= bottom:
#             return None
#
#         try:
#             img = ImageGrab.grab(bbox=bbox)
#             return img
#         except Exception:
#             return None
#
#     # ---- 窗口聚焦 ----
#     def focus_window(self, hwnd):
#         """
#         使用 AttachThreadInput 技术可靠地将窗口切换到前台。
#
#         原理：
#           Windows 阻止后台进程通过 SetForegroundWindow 抢占前台。
#           AttachThreadInput 将当前线程连接到前台线程后，
#           Windows 认为两个线程属于同一输入队列，从而允许切换。
#         """
#         # 获取当前前台窗口和线程
#         foreground_hwnd = user32.GetForegroundWindow()
#         current_thread_id = kernel32.GetCurrentThreadId()
#         foreground_thread_id = user32.GetWindowThreadProcessId(
#             foreground_hwnd, None
#         )
#
#         # 将当前线程附加到前台线程的输入队列
#         if current_thread_id != foreground_thread_id:
#             user32.AttachThreadInput(current_thread_id, foreground_thread_id, True)
#
#         # 现在可以可靠地设置前台窗口
#         user32.SetForegroundWindow(hwnd)
#         time.sleep(0.3)
#
#         # 分离线程
#         if current_thread_id != foreground_thread_id:
#             user32.AttachThreadInput(current_thread_id, foreground_thread_id, False)
#
#         return True
#
#     def click_at_current_position(self):
#         """
#         在鼠标当前位置点击左键（不移动光标）。
#
#         用途：用户鼠标已在 QQ 输入框上，点击一下让 QQ 获得键盘焦点，
#         后续 keyboard.write() 才能把文字打到 QQ 里。
#         """
#         user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
#         time.sleep(0.02)
#         user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
#         time.sleep(0.1)
#         return True
#
#
# # ============================================================================
# # 四、模块自行测试
# # ============================================================================
# if __name__ == "__main__":
#     print("=== QQ 窗口检测测试 ===\n")
#
#     cap = QQWindowCapture()
#
#     # 1. 列出所有可见窗口（调试用）
#     print("当前可见窗口：")
#     for hwnd, title in _enum_visible_windows():
#         print(f"  [{hwnd}] {title}")
#     print()
#
#     # 2. 查找 QQ 窗口
#     hwnd, title = cap.find_window()
#     if hwnd is None:
#         print("未找到 QQ 窗口！请确保 QQ 已打开，或修改关键词。")
#         exit(1)
#
#     print(f"找到 QQ 窗口: [{hwnd}] {title}")
#
#     # 3. 获取窗口矩形
#     rect = cap.get_window_rect(hwnd)
#     print(f"窗口矩形 (屏幕坐标): {rect}")
#
#     # 4. 获取聊天区域
#     bbox = cap.get_chat_bbox(hwnd)
#     print(f"截图区域 (屏幕坐标): {bbox}")
#
#     # 5. 截取聊天区域并保存
#     img = cap.capture_chat(hwnd)
#     if img:
#         img.save("test_chat_capture.png")
#         print("截图已保存为: test_chat_capture.png")
#     else:
#         print("截图失败！")
#
#     # 6. 测试聚焦
#     print("\n正在测试窗口聚焦（3秒后尝试将 QQ 切换到前台）...")
#     time.sleep(3)
#     cap.focus_window(hwnd)
#     print("聚焦完成！")


"""
固定屏幕区域截图模块
====================
直接截取屏幕上指定的矩形区域，无需定位QQ窗口，适合窗口位置固定的场景。
默认区域针对QQ聊天窗口左下角两行对方消息设置，可自行微调坐标。
"""
"""
固定屏幕区域截图模块
====================
坐标已永久固定，运行即直接截取指定屏幕区域，无需任何手动操作。
固定坐标：(0, 715, 1902, 807) —— 左上角(0,715)，右下角(1902,807)
"""

import ctypes
from ctypes import wintypes
from PIL import ImageGrab

# ============================================================================
# DPI 感知（避免高DPI缩放导致坐标偏移）
# ============================================================================
user32 = ctypes.windll.user32
BOOL = wintypes.BOOL

try:
    user32.SetProcessDPIAware.argtypes = []
    user32.SetProcessDPIAware.restype = BOOL
    _DPI_AWARE_SET = False
except AttributeError:
    _DPI_AWARE_SET = True

mouse_event  = user32.mouse_event
mouse_event.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPARAM]
mouse_event.restype  = None

MOUSEEVENTF_LEFTDOWN  = 0x0002
MOUSEEVENTF_LEFTUP    = 0x0004

import time  # noqa: E402

def _ensure_dpi_aware():
    global _DPI_AWARE_SET
    if not _DPI_AWARE_SET:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        _DPI_AWARE_SET = True

# ============================================================================
# 固定区域截图核心类
# ============================================================================
class FixedRegionCapture:
    def __init__(self):
        _ensure_dpi_aware()
        # 已永久固定的截图坐标，无需修改
        self.capture_bbox = (0, 715, 1902, 807)

    def capture(self):
        """
        截取固定屏幕区域
        返回：PIL.Image 截图对象，失败返回 None
        """
        left, top, right, bottom = self.capture_bbox
        if left >= right or top >= bottom:
            return None
        try:
            return ImageGrab.grab(bbox=self.capture_bbox)
        except Exception:
            return None

    def click_at_current_position(self):
        """
        在鼠标当前位置点击左键（不移动光标），让 QQ 输入框获得键盘焦点。
        """
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.02)
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.1)
        return True

# ============================================================================
# 直接运行：截图并保存
# ============================================================================
if __name__ == "__main__":
    cap = FixedRegionCapture()
    img = cap.capture()
    if img:
        img.save("fixed_capture.png")
        print(f"截图完成，区域：{cap.capture_bbox}，已保存为 fixed_capture.png")
    else:
        print("截图失败，请检查坐标有效性")