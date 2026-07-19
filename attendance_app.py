"""Desktop application for real-time face-recognition attendance."""

from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Set

import cv2
import numpy as np
from PIL import Image, ImageTk

from few_shot_face_classification.attendance import (
    ABSENT,
    PRESENT,
    AttendanceSession,
    RollingConfirmation,
    RosterError,
    filter_labeled_embeddings,
    load_roster,
    parse_label_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时人脸识别签到")
    parser.add_argument("--roster", type=Path, default=Path("data/roster.csv"), help="名单 CSV")
    parser.add_argument("--labeled", type=Path, default=Path("data/labeled"), help="标注照片目录")
    parser.add_argument(
        "--attendance-dir", type=Path, default=Path("data/attendance"), help="签到记录目录"
    )
    parser.add_argument("--cache", type=Path, default=Path("data/embeddings_cache.pkl"))
    parser.add_argument("--no-cache", action="store_true", help="不使用 embedding cache")
    parser.add_argument("--threshold", type=float, default=1.0, help="人脸距离阈值")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 摄像头编号")
    parser.add_argument("--width", type=int, default=0, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=0, help="摄像头高度")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="首次构建缓存的 CPU worker 数；桌面程序默认 1 以避免阻塞",
    )
    parser.add_argument("--confirm-hits", type=int, default=3)
    parser.add_argument("--confirm-window", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.confirm_hits <= args.confirm_window:
        parser.error("必须满足 1 <= --confirm-hits <= --confirm-window")
    return args


class RecognitionWorker:
    """Load models and continuously emit the newest annotated inference frame."""

    def __init__(self, args: argparse.Namespace, roster: List[str]) -> None:
        self.args = args
        self.roster = roster
        self.stop_event = threading.Event()
        self.frames: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="recognition-worker", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _put_latest_frame(self, item: Dict[str, Any]) -> None:
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(item)

    def _run(self) -> None:
        cap = None
        try:
            from few_shot_face_classification.cache import load_or_build_embeddings_cache
            from few_shot_face_classification.embed import embed_with_boxes, get_networks
            from few_shot_face_classification.similarity import _draw_faces_on_image, get_classes

            self.events.put({"type": "status", "text": "正在加载标注数据和模型…"})
            labeled_paths, labeled_embs = load_or_build_embeddings_cache(
                labeled_folder=self.args.labeled,
                cache_file=self.args.cache,
                batch_size=self.args.batch_size,
                use_cache=not self.args.no_cache,
                device=self.args.device,
                num_workers=self.args.num_workers,
            )
            labeled_paths, labeled_embs, warnings = filter_labeled_embeddings(
                labeled_paths, labeled_embs, self.roster
            )
            available_names = {
                label
                for path in labeled_paths
                for label in [parse_label_name(path)]
                if label is not None and label.lower() != "none"
            }
            for name in self.roster:
                if name not in available_names:
                    warnings.append(f"名单中的 {name} 没有有效标注照片")
            if warnings:
                self.events.put({"type": "warnings", "items": warnings})
            if not available_names:
                raise RuntimeError("没有可用于签到的有效标注照片，请使用“姓名_序号.jpg”命名")

            mtcnn, vggface2 = get_networks(device=self.args.device)
            cap = cv2.VideoCapture(self.args.camera)
            if self.args.width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            if self.args.height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.args.camera}")
            self.events.put({"type": "status", "text": "识别运行中"})

            roster_set = set(self.roster)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("摄像头读取失败或已断开")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                embeddings, boxes = embed_with_boxes(
                    image, mtcnn=mtcnn, vggface2=vggface2, device=self.args.device
                )
                classes = get_classes(
                    embeddings, labeled_paths, labeled_embs, thr=self.args.threshold
                )
                display_names = [name if name else "Unknown" for name in classes]
                detected: Set[str] = {name for name in classes if name in roster_set}
                if boxes:
                    image = _draw_faces_on_image(image, np.asarray(boxes), display_names)
                self._put_latest_frame({
                    "rgb": np.asarray(image),
                    "detected": detected,
                })
        except Exception as exc:
            if not self.stop_event.is_set():
                self.events.put({"type": "error", "text": str(exc)})
        finally:
            if cap is not None:
                cap.release()


class AttendanceApp:
    def __init__(
        self,
        root: Any,
        args: argparse.Namespace,
        roster: List[str],
        session: AttendanceSession,
    ) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = root
        self.args = args
        self.roster = roster
        self.session = session
        self.confirmation = RollingConfirmation(args.confirm_hits, args.confirm_window)
        self.worker = RecognitionWorker(args, roster)
        self.photo = None
        self.closing = False

        root.title("实时人脸识别签到")
        root.geometry("1200x720")
        root.minsize(900, 560)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._refresh_roster()
        self.worker.start()
        root.after(50, self._poll_worker)

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2, minsize=370)
        container.rowconfigure(0, weight=1)

        video_panel = ttk.Frame(container)
        video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)
        self.video_label = ttk.Label(video_panel, text="正在启动摄像头…", anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.status_var = tk.StringVar(value="正在初始化…")
        ttk.Label(video_panel, textvariable=self.status_var, anchor="w").grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )

        roster_panel = ttk.Frame(container)
        roster_panel.grid(row=0, column=1, sticky="nsew")
        roster_panel.columnconfigure(0, weight=1)
        roster_panel.rowconfigure(3, weight=1)
        ttk.Label(roster_panel, text="签到名单", font=("TkDefaultFont", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.summary_var = tk.StringVar()
        ttk.Label(roster_panel, textvariable=self.summary_var, font=("TkDefaultFont", 12)).grid(
            row=1, column=0, sticky="w", pady=(4, 8)
        )

        search_row = ttk.Frame(roster_panel)
        search_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        search_row.columnconfigure(1, weight=1)
        ttk.Label(search_row, text="搜索：").grid(row=0, column=0)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_roster())
        ttk.Entry(search_row, textvariable=self.search_var).grid(row=0, column=1, sticky="ew")

        table_frame = ttk.Frame(roster_panel)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_frame,
            columns=("name", "status", "time"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("name", text="姓名")
        self.tree.heading("status", text="状态")
        self.tree.heading("time", text="签到时间")
        self.tree.column("name", width=135, anchor="center")
        self.tree.column("status", width=75, anchor="center")
        self.tree.column("time", width=90, anchor="center")
        self.tree.tag_configure("present", background="#d8f3dc", foreground="#146c2e")
        self.tree.tag_configure("absent", background="#ffe0e0", foreground="#a11a1a")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<Double-1>", lambda _event: self._toggle_selected())

        actions = ttk.Frame(roster_panel)
        actions.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Button(actions, text="切换选中状态", command=self._toggle_selected).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(actions, text="新建签到", command=self._new_session).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        ttk.Label(roster_panel, text="双击学生也可人工补签或撤销", foreground="#666666").grid(
            row=5, column=0, sticky="w", pady=(7, 0)
        )

    def _refresh_roster(self) -> None:
        selected = self._selected_name()
        self.tree.delete(*self.tree.get_children())
        query = self.search_var.get().strip().lower()
        for name in self.roster:
            if query and query not in name.lower():
                continue
            record = self.session.records[name]
            is_present = record.status == PRESENT
            display_time = ""
            if record.checkin_time and "T" in record.checkin_time:
                display_time = record.checkin_time.split("T", 1)[1][:8]
            item = self.tree.insert(
                "",
                "end",
                values=(name, "已到" if is_present else "未到", display_time),
                tags=("present" if is_present else "absent",),
            )
            if name == selected:
                self.tree.selection_set(item)
        arrived = len(self.session.present_names())
        self.summary_var.set(f"已到 {arrived} / 应到 {len(self.roster)} · 未到 {len(self.roster) - arrived}")

    def _selected_name(self) -> str:
        selected = self.tree.selection() if hasattr(self, "tree") else ()
        if not selected:
            return ""
        values = self.tree.item(selected[0], "values")
        return str(values[0]) if values else ""

    def _toggle_selected(self) -> None:
        name = self._selected_name()
        if not name:
            self.messagebox.showinfo("人工纠错", "请先选择一位同学")
            return
        if self.session.records[name].status == PRESENT:
            confirmed = self.messagebox.askyesno("确认撤销", f"确定将 {name} 改为未到吗？")
            if not confirmed:
                return
        self.session.toggle_manual(name)
        self.confirmation.reset()
        self._refresh_roster()

    def _new_session(self) -> None:
        if not self.messagebox.askyesno("新建签到", "旧记录将先归档，然后清空当前签到。是否继续？"):
            return
        destination = self.session.reset_and_archive()
        self.confirmation.reset()
        self._refresh_roster()
        self.messagebox.showinfo("新建签到", f"旧记录已归档到：\n{destination}")

    def _poll_worker(self) -> None:
        if self.closing:
            return
        while True:
            try:
                event = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if event["type"] == "status":
                self.status_var.set(event["text"])
            elif event["type"] == "warnings":
                items = event["items"]
                preview = "\n".join(items[:12])
                if len(items) > 12:
                    preview += f"\n…另有 {len(items) - 12} 条"
                self.messagebox.showwarning("标注数据提示", preview)
            elif event["type"] == "error":
                self.status_var.set(f"错误：{event['text']}")
                self.messagebox.showerror("签到程序错误", event["text"])

        latest = None
        while True:
            try:
                latest = self.worker.frames.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._show_frame(latest["rgb"])
            eligible = set(self.roster) - self.session.present_names()
            confirmed = self.confirmation.update(latest["detected"], eligible)
            changed = False
            for name in confirmed:
                changed |= self.session.set_present(name, True, "automatic")
            if changed:
                self._refresh_roster()
        self.root.after(50, self._poll_worker)

    def _show_frame(self, rgb: np.ndarray) -> None:
        image = Image.fromarray(rgb)
        max_width = max(320, self.video_label.winfo_width())
        max_height = max(240, self.video_label.winfo_height())
        scale = min(max_width / image.width, max_height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(size, resampling)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo, text="")

    def close(self) -> None:
        self.closing = True
        self.worker.stop()
        self.root.destroy()


def main() -> None:
    import tkinter as tk
    from tkinter import messagebox

    args = parse_args()
    try:
        roster = load_roster(args.roster)
        session = AttendanceSession(roster, args.attendance_dir)
    except (RosterError, OSError, ValueError) as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("无法启动签到", str(exc))
        root.destroy()
        raise SystemExit(2) from exc

    root = tk.Tk()
    AttendanceApp(root, args, roster, session)
    root.mainloop()


if __name__ == "__main__":
    main()
