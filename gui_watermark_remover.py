#!/usr/bin/env python3
"""
Gemini Watermark Remover - Web Version
Chạy web server local để upload ảnh và xóa watermark trực tiếp trên trình duyệt.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from flask import Flask, flash, redirect, render_template_string, request, send_file, url_for
from werkzeug.utils import secure_filename

from gemini_watermark_remover import remove_watermark

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
WEBP_QUALITY = 85

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "gemini-watermark-dev-key")

HTML_TEMPLATE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemini Watermark Remover</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f3f6fb; color: #222; }
    .container { max-width: 760px; margin: 32px auto; background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    h1 { margin: 0 0 16px; font-size: 26px; color: #1a73e8; }
    p.note { margin-top: 0; color: #5f6368; }
    .field { margin-bottom: 14px; }
    label { font-weight: 600; display: block; margin-bottom: 6px; }
    input[type=file], input[type=text], input[type=number] { width: 100%; padding: 10px; border: 1px solid #d0d7de; border-radius: 8px; }
    .row { display: flex; gap: 14px; flex-wrap: wrap; }
    .row > .field { flex: 1 1 240px; }
    .checkbox { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
    button { background: #34a853; color: #fff; border: 0; border-radius: 8px; padding: 11px 16px; font-size: 15px; font-weight: 700; cursor: pointer; }
    button:hover { filter: brightness(0.95); }
    .flash-wrap { margin-bottom: 16px; }
    .flash { padding: 10px; border-radius: 8px; margin-bottom: 8px; }
    .flash.error { background: #fde8e8; color: #b42318; }
    .flash.info { background: #e8f1ff; color: #1d4ed8; }
    .support { margin-top: 14px; font-size: 13px; color: #5f6368; }
  </style>
</head>
<body>
  <div class="container">
    <h1>✦ Gemini Watermark Remover</h1>
    <p class="note">Upload ảnh và tải về ảnh đã xóa watermark ngay.</p>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
      <div class="flash-wrap">
        {% for category, message in messages %}
          <div class="flash {{ category }}">{{ message }}</div>
        {% endfor %}
      </div>
      {% endif %}
    {% endwith %}

    <form method="post" action="{{ url_for('process_images') }}" enctype="multipart/form-data">
      <div class="field">
        <label for="image">Ảnh đầu vào</label>
        <input id="image" name="image" type="file" required>
      </div>

      <div class="row">
        <div class="field">
          <label for="suffix">Hậu tố tên file</label>
          <input id="suffix" name="suffix" type="text" value="_clean" maxlength="50">
        </div>
        <div class="field">
          <label for="alpha_scale">Alpha scale (0.5 → 3.0)</label>
          <input id="alpha_scale" name="alpha_scale" type="number" min="0.5" max="3.0" step="0.05" value="1.0" required>
        </div>
      </div>

      <div class="field">
        <label for="output_mode">Định dạng đầu ra</label>
        <select id="output_mode" name="output_mode" style="width: 100%; padding: 10px; border: 1px solid #d0d7de; border-radius: 8px;">
          <option value="original" selected>Giữ nguyên định dạng/kích thước (mặc định)</option>
          <option value="webp">Xuất ảnh sang WebP</option>
        </select>
      </div>

      <button type="submit">▶ Bắt đầu xử lý</button>
    </form>

    <div class="support">
      Định dạng hỗ trợ: {{ supported }}
    </div>
  </div>
</body>
</html>
"""


def _is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


@app.get("/")
def home():
    return render_template_string(
        HTML_TEMPLATE,
        supported=", ".join(sorted(SUPPORTED_EXTENSIONS)),
    )


@app.post("/process")
def process_images():
    uploaded_file = request.files.get("image")
    if not uploaded_file or not uploaded_file.filename:
        flash("Vui lòng chọn 1 ảnh.", "error")
        return redirect(url_for("home"))

    original_name = secure_filename(uploaded_file.filename)
    if not original_name or not _is_supported(original_name):
        flash("Định dạng ảnh không được hỗ trợ.", "error")
        return redirect(url_for("home"))

    suffix = request.form.get("suffix", "_clean").strip() or "_clean"
    output_mode = (request.form.get("output_mode") or "original").strip().lower()
    convert_webp = output_mode == "webp"

    try:
        alpha_scale = float(request.form.get("alpha_scale", "1.0"))
    except ValueError:
        flash("Alpha scale không hợp lệ.", "error")
        return redirect(url_for("home"))

    alpha_scale = max(0.5, min(3.0, alpha_scale))

    try:
        cleaned = remove_watermark(uploaded_file.read(), alpha_scale=alpha_scale)
    except Exception as exc:
        flash(f"Lỗi xử lý ảnh: {exc}", "error")
        return redirect(url_for("home"))

    src_path = Path(original_name)
    output_ext = ".webp" if convert_webp else src_path.suffix.lower()
    output_name = f"{src_path.stem}{suffix}{output_ext}"

    image_buffer = BytesIO()
    if convert_webp:
        cleaned.save(image_buffer, format="WEBP", quality=WEBP_QUALITY, method=6)
    else:
        format_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".bmp": "BMP",
            ".tiff": "TIFF",
            ".webp": "WEBP",
        }
        save_format = format_map.get(output_ext, "PNG")
        cleaned.save(image_buffer, format=save_format)

    image_buffer.seek(0)
    return send_file(
        image_buffer,
        as_attachment=True,
        download_name=output_name,
        mimetype="image/png" if output_ext == ".png" else "image/jpeg" if output_ext in (".jpg", ".jpeg") else "image/webp" if output_ext == ".webp" else "image/bmp",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
