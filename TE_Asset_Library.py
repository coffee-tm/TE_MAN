import io
import os
import re
import shutil
import time
from pathlib import Path

import folder_paths
import numpy as np
import torch
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence

try:
    from server import PromptServer
except Exception:  # pragma: no cover
    PromptServer = None


ASSET_ROOT_NAME = "TE_MAN"
DEFAULT_PROJECT = "默认项目"
DEFAULT_CATEGORIES = ("人物", "产品", "场景", "背景")
PROTECTED_CATEGORIES = {"", "__all__", "全部", "未分类"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def _json_error(message, status=400):
    return web.json_response({"success": False, "error": str(message)}, status=status)


def _input_dir():
    return Path(folder_paths.get_input_directory()).resolve()


def _asset_root():
    root = _input_dir() / ASSET_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _is_inside(path, root):
    try:
        return os.path.commonpath([str(Path(path).resolve()), str(Path(root).resolve())]) == str(Path(root).resolve())
    except Exception:
        return False


def _clean_segment(value, fallback):
    text = str(value or "").strip()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r'[<>:"|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text or text in {".", ".."}:
        return fallback
    return text[:80]


def _clean_filename_stem(value, fallback):
    text = _clean_segment(value, fallback)
    return Path(text).stem or fallback


def _ensure_default_structure():
    root = _asset_root()
    has_project = any(child.is_dir() for child in root.iterdir())
    if has_project:
        return

    project_dir = root / DEFAULT_PROJECT
    for category in DEFAULT_CATEGORIES:
        (project_dir / category).mkdir(parents=True, exist_ok=True)


def _project_dir(project, create=False):
    root = _asset_root()
    safe_project = _clean_segment(project, DEFAULT_PROJECT)
    path = (root / safe_project).resolve()
    if not _is_inside(path, root):
        raise ValueError("非法项目路径")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path, safe_project


def _category_dir(project, category, create=False):
    project_path, safe_project = _project_dir(project, create=create)
    safe_category = _clean_segment(category, DEFAULT_CATEGORIES[0])
    path = (project_path / safe_category).resolve()
    if not _is_inside(path, _asset_root()):
        raise ValueError("非法分类路径")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path, safe_project, safe_category


def _normalize_asset_relative_path(relative_path):
    text = str(relative_path or "").replace("\\", "/").lstrip("/")
    prefix = f"{ASSET_ROOT_NAME}/"
    if text == ASSET_ROOT_NAME:
        text = ""
    elif text.startswith(prefix):
        text = text[len(prefix):]
    parts = [part for part in text.split("/") if part and part not in {".", ".."}]
    if len(parts) < 3:
        raise ValueError("素材路径需要是 项目/分类/文件名")
    return parts


def _resolve_asset_path(relative_path):
    root = _asset_root()
    parts = _normalize_asset_relative_path(relative_path)
    path = root.joinpath(*parts).resolve()
    if not _is_inside(path, root):
        raise ValueError("非法素材路径")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("不支持的图片格式")
    return path


def _input_relative_path(path):
    rel = Path(path).resolve().relative_to(_input_dir())
    return rel.as_posix()


def _view_image_info(path):
    input_relative = _input_relative_path(path)
    subfolder = str(Path(input_relative).parent).replace("\\", "/")
    if subfolder == ".":
        subfolder = ""
    return {
        "filename": Path(input_relative).name,
        "subfolder": subfolder,
        "type": "input",
    }


def _is_image_path(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS and Path(path).is_file()


def _iter_image_files(directory):
    if not Path(directory).is_dir():
        return
    for child in sorted(Path(directory).iterdir(), key=lambda item: item.name.lower()):
        if _is_image_path(child):
            yield child


def _image_dimensions(path):
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 0, 0


def _asset_record(path, project, category):
    stat = path.stat()
    width, height = _image_dimensions(path)
    input_relative = _input_relative_path(path)
    return {
        "name": path.name,
        "project": project,
        "category": category,
        "relative_path": input_relative,
        "subfolder": str(Path(input_relative).parent).replace("\\", "/"),
        "type": "input",
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "width": width,
        "height": height,
    }


def _project_records():
    _ensure_default_structure()
    root = _asset_root()
    projects = []

    for project_path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
        categories = []
        for category_path in sorted((item for item in project_path.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
            count = sum(1 for _ in _iter_image_files(category_path))
            categories.append({"name": category_path.name, "count": count})

        loose_count = sum(1 for _ in _iter_image_files(project_path))
        if loose_count:
            categories.insert(0, {"name": "未分类", "count": loose_count})

        projects.append({"name": project_path.name, "categories": categories})

    if not projects:
        _ensure_default_structure()
        return _project_records()

    return projects


def _list_assets(project, category="", query=""):
    project_path, safe_project = _project_dir(project or DEFAULT_PROJECT, create=True)
    normalized_category = str(category or "").strip()
    normalized_query = str(query or "").strip().lower()
    assets = []

    if normalized_category and normalized_category not in {"__all__", "全部"}:
        if normalized_category == "未分类":
            category_paths = [(project_path, "未分类")]
        else:
            category_path, _safe_project, safe_category = _category_dir(safe_project, normalized_category, create=False)
            category_paths = [(category_path, safe_category)]
    else:
        category_paths = []
        loose_images = list(_iter_image_files(project_path))
        if loose_images:
            category_paths.append((project_path, "未分类"))
        for child in sorted((item for item in project_path.iterdir() if item.is_dir()), key=lambda item: item.name.lower()):
            category_paths.append((child, child.name))

    for category_path, category_name in category_paths:
        for image_path in _iter_image_files(category_path):
            if normalized_query and normalized_query not in image_path.name.lower():
                continue
            assets.append(_asset_record(image_path, safe_project, category_name))

    assets.sort(key=lambda item: (item["mtime"], item["name"].lower()), reverse=True)
    return assets


def _resolve_source_image(payload):
    upload_image = str(payload.get("upload_image") or "").strip()
    image_info = payload.get("image_info") or {}

    if upload_image:
        source_path = Path(folder_paths.get_annotated_filepath(upload_image)).resolve()
        base_dir = _input_dir()
    elif isinstance(image_info, dict) and image_info.get("filename"):
        image_type = str(image_info.get("type") or "input")
        base = folder_paths.get_directory_by_type(image_type)
        if not base:
            raise ValueError(f"不支持的图片目录类型: {image_type}")

        base_dir = Path(base).resolve()
        subfolder = str(image_info.get("subfolder") or "").strip()
        filename = os.path.basename(str(image_info.get("filename") or ""))
        source_path = (base_dir / subfolder / filename).resolve()
    else:
        raise ValueError("没有找到可添加的图片")

    if not _is_inside(source_path, base_dir):
        raise ValueError("非法图片路径")
    if not source_path.is_file():
        raise FileNotFoundError(f"图片不存在: {source_path.name}")
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("只支持添加图片文件")
    return source_path


def _dedupe_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _copy_to_library(source_path, project, category, filename_stem=""):
    target_dir, safe_project, safe_category = _category_dir(project, category, create=True)
    stem = _clean_filename_stem(filename_stem, Path(source_path).stem)
    suffix = Path(source_path).suffix.lower()
    target_path = _dedupe_path(target_dir / f"{stem}{suffix}")
    shutil.copy2(source_path, target_path)
    return _asset_record(target_path, safe_project, safe_category)


def _infer_project_category(path):
    rel = _input_relative_path(path)
    parts = _normalize_asset_relative_path(rel)
    project = parts[0] if len(parts) > 0 else DEFAULT_PROJECT
    category = parts[1] if len(parts) > 2 else "未分类"
    return project, category


def _rename_asset(relative_path, new_name):
    source_path = _resolve_asset_path(relative_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"素材不存在: {relative_path}")

    raw_name = str(new_name or "").strip().replace("\\", "/")
    raw_name = os.path.basename(raw_name)
    if not raw_name:
        raise ValueError("新文件名不能为空")

    requested_suffix = Path(raw_name).suffix.lower()
    suffix = requested_suffix or source_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("不支持的图片格式")

    stem_source = raw_name[:-len(requested_suffix)] if requested_suffix else raw_name
    stem = _clean_filename_stem(stem_source, source_path.stem)
    target_path = (source_path.parent / f"{stem}{suffix}").resolve()
    if not _is_inside(target_path, _asset_root()):
        raise ValueError("非法目标路径")

    if target_path == source_path:
        project, category = _infer_project_category(source_path)
        return _asset_record(source_path, project, category)
    if target_path.exists():
        raise FileExistsError(f"文件已存在: {target_path.name}")

    source_path.rename(target_path)
    project, category = _infer_project_category(target_path)
    return _asset_record(target_path, project, category)


def _delete_asset(relative_path):
    path = _resolve_asset_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"素材不存在: {relative_path}")
    project, category = _infer_project_category(path)
    record = _asset_record(path, project, category)
    path.unlink()
    return record


def _delete_category(project, category):
    raw_category = str(category or "").strip()
    if raw_category in PROTECTED_CATEGORIES:
        raise ValueError("请选择一个可删除的分类")

    category_path, safe_project, safe_category = _category_dir(project or DEFAULT_PROJECT, raw_category, create=False)
    if safe_category in PROTECTED_CATEGORIES:
        raise ValueError("请选择一个可删除的分类")
    if not category_path.exists():
        raise FileNotFoundError(f"分类不存在: {safe_category}")
    if category_path.is_symlink() or not category_path.is_dir():
        raise ValueError("只能删除素材库里的分类文件夹")

    project_path, _safe_project = _project_dir(safe_project, create=False)
    if category_path == project_path or not _is_inside(category_path, project_path):
        raise ValueError("非法分类路径")

    shutil.rmtree(category_path)
    return {"project": safe_project, "category": safe_category}


def _load_image_tensor(path):
    image = Image.open(path)
    output_images = []
    width = None
    height = None

    for frame in ImageSequence.Iterator(image):
        frame = ImageOps.exif_transpose(frame)
        if frame.mode == "I":
            frame = frame.point(lambda value: value * (1 / 255))
        rgb = frame.convert("RGB")

        if width is None:
            width, height = rgb.size
        elif rgb.size != (width, height):
            continue

        array = np.array(rgb).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(array).unsqueeze(0))

    if not output_images:
        raise ValueError("图片无法读取")
    return torch.cat(output_images, dim=0)


def _all_asset_choices():
    _ensure_default_structure()
    root = _asset_root()
    choices = []
    for path in root.rglob("*"):
        if _is_image_path(path):
            choices.append(_input_relative_path(path))
    return [""] + sorted(set(choices))


class TEAssetLibrary:
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "asset_path")
    FUNCTION = "load_asset"
    CATEGORY = "TE MAN/Utils"
    DESCRIPTION = "从 input/TE_MAN 项目素材库读取图片。主要配合右侧 TE MAN 资产素材库面板使用。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "asset": (
                    _all_asset_choices(),
                    {
                        "default": "",
                        "tooltip": "素材路径，例如 TE_MAN/项目/人物/a.png。也可以从 TE MAN 资产素材库面板拖拽生成加载图像节点。",
                    },
                ),
            },
        }

    def load_asset(self, asset=""):
        if not asset:
            raise ValueError("请先选择一个素材。")
        path = _resolve_asset_path(asset)
        image = _load_image_tensor(path)
        image_info = _view_image_info(path)
        return {
            "ui": {"images": [image_info]},
            "result": (image, _input_relative_path(path)),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, asset):
        if not asset:
            return True
        try:
            path = _resolve_asset_path(asset)
        except Exception as exc:
            return str(exc)
        if not path.is_file():
            return f"素材不存在: {asset}"
        return True


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/te_asset_library/list")
    async def te_asset_library_list(request):
        try:
            projects = _project_records()
            requested_project = request.rel_url.query.get("project") or projects[0]["name"]
            project_names = {project["name"] for project in projects}
            if requested_project not in project_names:
                requested_project = projects[0]["name"]
            category = request.rel_url.query.get("category") or "__all__"
            query = request.rel_url.query.get("q") or ""
            assets = _list_assets(requested_project, category, query)
            return web.json_response(
                {
                    "success": True,
                    "root": f"input/{ASSET_ROOT_NAME}",
                    "current_project": requested_project,
                    "current_category": category,
                    "projects": projects,
                    "assets": assets,
                    "total": len(assets),
                }
            )
        except Exception as exc:
            return _json_error(exc, status=500)

    async def _handle_te_asset_library_project(request):
        try:
            payload = request.rel_url.query if request.method == "GET" else await request.json()
            project_path, project = _project_dir(payload.get("project"), create=True)
            if not any(project_path.iterdir()):
                for category in DEFAULT_CATEGORIES:
                    (project_path / category).mkdir(parents=True, exist_ok=True)
            return web.json_response({"success": True, "project": project})
        except Exception as exc:
            return _json_error(exc)

    PromptServer.instance.routes.post("/te_asset_library/project")(_handle_te_asset_library_project)
    PromptServer.instance.routes.get("/te_asset_library/project")(_handle_te_asset_library_project)

    async def _handle_te_asset_library_category(request):
        try:
            payload = request.rel_url.query if request.method == "GET" else await request.json()
            _path, project, category = _category_dir(payload.get("project"), payload.get("category"), create=True)
            return web.json_response({"success": True, "project": project, "category": category})
        except Exception as exc:
            return _json_error(exc)

    PromptServer.instance.routes.post("/te_asset_library/category")(_handle_te_asset_library_category)
    PromptServer.instance.routes.get("/te_asset_library/category")(_handle_te_asset_library_category)

    async def _handle_te_asset_library_delete_category(request):
        try:
            payload = request.rel_url.query if request.method == "GET" else await request.json()
            deleted = _delete_category(payload.get("project") or DEFAULT_PROJECT, payload.get("category") or "")
            return web.json_response({"success": True, **deleted})
        except FileNotFoundError as exc:
            return _json_error(exc, status=404)
        except Exception as exc:
            return _json_error(exc)

    PromptServer.instance.routes.post("/te_asset_library/category/delete")(_handle_te_asset_library_delete_category)
    PromptServer.instance.routes.get("/te_asset_library/category/delete")(_handle_te_asset_library_delete_category)

    @PromptServer.instance.routes.post("/te_asset_library/add")
    async def te_asset_library_add(request):
        try:
            payload = await request.json()
            source_path = _resolve_source_image(payload)
            asset = _copy_to_library(
                source_path=source_path,
                project=payload.get("project") or DEFAULT_PROJECT,
                category=payload.get("category") or DEFAULT_CATEGORIES[0],
                filename_stem=payload.get("filename_stem") or "",
            )
            return web.json_response({"success": True, "asset": asset})
        except FileNotFoundError as exc:
            return _json_error(exc, status=404)
        except Exception as exc:
            return _json_error(exc)

    async def _handle_te_asset_library_rename(request):
        try:
            payload = request.rel_url.query if request.method == "GET" else await request.json()
            asset = _rename_asset(
                relative_path=payload.get("relative_path") or payload.get("path") or "",
                new_name=payload.get("new_name") or "",
            )
            return web.json_response({"success": True, "asset": asset})
        except FileNotFoundError as exc:
            return _json_error(exc, status=404)
        except FileExistsError as exc:
            return _json_error(exc, status=409)
        except Exception as exc:
            return _json_error(exc)

    PromptServer.instance.routes.post("/te_asset_library/rename")(_handle_te_asset_library_rename)
    PromptServer.instance.routes.get("/te_asset_library/rename")(_handle_te_asset_library_rename)

    async def _handle_te_asset_library_delete(request):
        try:
            payload = request.rel_url.query if request.method == "GET" else await request.json()
            asset = _delete_asset(payload.get("relative_path") or payload.get("path") or "")
            return web.json_response({"success": True, "asset": asset})
        except FileNotFoundError as exc:
            return _json_error(exc, status=404)
        except Exception as exc:
            return _json_error(exc)

    PromptServer.instance.routes.post("/te_asset_library/delete")(_handle_te_asset_library_delete)
    PromptServer.instance.routes.get("/te_asset_library/delete")(_handle_te_asset_library_delete)

    @PromptServer.instance.routes.get("/te_asset_library/thumbnail")
    async def te_asset_library_thumbnail(request):
        try:
            relative_path = request.rel_url.query.get("path") or ""
            size = int(request.rel_url.query.get("size") or 240)
            size = max(80, min(size, 640))
            path = _resolve_asset_path(relative_path)
            if not path.is_file():
                return web.Response(status=404)

            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
                image.thumbnail((size, size), resampling)
                buffer = io.BytesIO()
                image.save(buffer, format="WEBP", quality=82, method=4)
                buffer.seek(0)

            headers = {
                "Cache-Control": "no-cache",
                "X-TE-Asset-MTime": str(int(path.stat().st_mtime)),
            }
            return web.Response(body=buffer.read(), content_type="image/webp", headers=headers)
        except FileNotFoundError:
            return web.Response(status=404)
        except Exception as exc:
            return _json_error(exc, status=400)


NODE_CLASS_MAPPINGS = {
    "TE_image_pro_asset_library": TEAssetLibrary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TE_image_pro_asset_library": "TE MAN 资产素材库",
}
