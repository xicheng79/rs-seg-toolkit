"""batch_rename_gis_files 测试。"""
from __future__ import annotations

import os
import pytest

from batch_rename_gis_files import (
    parse_mapping_file,
    find_related_files,
    batch_rename_gis_files,
    SHP_SIDECAR_EXTS,
    TIF_SIDECAR_EXTS,
)


def test_parse_mapping_tab_separated(tmp_path):
    p = tmp_path / "map.txt"
    p.write_text("a\tb\nc\td\n", encoding='utf-8')
    m = parse_mapping_file(str(p))
    assert m == {'a': 'b', 'c': 'd'}


def test_parse_mapping_comma_separated(tmp_path):
    p = tmp_path / "map.txt"
    p.write_text("a,b\nc,d\n", encoding='utf-8')
    m = parse_mapping_file(str(p))
    assert m == {'a': 'b', 'c': 'd'}


def test_parse_mapping_space_separated(tmp_path):
    p = tmp_path / "map.txt"
    p.write_text("a b\nc d\n", encoding='utf-8')
    m = parse_mapping_file(str(p))
    assert m == {'a': 'b', 'c': 'd'}


def test_parse_mapping_skips_blank_and_comments(tmp_path):
    p = tmp_path / "map.txt"
    p.write_text("# comment\n\na\tb\n  \nc\td\n", encoding='utf-8')
    m = parse_mapping_file(str(p))
    assert m == {'a': 'b', 'c': 'd'}


def test_find_related_files_shp_sidecars(tmp_path):
    """正确收集 sidecar 白名单内的文件，不收无关文件。"""
    for ext in ['.shp', '.shx', '.dbf', '.prj']:
        (tmp_path / f"tile_001{ext}").write_text('x', encoding='utf-8')
    # 干扰文件：同前缀但不在白名单
    (tmp_path / "tile_001.exe").write_text('x', encoding='utf-8')
    (tmp_path / "tile_001_thumbnail.jpg").write_text('x', encoding='utf-8')
    # 干扰文件：不同前缀
    (tmp_path / "tile_002.shp").write_text('x', encoding='utf-8')

    files = find_related_files(str(tmp_path), 'tile_001', SHP_SIDECAR_EXTS)
    found_basenames = sorted(os.path.basename(f) for f in files)
    assert found_basenames == sorted([
        'tile_001.shp', 'tile_001.shx',
        'tile_001.dbf', 'tile_001.prj',
    ])


def test_dry_run_does_not_modify_filesystem(tmp_path, capsys):
    """dry-run 模式仅打印计划，不真实改名。"""
    for ext in ['.tif', '.tfw']:
        (tmp_path / f"old{ext}").write_text('x', encoding='utf-8')

    mapping = tmp_path / "map.txt"
    mapping.write_text("old\tnew\n", encoding='utf-8')

    batch_rename_gis_files(
        mapping_path=str(mapping),
        target_folder=str(tmp_path),
        kinds=('tif',),
        dry_run=True,
    )

    # 文件未被改名
    assert (tmp_path / "old.tif").exists()
    assert not (tmp_path / "new.tif").exists()


def test_apply_renames_files_with_sidecars(tmp_path):
    """实际重命名 GeoTIFF + 全部 sidecar。"""
    for ext in ['.tif', '.tfw']:
        (tmp_path / f"old{ext}").write_text(f"data_{ext}", encoding='utf-8')

    mapping = tmp_path / "map.txt"
    mapping.write_text("old\tnew\n", encoding='utf-8')

    batch_rename_gis_files(
        mapping_path=str(mapping),
        target_folder=str(tmp_path),
        kinds=('tif',),
        dry_run=False,
    )

    assert not (tmp_path / "old.tif").exists()
    assert not (tmp_path / "old.tfw").exists()
    assert (tmp_path / "new.tif").read_text(encoding='utf-8') == "data_.tif"
    assert (tmp_path / "new.tfw").read_text(encoding='utf-8') == "data_.tfw"


def test_swap_a_and_b_no_overwrite(tmp_path):
    """两阶段重命名核心：a->b 且 b->a 时不发生覆盖丢失。"""
    (tmp_path / "a.tif").write_text("data_A", encoding='utf-8')
    (tmp_path / "b.tif").write_text("data_B", encoding='utf-8')

    mapping = tmp_path / "map.txt"
    mapping.write_text("a\tb\nb\ta\n", encoding='utf-8')

    batch_rename_gis_files(
        mapping_path=str(mapping),
        target_folder=str(tmp_path),
        kinds=('tif',),
        dry_run=False,
    )

    # 内容互换：a.tif 现在应为原 B 内容；b.tif 应为原 A 内容
    assert (tmp_path / "a.tif").read_text(encoding='utf-8') == "data_B"
    assert (tmp_path / "b.tif").read_text(encoding='utf-8') == "data_A"


def test_does_not_touch_unwhitelisted_files(tmp_path):
    """白名单外的文件即使前缀相同也不应被改名。"""
    (tmp_path / "old.tif").write_text("real", encoding='utf-8')
    (tmp_path / "old.exe").write_text("malware", encoding='utf-8')

    mapping = tmp_path / "map.txt"
    mapping.write_text("old\tnew\n", encoding='utf-8')

    batch_rename_gis_files(
        mapping_path=str(mapping),
        target_folder=str(tmp_path),
        kinds=('tif',),
        dry_run=False,
    )

    assert (tmp_path / "new.tif").exists()
    assert (tmp_path / "old.exe").exists(), "白名单外文件不应被动到"
    assert not (tmp_path / "new.exe").exists()
