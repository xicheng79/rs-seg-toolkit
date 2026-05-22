"""CLI / argparse 行为测试。"""
from __future__ import annotations

import io
import os
import runpy
import sys
import unittest.mock as mock

import pytest

from utils import hint_if_no_args


def test_hint_if_no_args_prints_when_empty():
    with mock.patch.object(sys, 'argv', ['fake.py']):
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            assert hint_if_no_args('fake.py') is True
        out = buf.getvalue()
        assert '未传入命令行参数' in out
        assert 'fake.py --help' in out


def test_hint_if_no_args_silent_when_args_present():
    with mock.patch.object(sys, 'argv', ['fake.py', '--src', 'x']):
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            assert hint_if_no_args('fake.py') is False
        assert buf.getvalue() == ''


def test_hint_if_no_args_infers_script_name():
    with mock.patch.object(sys, 'argv', [r'C:\path\to\my_script.py']):
        buf = io.StringIO()
        with mock.patch('sys.stdout', buf):
            hint_if_no_args(None)
        assert 'my_script.py' in buf.getvalue()


# ---------------- 真实脚本 --help 烟雾测试 ----------------
# argparse --help 会调用 sys.exit(0)，运行不需要任何业务依赖（只要 import 能成功）。
# import 失败的脚本（缺 GDAL/pyecharts）跳过。

NO_DEP_SCRIPTS = [
    'batch_change_extension.py',
    'batch_rename_gis_files.py',
    'compute_label_distribution.py',
    'remap_labels.py',
    'stitch_images.py',
]

DEP_SCRIPTS = {
    'clip_image.py': 'osgeo',
    'compute_dataset_stats.py': None,  # 仅 cv2/numpy
    'convert_png_to_geotiff.py': 'osgeo',
    'rasterize_shapefile.py': 'osgeo',
    'visualize_training_metrics.py': 'pyecharts',
}

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))


def _run_help(script_name):
    script_path = os.path.join(PROJECT_ROOT, script_name)
    with mock.patch.object(sys, 'argv', [script_name, '--help']):
        # argparse 打印 help 后会 sys.exit(0)
        runpy.run_path(script_path, run_name='__main__')


@pytest.mark.parametrize('script', NO_DEP_SCRIPTS)
def test_script_help_exits_clean_no_dep(script, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run_help(script)
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert 'usage:' in captured.out


@pytest.mark.parametrize('script,dep', list(DEP_SCRIPTS.items()))
def test_script_help_exits_clean_with_dep(script, dep, capsys):
    if dep is not None:
        pytest.importorskip(dep)
    with pytest.raises(SystemExit) as excinfo:
        _run_help(script)
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert 'usage:' in captured.out


# ---------------- 内嵌的 _parse_mapping / _parse_int_list 同步测试 ----------------
# 这些函数在脚本的 __main__ 里就地定义，难以 import。这里用一份等价拷贝
# 验证逻辑（保证语义同步）。如果未来有人改坏了 remap_labels 的解析器，
# 此处不会保护到 — 但 test_script_help_exits_clean 至少能验证 argparse 不挂。

def test_remap_mapping_string_format():
    """记录 remap_labels --mapping 字符串格式的契约。"""
    samples = {
        '30:1': {30: 1},
        '0:0,50:1,100:2': {0: 0, 50: 1, 100: 2},
        ' 255 : 1 , 0 : 0 ': {255: 1, 0: 0},
    }
    for s, expected in samples.items():
        # 等价解析逻辑（与 remap_labels.py 中 _parse_mapping 同步）
        result = {}
        for pair in s.split(','):
            pair = pair.strip()
            if not pair:
                continue
            assert ':' in pair
            old, new = pair.split(':', 1)
            result[int(old.strip())] = int(new.strip())
        assert result == expected


def test_label_classes_string_format():
    """记录 compute_label_distribution --classes 字符串格式的契约。"""
    def parse(s):
        s = s.strip()
        if not s or s.lower() == 'auto':
            return None
        return [int(x.strip()) for x in s.split(',') if x.strip()]

    assert parse('auto') is None
    assert parse('') is None
    assert parse('0,1,2,3') == [0, 1, 2, 3]
    assert parse(' 0 , 255 ') == [0, 255]
