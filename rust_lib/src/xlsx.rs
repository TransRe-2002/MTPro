//! xlsx 时序文件解析（Rust 加速 for io_utils/xlsx_io.py）。
//!
//! 专门处理 inlineStr 格式的 sheet：每行前两列为「时间字符串」+「数值字符串」，
//! 返回两个列向量，对应 Python `read_xlsx_raw_series` 中 `_iter_sheet_rows`
//! /`_iter_sheet_rows_fast` 的内部解析结果。Python 端继续负责时间戳解析、
//! 时区转换、重采样与合并。
//!
//! 使用 `zip` 读取 xlsx 容器、`quick-xml` 做 StAX-style XML 流式解析。

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::io::{BufRead, Read};

/// 提取 inlineStr 行：从单个 sheet 的 XML 字节流中提取所有行的前两列文本。
fn parse_inline_str_sheet<R: BufRead>(mut reader: Reader<R>) -> pyo3::PyResult<(Vec<Option<String>>, Vec<Option<String>>)> {
    let mut col_a: Vec<Option<String>> = Vec::new();
    let mut col_b: Vec<Option<String>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut cell_texts: Vec<Option<String>> = Vec::new(); // 当前 row 内所有 cell 文本
    let mut in_t: bool = false;
    let mut text_buf: Vec<u8> = Vec::new();

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"row" => cell_texts.clear(),
                    // 每个 cell 开始时清空缓冲；富文本 cell（多个 <r><t> run）
                    // 的所有 <t> 文本随后累积拼接，与 ElementTree 路径一致
                    b"c" => text_buf.clear(),
                    b"t" => in_t = true,
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"c" => {
                        // 当前 cell 完毕：把已收集的 text 推入 cell_texts
                        let text = if !text_buf.is_empty() {
                            let s = std::str::from_utf8(&text_buf).unwrap_or("");
                            let unescaped = quick_xml::escape::unescape(s)
                                .map(|c| c.into_owned())
                                .unwrap_or_else(|_| s.to_string());
                            Some(unescaped)
                        } else {
                            None
                        };
                        cell_texts.push(text);
                    }
                    b"t" => in_t = false,
                    b"row" => {
                        let a = cell_texts.get(0).cloned().flatten();
                        let b = cell_texts.get(1).cloned().flatten();
                        col_a.push(a);
                        col_b.push(b);
                        cell_texts.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_t {
                    text_buf.extend_from_slice(e.as_ref());
                }
            }
            Ok(Event::Empty(ref e)) => {
                // 空 cell（自闭合 `<c/>`）：不追加到 cell_texts，让 get(k) 返回 None
                if e.name().as_ref() == b"c" {
                    cell_texts.push(None);
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "xlsx XML parse error at byte {}: {}",
                    reader.buffer_position(),
                    e
                )));
            }
            _ => {}
        }
    }
    Ok((col_a, col_b))
}

/// Python-callable: `read_xlsx_sheet_inline_str_rs(path, sheet_path)`.
///
/// 打开 xlsx 文件，定位内部 sheet 路径（如 `xl/worksheets/sheet1.xml`），
/// 用 `quick-xml` StAX 解析器在 sheet XML 字节流上顺序枚举 events，
/// 提取 inlineStr 格式下每个 `<row>` 的前两个 `<c>` 单元格文本。
///
/// 返回 `(col_a: list[str|None], col_b: list[str|None])`，list 长度等于数据行数（含表头）。
#[pyfunction]
#[pyo3(name = "read_xlsx_sheet_inline_str_rs")]
pub fn read_xlsx_sheet_inline_str_rs<'py>(
    py: Python<'py>,
    path: &str,
    sheet_path: &str,
) -> pyo3::PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    // zip 读取与 XML 解析期间释放 GIL（pyo3 0.28 的 detach），使 Python
    // 侧线程池能对多个 sheet/文件真正并行解析；结果转 PyList 需要
    // GIL，在取回后进行。
    let path_owned = path.to_owned();
    let sheet_owned = sheet_path.to_owned();
    let (col_a, col_b) = py.detach(
        move || -> pyo3::PyResult<(Vec<Option<String>>, Vec<Option<String>>)> {
            let zipfile = std::fs::File::open(&path_owned).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("open {}: {}", path_owned, e))
            })?;
            let mut archive = zip::ZipArchive::new(zipfile).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("read zip {}: {}", path_owned, e))
            })?;
            let mut sheet_bytes: Vec<u8> = Vec::new();
            {
                let mut f = archive.by_name(&sheet_owned).map_err(|e| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "sheet {} not found: {}",
                        sheet_owned, e
                    ))
                })?;
                f.read_to_end(&mut sheet_bytes).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "read sheet {}: {}",
                        sheet_owned, e
                    ))
                })?;
            }
            let reader = Reader::from_reader(sheet_bytes.as_slice());
            parse_inline_str_sheet(reader)
        },
    )?;

    let list_a = to_py_opt_string_list(py, &col_a)?;
    let list_b = to_py_opt_string_list(py, &col_b)?;
    Ok((list_a, list_b))
}

fn to_py_opt_string_list<'py>(py: Python<'py>, items: &[Option<String>]) -> pyo3::PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for opt in items {
        match opt {
            Some(s) => {
                let pystr = PyString::new(py, s);
                list.append(pystr)?;
            }
            None => {
                list.append(py.None())?;
            }
        }
    }
    Ok(list)
}