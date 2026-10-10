//! MT 采集仪空白分隔 TXT 时序文件解析。
//!
//! 格式沿用 MT Portable：前 6 列为年月日时分秒，7--9 列为
//! Hx/Hy/Hz，12--15 列为 Ex1/Ey1/Ex2/Ey2。若存在 18--21 列，
//! 则按 NMEA 度分格式解析逐行纬度、南北标记、经度和东西标记。

use chrono::{NaiveDate, NaiveDateTime};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug)]
struct MtTimeSeries {
    start_time: NaiveDateTime,
    end_time: NaiveDateTime,
    hx: Vec<f64>,
    hy: Vec<f64>,
    hz: Vec<f64>,
    ex1: Vec<f64>,
    ey1: Vec<f64>,
    ex2: Vec<f64>,
    ey2: Vec<f64>,
    detected_dt_seconds: Option<f64>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    gps_samples: usize,
}

#[derive(Debug, Error)]
enum MtTextError {
    #[error("读取第{line_number}行失败：{message}")]
    ReadLine { line_number: usize, message: String },
    #[error("文件为空")]
    EmptyFile,
    #[error("第{line_number}行只有{actual}列，至少需要{required}列")]
    TooFewColumns {
        line_number: usize,
        actual: usize,
        required: usize,
    },
    #[error("第{line_number}行的{name}无法解析：{value}（{message}）")]
    InvalidField {
        line_number: usize,
        name: String,
        value: String,
        message: String,
    },
    #[error("第{line_number}行日期无效")]
    InvalidDate { line_number: usize },
    #[error("第{line_number}行时间无效")]
    InvalidTime { line_number: usize },
    #[error("第{line_number}行时间没有递增")]
    NonIncreasingTime { line_number: usize },
    #[error("第{line_number}行的采样间隔为{actual_seconds}秒，预期为{expected_seconds}秒")]
    IrregularSampling {
        line_number: usize,
        expected_seconds: i64,
        actual_seconds: i64,
    },
}

/// Python-callable: `read_mt_txt_rs(path)`.
///
/// 文件读取与数值解析期间释放 GIL；结果中的七个通道以 NumPy 数组返回。
#[pyfunction]
#[pyo3(name = "read_mt_txt_rs")]
pub fn read_mt_txt_rs<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let path_owned = path.to_owned();
    let series = py.detach(move || -> PyResult<MtTimeSeries> {
        let file = File::open(&path_owned).map_err(|error| {
            pyo3::exceptions::PyIOError::new_err(format!("无法打开 {}：{}", path_owned, error))
        })?;
        read_mt_time_series(file).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("解析 {} 失败：{}", path_owned, error))
        })
    })?;

    let result = PyDict::new(py);
    result.set_item(
        "start_time",
        series.start_time.format("%Y-%m-%d %H:%M:%S").to_string(),
    )?;
    result.set_item(
        "end_time",
        series.end_time.format("%Y-%m-%d %H:%M:%S").to_string(),
    )?;
    result.set_item("dt_seconds", series.detected_dt_seconds)?;
    result.set_item("latitude", series.latitude)?;
    result.set_item("longitude", series.longitude)?;
    result.set_item("gps_samples", series.gps_samples)?;
    result.set_item("row_count", series.hx.len())?;
    result.set_item("Hx", series.hx.into_pyarray(py))?;
    result.set_item("Hy", series.hy.into_pyarray(py))?;
    result.set_item("Hz", series.hz.into_pyarray(py))?;
    result.set_item("Ex1", series.ex1.into_pyarray(py))?;
    result.set_item("Ey1", series.ey1.into_pyarray(py))?;
    result.set_item("Ex2", series.ex2.into_pyarray(py))?;
    result.set_item("Ey2", series.ey2.into_pyarray(py))?;
    Ok(result)
}

fn read_mt_time_series<R: Read>(reader: R) -> Result<MtTimeSeries, MtTextError> {
    let reader = BufReader::new(reader);
    let mut hx = Vec::new();
    let mut hy = Vec::new();
    let mut hz = Vec::new();
    let mut ex1 = Vec::new();
    let mut ey1 = Vec::new();
    let mut ex2 = Vec::new();
    let mut ey2 = Vec::new();
    let mut latitudes = Vec::new();
    let mut longitudes = Vec::new();
    let mut first_timestamp = None;
    let mut previous_timestamp = None;
    let mut detected_delta_seconds = None;

    for (index, line_result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line_result.map_err(|error| MtTextError::ReadLine {
            line_number,
            message: error.to_string(),
        })?;
        let sample = parse_mt_line(&line, line_number)?;

        if let Some(previous) = previous_timestamp {
            let delta_seconds = sample
                .timestamp
                .signed_duration_since(previous)
                .num_seconds();
            if delta_seconds <= 0 {
                return Err(MtTextError::NonIncreasingTime { line_number });
            }
            if let Some(expected_seconds) = detected_delta_seconds {
                if delta_seconds != expected_seconds {
                    return Err(MtTextError::IrregularSampling {
                        line_number,
                        expected_seconds,
                        actual_seconds: delta_seconds,
                    });
                }
            } else {
                detected_delta_seconds = Some(delta_seconds);
            }
        } else {
            first_timestamp = Some(sample.timestamp);
        }

        hx.push(sample.hx);
        hy.push(sample.hy);
        hz.push(sample.hz);
        ex1.push(sample.ex1);
        ey1.push(sample.ey1);
        ex2.push(sample.ex2);
        ey2.push(sample.ey2);
        if let Some((latitude, longitude)) = sample.gps {
            latitudes.push(latitude);
            longitudes.push(longitude);
        }
        previous_timestamp = Some(sample.timestamp);
    }

    let start_time = first_timestamp.ok_or(MtTextError::EmptyFile)?;
    let end_time = previous_timestamp.expect("non-empty series has an end timestamp");
    let gps_samples = latitudes.len();
    Ok(MtTimeSeries {
        start_time,
        end_time,
        hx,
        hy,
        hz,
        ex1,
        ey1,
        ex2,
        ey2,
        detected_dt_seconds: detected_delta_seconds.map(|value| value as f64),
        latitude: median(&mut latitudes),
        longitude: median(&mut longitudes),
        gps_samples,
    })
}

struct ParsedSample {
    timestamp: NaiveDateTime,
    hx: f64,
    hy: f64,
    hz: f64,
    ex1: f64,
    ey1: f64,
    ex2: f64,
    ey2: f64,
    gps: Option<(f64, f64)>,
}

fn parse_mt_line(line: &str, line_number: usize) -> Result<ParsedSample, MtTextError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 15 {
        return Err(MtTextError::TooFewColumns {
            line_number,
            actual: fields.len(),
            required: 15,
        });
    }

    let year: i32 = parse_field(&fields, 0, line_number, "年")?;
    let month: u32 = parse_field(&fields, 1, line_number, "月")?;
    let day: u32 = parse_field(&fields, 2, line_number, "日")?;
    let hour: u32 = parse_field(&fields, 3, line_number, "时")?;
    let minute: u32 = parse_field(&fields, 4, line_number, "分")?;
    let second: u32 = parse_field(&fields, 5, line_number, "秒")?;
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or(MtTextError::InvalidDate { line_number })?;
    let timestamp = date
        .and_hms_opt(hour, minute, second)
        .ok_or(MtTextError::InvalidTime { line_number })?;

    let gps = if fields.len() >= 21 {
        let latitude_raw = parse_finite_f64(&fields, 17, line_number, "纬度")?;
        let longitude_raw = parse_finite_f64(&fields, 19, line_number, "经度")?;
        Some((
            parse_nmea_coordinate(latitude_raw, fields[18], true, line_number)?,
            parse_nmea_coordinate(longitude_raw, fields[20], false, line_number)?,
        ))
    } else {
        None
    };

    Ok(ParsedSample {
        timestamp,
        hx: parse_finite_f64(&fields, 6, line_number, "Hx")?,
        hy: parse_finite_f64(&fields, 7, line_number, "Hy")?,
        hz: parse_finite_f64(&fields, 8, line_number, "Hz")?,
        ex1: parse_finite_f64(&fields, 11, line_number, "Ex1")?,
        ey1: parse_finite_f64(&fields, 12, line_number, "Ey1")?,
        ex2: parse_finite_f64(&fields, 13, line_number, "Ex2")?,
        ey2: parse_finite_f64(&fields, 14, line_number, "Ey2")?,
        gps,
    })
}

fn parse_nmea_coordinate(
    raw: f64,
    hemisphere: &str,
    is_latitude: bool,
    line_number: usize,
) -> Result<f64, MtTextError> {
    let degrees = (raw / 100.0).floor();
    let minutes = raw - degrees * 100.0;
    let max_degrees = if is_latitude { 90.0 } else { 180.0 };
    let allowed = if is_latitude { "N 或 S" } else { "E 或 W" };
    let sign = match hemisphere.to_ascii_uppercase().as_str() {
        "N" if is_latitude => 1.0,
        "S" if is_latitude => -1.0,
        "E" if !is_latitude => 1.0,
        "W" if !is_latitude => -1.0,
        _ => {
            return Err(MtTextError::InvalidField {
                line_number,
                name: if is_latitude {
                    "南北标记"
                } else {
                    "东西标记"
                }
                .to_owned(),
                value: hemisphere.to_owned(),
                message: format!("必须是{allowed}"),
            });
        }
    };
    if raw < 0.0 || minutes >= 60.0 || degrees > max_degrees {
        return Err(MtTextError::InvalidField {
            line_number,
            name: if is_latitude { "纬度" } else { "经度" }.to_owned(),
            value: raw.to_string(),
            message: "不是有效的 NMEA 度分坐标".to_owned(),
        });
    }
    Ok(sign * (degrees + minutes / 60.0))
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

fn parse_field<T>(
    fields: &[&str],
    index: usize,
    line_number: usize,
    name: &str,
) -> Result<T, MtTextError>
where
    T: FromStr,
    T::Err: Display,
{
    fields[index]
        .parse::<T>()
        .map_err(|error| MtTextError::InvalidField {
            line_number,
            name: name.to_owned(),
            value: fields[index].to_owned(),
            message: error.to_string(),
        })
}

fn parse_finite_f64(
    fields: &[&str],
    index: usize,
    line_number: usize,
    name: &str,
) -> Result<f64, MtTextError> {
    let value: f64 = parse_field(fields, index, line_number, name)?;
    if !value.is_finite() {
        return Err(MtTextError::InvalidField {
            line_number,
            name: name.to_owned(),
            value: fields[index].to_owned(),
            message: "必须是有限数值".to_owned(),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW_1: &str = concat!(
        "2026 05 31 03 55 00 26181.621 -61.059 48518.992 ",
        "43.36 201.69 -19.898 -29.367 -4.446 -123.712 ",
        "12.98 1009.1 4023.669619 N 08120.624305 E 8 2 0\n",
    );
    const ROW_2: &str = concat!(
        "2026 05 31 03 55 01 26181.604 -61.084 48519.020 ",
        "43.40 201.69 -19.903 -29.370 -4.452 -123.712 ",
        "12.98 1009.1 4023.669619 N 08120.624305 E 8 2 0\n",
    );

    #[test]
    fn parses_all_channels_dt_and_gps() {
        let input = format!("{ROW_1}{ROW_2}");
        let series = read_mt_time_series(input.as_bytes()).unwrap();

        assert_eq!(series.hx.len(), 2);
        assert_eq!(series.detected_dt_seconds, Some(1.0));
        assert!((series.ex1[0] - (-19.898)).abs() < 1e-9);
        assert!((series.ey2[0] - (-123.712)).abs() < 1e-9);
        assert!((series.latitude.unwrap() - 40.39449365).abs() < 1e-8);
        assert!((series.longitude.unwrap() - 81.3437384167).abs() < 1e-8);
        assert_eq!(series.gps_samples, 2);
    }

    #[test]
    fn accepts_core_format_without_gps_columns() {
        let row = "2026 05 31 03 55 00 1 2 3 4 5 6 7 8 9\n";
        let series = read_mt_time_series(row.as_bytes()).unwrap();
        assert_eq!(series.latitude, None);
        assert_eq!(series.longitude, None);
    }

    #[test]
    fn rejects_irregular_timestamps() {
        let row_3 = ROW_2.replacen("03 55 01", "03 55 03", 1);
        let input = format!("{ROW_1}{ROW_2}{row_3}");
        let error = read_mt_time_series(input.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            MtTextError::IrregularSampling {
                line_number: 3,
                expected_seconds: 1,
                actual_seconds: 2,
            }
        ));
    }
}
