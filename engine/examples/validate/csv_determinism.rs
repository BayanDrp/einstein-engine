//! V9 CSV schema and determinism: every visualization CSV producer runs
//! twice into isolated directories and must agree byte for byte.
//!
//! Covered producers: `geodesic_orbits` (three files), `coupled_dust` (two
//! files), and `matter_response` (one file). Schemas, minimum row counts,
//! monotonic time columns, finite numeric fields, stable text fields, and
//! byte equality follow VALIDATION.md, V9.
//!
//! The committed `visualization/data/schwarzschild` snapshot, which is the
//! real data the renderer consumes, is validated against the same schemas
//! and compared with the fresh `geodesic_orbits` output. The two fresh runs
//! must still be byte identical; the committed snapshot comparison uses the
//! documented cross-platform tolerance.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Relative tolerance (with an absolute floor below unit scale) for comparing
/// the committed visualization snapshot with a fresh run.
const FRESHNESS_TOLERANCE: f64 = 1.0e-5;

struct CsvSpec {
    name: &'static str,
    header: &'static str,
    /// Columns that must parse as finite `f64`; every other column is text.
    numeric: &'static [&'static str],
    min_rows: usize,
    /// Single monotonic time axis. Grouped by `series` when present.
    time: Option<&'static str>,
    series: Option<&'static str>,
}

struct Example {
    name: &'static str,
    /// `geodesic_orbits` takes its output directory as its only argument;
    /// the others write into their working directory.
    takes_output_dir: bool,
    files: &'static [CsvSpec],
}

const GEODESIC_FILES: &[CsvSpec] = &[
    CsvSpec {
        name: "geodesic_orbits_static.csv",
        header: "particle_id,label,revolutions,t,x,y,z,vx,vy,vz,r_areal,g_ww",
        numeric: &[
            "particle_id",
            "revolutions",
            "t",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "r_areal",
            "g_ww",
        ],
        min_rows: 4000,
        time: Some("t"),
        series: Some("label"),
    },
    CsvSpec {
        name: "geodesic_orbits_static_summary.csv",
        header: "label,revolutions,r_areal_target,omega,drift_areal_max,phase_err_max_rad,g_ww_drift_max,end_err,closure_full_rev,orbit_period_coord",
        numeric: &[
            "revolutions",
            "r_areal_target",
            "omega",
            "drift_areal_max",
            "phase_err_max_rad",
            "g_ww_drift_max",
            "end_err",
            "closure_full_rev",
            "orbit_period_coord",
        ],
        min_rows: 2,
        time: None,
        series: None,
    },
    CsvSpec {
        name: "geodesic_orbits_dynamic.csv",
        header: "particle_id,mode,t,x,y,z,vx,vy,vz,speed",
        numeric: &[
            "particle_id", "t", "x", "y", "z", "vx", "vy", "vz", "speed",
        ],
        min_rows: 2,
        time: Some("t"),
        series: None,
    },
];

const COUPLED_DUST_FILES: &[CsvSpec] = &[
    CsvSpec {
        name: "coupled_dust.csv",
        header: "step,t,dt,min_alpha,max_abs_K,rho_mean,rho_min,max_abs_v,Hmax,Mmax,mass_rel_err",
        numeric: &[
            "step",
            "t",
            "dt",
            "min_alpha",
            "max_abs_K",
            "rho_mean",
            "rho_min",
            "max_abs_v",
            "Hmax",
            "Mmax",
            "mass_rel_err",
        ],
        min_rows: 17,
        time: Some("t"),
        series: None,
    },
    CsvSpec {
        name: "coupled_dust_transport.csv",
        header: "t,profile_err,mass_rel_err,Hmax,max_abs_v",
        numeric: &["t", "profile_err", "mass_rel_err", "Hmax", "max_abs_v"],
        min_rows: 1,
        time: Some("t"),
        series: None,
    },
];

const MATTER_RESPONSE_FILES: &[CsvSpec] = &[CsvSpec {
    name: "matter_response.csv",
    header: "step,t,min_alpha,max_abs_K,Hmax,Mmax",
    numeric: &["step", "t", "min_alpha", "max_abs_K", "Hmax", "Mmax"],
    min_rows: 17,
    time: Some("t"),
    series: None,
}];

const EXAMPLES: &[Example] = &[
    Example {
        name: "geodesic_orbits",
        takes_output_dir: true,
        files: GEODESIC_FILES,
    },
    Example {
        name: "coupled_dust",
        takes_output_dir: false,
        files: COUPLED_DUST_FILES,
    },
    Example {
        name: "matter_response",
        takes_output_dir: false,
        files: MATTER_RESPONSE_FILES,
    },
];

pub fn run() -> Result<(), String> {
    let base = std::env::temp_dir().join(format!("einstein_validate_v9_{}", std::process::id()));
    let _ = fs::remove_dir_all(&base);

    let dirs = [base.join("a"), base.join("b")];
    for dir in &dirs {
        fs::create_dir_all(dir)
            .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    }

    build_examples()?;

    for example in EXAMPLES {
        run_pair(example, &dirs)?;
    }

    let mut total_rows = 0usize;
    for example in EXAMPLES {
        for spec in example.files {
            let a = fs::read(dirs[0].join(spec.name))
                .map_err(|error| format!("{}: run a: {error}", spec.name))?;
            let b = fs::read(dirs[1].join(spec.name))
                .map_err(|error| format!("{}: run b: {error}", spec.name))?;

            if a != b {
                return Err(format!(
                    "{}: two runs are not byte identical ({} vs {} bytes)",
                    spec.name,
                    a.len(),
                    b.len()
                ));
            }

            total_rows += validate_csv(spec, &a)?;
        }
    }

    validate_visualization_data(&dirs[0])?;

    let _ = fs::remove_dir_all(&base);

    println!(
        "V9 CSV determinism: {total_rows} rows across six files, byte identical on both runs; committed visualization data matches the fresh geodesic run"
    );

    Ok(())
}

/// Build the three producer examples in the profile that is running this
/// validation binary, so both runs execute the same fresh executable.
fn build_examples() -> Result<(), String> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");

    let mut command = Command::new(cargo());
    command
        .arg("build")
        .arg("--quiet")
        .arg("--manifest-path")
        .arg(&manifest)
        .current_dir(env!("CARGO_MANIFEST_DIR"));

    match profile() {
        Some(profile) if profile == "release" => {
            command.arg("--release");
        }
        Some(profile) if profile != "debug" => {
            command.arg("--profile").arg(profile);
        }
        _ => {}
    }

    for example in EXAMPLES {
        command.arg("--example").arg(example.name);
    }

    let output = command
        .output()
        .map_err(|error| format!("cannot spawn cargo build: {error}"))?;

    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo build failed: {}",
            child_details(&output.stdout, &output.stderr)
        ))
    }
}

/// Run the same producer twice concurrently. Both processes are
/// single-threaded and write to separate directories, so the outputs are
/// independent of scheduling; running them together only shortens wall time.
fn run_pair(example: &Example, dirs: &[PathBuf; 2]) -> Result<(), String> {
    let binary = example_binary(example.name)?;

    let mut children = Vec::with_capacity(2);
    for (label, dir) in ["a", "b"].iter().zip(dirs) {
        let mut command = Command::new(&binary);
        command
            .current_dir(dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if example.takes_output_dir {
            command.arg(dir);
        }

        let child = command
            .spawn()
            .map_err(|error| format!("cannot spawn {} (run {label}): {error}", example.name))?;
        children.push((*label, child));
    }

    for (label, child) in children {
        let output = child
            .wait_with_output()
            .map_err(|error| format!("cannot wait for {} (run {label}): {error}", example.name))?;

        if !output.status.success() {
            return Err(format!(
                "{} (run {label}, {}) failed: {}",
                example.name,
                output.status,
                child_details(&output.stdout, &output.stderr)
            ));
        }
    }

    Ok(())
}

fn example_binary(name: &str) -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .map_err(|error| format!("cannot locate validation binary: {error}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| "validation binary has no parent directory".to_string())?;
    let candidate = dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));

    if candidate.is_file() {
        Ok(candidate)
    } else {
        Err(format!(
            "producer binary {} is missing after cargo build",
            candidate.display()
        ))
    }
}

/// Name of the profile directory containing this binary: `release`,
/// `debug`, or a custom profile name. `None` for an unusual layout.
fn profile() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let examples_dir = exe.parent()?;
    let profile_dir = examples_dir.parent()?;
    Some(profile_dir.file_name()?.to_string_lossy().into_owned())
}

fn cargo() -> std::ffi::OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}

fn child_details(stdout: &[u8], stderr: &[u8]) -> String {
    let mut details = String::new();
    details.push_str(&String::from_utf8_lossy(stdout));
    details.push_str(&String::from_utf8_lossy(stderr));
    let trimmed = details.trim();
    let text: String = trimmed.chars().take(1500).collect();
    if text.is_empty() {
        "(no output)".to_string()
    } else {
        text
    }
}

fn validate_csv(spec: &CsvSpec, bytes: &[u8]) -> Result<usize, String> {
    if !bytes.is_ascii() {
        return Err(format!("{}: output contains non-ASCII bytes", spec.name));
    }

    let text = std::str::from_utf8(bytes).map_err(|error| format!("{}: {error}", spec.name))?;
    let mut lines = text.lines();

    let header = lines
        .next()
        .ok_or_else(|| format!("{}: missing header", spec.name))?;
    if header != spec.header {
        return Err(format!(
            "{}: header mismatch\n  expected: {}\n  actual:   {}",
            spec.name, spec.header, header
        ));
    }

    let columns: Vec<&str> = header.split(',').collect();
    let index = |name: &str| {
        columns
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| format!("{}: header lacks column '{name}'", spec.name))
    };

    let numeric: Vec<usize> = spec
        .numeric
        .iter()
        .map(|name| index(name))
        .collect::<Result<_, _>>()?;
    let time_col = spec.time.map(index).transpose()?;
    let series_col = spec.series.map(index).transpose()?;

    let mut last_time: HashMap<String, f64> = HashMap::new();
    let mut rows = 0usize;

    for (offset, line) in lines.enumerate() {
        let line_no = offset + 2;
        if line.is_empty() {
            return Err(format!("{}:{line_no}: empty row", spec.name));
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != columns.len() {
            return Err(format!(
                "{}:{line_no}: expected {} fields, got {}",
                spec.name,
                columns.len(),
                fields.len()
            ));
        }

        for (col, field) in fields.iter().enumerate() {
            if numeric.contains(&col) {
                let value: f64 = field.parse().map_err(|_| {
                    format!(
                        "{}:{line_no}: column '{}' is not numeric: '{field}'",
                        spec.name, columns[col]
                    )
                })?;

                if !value.is_finite() {
                    return Err(format!(
                        "{}:{line_no}: column '{}' is not finite: '{field}'",
                        spec.name, columns[col]
                    ));
                }

                if Some(col) == time_col {
                    let key = series_col
                        .map(|series| fields[series].to_string())
                        .unwrap_or_default();
                    if let Some(previous) = last_time.get(&key) {
                        if value < *previous {
                            return Err(format!(
                                "{}:{line_no}: time went backwards ({value} after {previous})",
                                spec.name
                            ));
                        }
                    }
                    last_time.insert(key, value);
                }
            } else {
                check_text(spec.name, line_no, columns[col], field)?;
            }
        }

        rows += 1;
    }

    if rows < spec.min_rows {
        return Err(format!(
            "{}: {} data rows, minimum is {}",
            spec.name, rows, spec.min_rows
        ));
    }

    Ok(rows)
}

/// Text fields must be stable identifiers. That rules out absolute paths
/// (no separators), timestamps (no colon, no leading digit), and locale
/// dependent text (ASCII only, no spaces).
fn check_text(file: &str, line_no: usize, column: &str, field: &str) -> Result<(), String> {
    let mut chars = field.chars();
    let starts_ok = matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_');
    let body_ok = field
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');

    if starts_ok && body_ok {
        Ok(())
    } else {
        Err(format!(
            "{file}:{line_no}: column '{column}' is not a stable text field: '{field}'"
        ))
    }
}

fn validate_visualization_data(fresh_dir: &Path) -> Result<(), String> {
    let snapshot_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../visualization/data/schwarzschild");

    for spec in GEODESIC_FILES {
        let snapshot_path = snapshot_dir.join(spec.name);
        let snapshot = fs::read(&snapshot_path)
            .map_err(|error| format!("{}: cannot read committed snapshot: {error}", spec.name))?;
        validate_csv(spec, &snapshot)?;

        let fresh_path = fresh_dir.join(spec.name);
        let fresh = fs::read(&fresh_path)
            .map_err(|error| format!("{}: cannot read fresh output: {error}", spec.name))?;
        compare_csv_freshness(spec, &snapshot, &fresh)?;
    }

    Ok(())
}

fn compare_csv_freshness(spec: &CsvSpec, snapshot: &[u8], fresh: &[u8]) -> Result<(), String> {
    let snapshot_text = std::str::from_utf8(snapshot)
        .map_err(|error| format!("{}: snapshot is not UTF-8: {error}", spec.name))?;
    let fresh_text = std::str::from_utf8(fresh)
        .map_err(|error| format!("{}: fresh output is not UTF-8: {error}", spec.name))?;

    let snapshot_rows: Vec<&str> = snapshot_text.lines().skip(1).collect();
    let fresh_rows: Vec<&str> = fresh_text.lines().skip(1).collect();
    if snapshot_rows.len() != fresh_rows.len() {
        return Err(format!(
            "{}: committed snapshot has {} rows, fresh output has {}",
            spec.name,
            snapshot_rows.len(),
            fresh_rows.len()
        ));
    }

    let numeric_columns: Vec<usize> = spec
        .numeric
        .iter()
        .map(|name| {
            spec.header
                .split(',')
                .position(|column| column == *name)
                .ok_or_else(|| format!("{}: header lacks numeric column '{name}'", spec.name))
        })
        .collect::<Result<_, _>>()?;

    for (offset, (snapshot_row, fresh_row)) in snapshot_rows.iter().zip(&fresh_rows).enumerate() {
        let line = offset + 2;
        let snapshot_fields: Vec<&str> = snapshot_row.split(',').collect();
        let fresh_fields: Vec<&str> = fresh_row.split(',').collect();

        for column in 0..snapshot_fields.len() {
            if numeric_columns.contains(&column) {
                let old: f64 = snapshot_fields[column].parse().map_err(|_| {
                    format!(
                        "{}:{line}: snapshot column {} is not numeric",
                        spec.name, column
                    )
                })?;
                let new: f64 = fresh_fields[column].parse().map_err(|_| {
                    format!(
                        "{}:{line}: fresh column {} is not numeric",
                        spec.name, column
                    )
                })?;
                let tolerance = FRESHNESS_TOLERANCE * old.abs().max(new.abs()).max(1.0);
                if (old - new).abs() > tolerance {
                    return Err(format!(
                        "{}:{line}: column {} differs from snapshot: {old} vs {new} (tolerance {tolerance})",
                        spec.name, column
                    ));
                }
            } else if snapshot_fields[column] != fresh_fields[column] {
                return Err(format!(
                    "{}:{line}: text column {} differs from snapshot: '{}' vs '{}'",
                    spec.name, column, snapshot_fields[column], fresh_fields[column]
                ));
            }
        }
    }

    Ok(())
}
