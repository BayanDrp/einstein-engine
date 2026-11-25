//! Validation scenarios for the 3+1 release gate.
//!
//! Implemented scenarios run first; V3 remains unwired.

mod bssn_boundedness;
mod csv_determinism;
mod dust_conservation;
mod flrw_dust;
mod geodesics;
mod linear_wave;
mod minkowski;
mod schwarzschild;
mod stage_coupling;
mod webgl;

use std::process::ExitCode;

/// Run every validation scenario and return a process exit code.
///
/// This deliberately fails until every scenario is implemented, so an
/// unfinished suite can never be mistaken for a passing release gate.
pub fn run() -> ExitCode {
    match minkowski::run() {
        Ok(()) => println!("V1 Minkowski: PASS"),
        Err(error) => eprintln!("V1 Minkowski: FAIL: {error}"),
    }

    match linear_wave::run() {
        Ok(()) => println!("V2 Linear wave: PASS"),
        Err(error) => eprintln!("V2 Linear wave: FAIL: {error}"),
    }

    match geodesics::run() {
        Ok(()) => println!("V4 Geodesics: PASS"),
        Err(error) => eprintln!("V4 Geodesics: FAIL: {error}"),
    }

    match flrw_dust::run() {
        Ok(()) => println!("V5 FLRW dust: PASS"),
        Err(error) => eprintln!("V5 FLRW dust: FAIL: {error}"),
    }

    match dust_conservation::run() {
        Ok(()) => println!("V6 Dust conservation: PASS"),
        Err(error) => eprintln!("V6 Dust conservation: FAIL: {error}"),
    }

    match bssn_boundedness::run() {
        Ok(()) => println!("V7 BSSN boundedness: PASS"),
        Err(error) => eprintln!("V7 BSSN boundedness: FAIL: {error}"),
    }

    match stage_coupling::run() {
        Ok(()) => println!("V8 Stage coupling: PASS"),
        Err(error) => eprintln!("V8 Stage coupling: FAIL: {error}"),
    }

    match csv_determinism::run() {
        Ok(()) => println!("V9 CSV determinism: PASS"),
        Err(error) => eprintln!("V9 CSV determinism: FAIL: {error}"),
    }

    match webgl::run() {
        Ok(()) => println!("V10 WebGL renderer: PASS"),
        Err(error) => eprintln!("V10 WebGL renderer: FAIL: {error}"),
    }

    eprintln!("validation scenario V3 is not implemented yet");
    ExitCode::FAILURE
}
