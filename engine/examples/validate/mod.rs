//! Validation scenarios for the 3+1 release gate.

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
/// A failed scenario makes the release gate fail after every scenario has run.
pub fn run() -> ExitCode {
    let mut all_passed = true;
    all_passed &= report("V1 Minkowski", minkowski::run());
    all_passed &= report("V2 Linear wave", linear_wave::run());
    all_passed &= report("V3 Schwarzschild", schwarzschild::run());
    all_passed &= report("V4 Geodesics", geodesics::run());
    all_passed &= report("V5 FLRW dust", flrw_dust::run());
    all_passed &= report("V6 Dust conservation", dust_conservation::run());
    all_passed &= report("V7 BSSN boundedness", bssn_boundedness::run());
    all_passed &= report("V8 Stage coupling", stage_coupling::run());
    all_passed &= report("V9 CSV determinism", csv_determinism::run());
    all_passed &= report("V10 WebGL renderer", webgl::run());

    if all_passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn report(scenario: &str, outcome: Result<(), String>) -> bool {
    match outcome {
        Ok(()) => {
            println!("{scenario}: PASS");
            true
        }
        Err(error) => {
            eprintln!("{scenario}: FAIL: {error}");
            false
        }
    }
}
